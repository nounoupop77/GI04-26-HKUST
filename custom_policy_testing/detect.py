from __future__ import annotations
import os
import csv
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONSTANT
# ---------------------------------------------------------------------------
IMG_SIZE            = (224, 224)
COLOR_DIST_GAIN     = 2.0
MIN_BLOCK_AREA      = 30
MAX_AREA_FRAC       = 0.20
WATERSHED_SEED_FRAC = 0.25
MEDIAN_K            = 41

S_MIN = 60
C_MIN = 18
V_MIN = 30
V_MAX = 245

MAX_TARGET_R    = 80.0             
CROSS_IGNORE_K  = 1.35             
# ===========================================================================


# ---------------------------------------------------------------------------
# detect target position  (VERBATIM from user)
# ---------------------------------------------------------------------------
def detect_cross_by_shape(frame0_bgr):
    h, w = frame0_bgr.shape[:2]

    lab = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2LAB)
    l_chan, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g0 = cv2.GaussianBlur(clahe.apply(l_chan), (3, 3), 0)

    dark_adapt = cv2.adaptiveThreshold(
        g0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=16
    )
    blackhat = cv2.morphologyEx(g0, cv2.MORPH_BLACKHAT, np.ones((15, 15), np.uint8))
    _, dark_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark0 = cv2.bitwise_and(dark_adapt, dark_bh)
    dark0 = cv2.morphologyEx(dark0, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    mx, my = int(w * 0.04), int(h * 0.04)
    dark0[:my, :] = 0
    dark0[h - my:, :] = 0
    dark0[:, :mx] = 0
    dark0[:, w - mx:] = 0

    contours, _ = cv2.findContours(dark0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate, best_score, best_r = None, -1.0, 12.0

    for c in contours:
        area = cv2.contourArea(c)
        if not (25 <= area <= 1200):
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(max(1, bh))
        if not (0.65 <= aspect <= 1.50):
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area == 0:
            continue

        solidity = area / hull_area
        if not (0.22 <= solidity <= 0.66):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if not (7 <= len(approx) <= 16):
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        mask_c = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        ring = cv2.subtract(
            cv2.dilate(mask_c, np.ones((9, 9), np.uint8)),
            cv2.dilate(mask_c, np.ones((3, 3), np.uint8)),
        )
        fg_mean = cv2.mean(g0, mask=mask_c)[0]
        bg_mean = cv2.mean(g0, mask=ring)[0] if cv2.countNonZero(ring) else 255.0
        contrast = bg_mean - fg_mean
        if contrast < 20.0:
            continue

        shape_score = (1.0 - abs(solidity - 0.45)) * 100.0 + contrast
        if shape_score > best_score:
            best_score = shape_score
            best_candidate = (cx, cy)
            best_r = 0.5 * max(bw, bh)

    if best_candidate is not None:
        return best_candidate[0], best_candidate[1], best_r, True
    return 0, 0, 12.0, False


def colored_mask(bgr):
    """Keep only genuinely colored pixels (not black/white/gray)."""
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)

    #check from hsv
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    m_hsv = ((S >= S_MIN) & (V >= V_MIN) & (V <= V_MAX)).astype(np.uint8) * 255

    #check from lab
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    _, a, b = cv2.split(lab)
    chroma = np.hypot(a - 128.0, b - 128.0)
    m_lab = (chroma >= C_MIN).astype(np.uint8) * 255

    #morph close and open to clean
    mask = cv2.bitwise_and(m_hsv, m_lab)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return mask


def segment_blocks(bgr, mask, cross_xy = None):
    """Watershed-split the colored mask into per-block boxes."""
    """Output: {area,cx,cy,quad,score}"""
    h, w = mask.shape[:2]

    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    pos = dist[dist > 0]
    seed_thr = max(2.0, WATERSHED_SEED_FRAC * float(np.percentile(pos, 75)))
    _, sure_fg = cv2.threshold(dist, seed_thr, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(m, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.GaussianBlur(bgr, (5, 5), 0), markers)

    max_area = MAX_AREA_FRAC * h * w
    boxes = []

    for label_id in range(2, num_labels + 1):
        comp = np.uint8(markers == label_id) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < MIN_BLOCK_AREA or area > max_area:
            continue

        rect = cv2.minAreaRect(c)
        quad = np.int32(np.round(cv2.boxPoints(rect))).reshape(-1, 1, 2)

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if cross_xy is not None:
            score = -float(np.hypot(cx-cross_xy[0],cy-cross_xy[1]))
        else:
            score = float(area)
        # score = area
        boxes.append(dict(area=float(area), cx=cx, cy=cy, quad=quad,
                          score=score))

    return boxes


def run_frame0_pipeline(frame_bgr, label=""):
    """check from frame0"""
    frame = cv2.resize(frame_bgr, IMG_SIZE)
    vis = frame.copy()

    # Detect the cross (target) 
    tx, ty, cross_r, cross_found = detect_cross_by_shape(frame)
    locked_cross = (tx, ty) if cross_found else None

    # Segment all colored blocks 
    cmask  = colored_mask(frame)
    blocks = segment_blocks(frame, cmask, cross_xy = locked_cross)

    # Drop block if that is < MIN_BLOCK_AREA
    blocks = [b for b in blocks if b["area"] >= MIN_BLOCK_AREA]

    # Draw cross + blocks for the debug image ---------------------
    if locked_cross is not None:
        cx_t, cy_t = locked_cross
        r = int(round(cross_r))
        cv2.circle(vis, (cx_t, cy_t), r, (0, 0, 255), 1)
        cv2.drawMarker(vis, (cx_t, cy_t), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(
            vis, f"TARGET ({cx_t},{cy_t}) r={r}",
            (cx_t + 5, max(10, cy_t - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
        )

    for i, b in enumerate(blocks, 1):
        cv2.drawContours(vis, [b["quad"]], -1, (0, 255, 0), 1)
        cv2.putText(
            vis, f"#{i} ({b['cx']},{b['cy']})",
            (b["cx"] - 15, max(10, b["cy"] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1,
        )

    hud = f"{label} blocks={len(blocks)} target={'OK' if locked_cross else 'NO'}"
    cv2.rectangle(vis, (0, 0), (IMG_SIZE[0], 18), (20, 20, 20), -1)
    cv2.putText(vis, hud, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

    return vis, locked_cross, blocks


# ---------------------------------------------------------------------------
# === Canonical wrapper used by the model ===
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    cross_xy: tuple | None            # (x, y) in 224x224 pixels or None
    block_xy: tuple | None            # (x, y) of chosen block or None
    blocks: list = field(default_factory=list)
    vis: np.ndarray | None = None     # annotated BGR image (224x224)

    def to_vector(self) -> np.ndarray:
        """4-dim vector in [0, 1]: block_x, block_y, target_x, target_y.
        Missing entries -> 0.0; use presence_mask() to distinguish."""
        bx, by = self.block_xy if self.block_xy is not None else (0.0, 0.0)
        tx, ty = self.cross_xy if self.cross_xy is not None else (0.0, 0.0)
        W, H = IMG_SIZE
        return np.array([bx / W, by / H, tx / W, ty / H], dtype=np.float32)

    def presence_mask(self) -> np.ndarray:
        return np.array(
            [1.0 if self.block_xy is not None else 0.0,
             1.0 if self.block_xy is not None else 0.0,
             1.0 if self.cross_xy is not None else 0.0,
             1.0 if self.cross_xy is not None else 0.0],
            dtype=np.float32,
        )

    def summary(self) -> dict:
        return {
            "cross_xy": list(self.cross_xy) if self.cross_xy is not None else None,
            "block_xy": list(self.block_xy) if self.block_xy is not None else None,
            "n_blocks": len(self.blocks),
            "blocks": [
                {"cx": b["cx"], "cy": b["cy"], "area": float(b["area"]),
                 "score": float(b["score"])}
                for b in self.blocks
            ],
        }


def detect_frame0(frame_bgr) -> DetectionResult:
    """Returns block + cross coordinates in 224x224 space + annotated visualization."""
    vis, cross, blocks = run_frame0_pipeline(frame_bgr, label="detect")

    # choose the box closest to the target, if no target then choose the largest box
    chosen = max(blocks, key=lambda b: b["score"]) if blocks else None

    return DetectionResult(
        cross_xy=tuple(cross) if cross is not None else None,
        block_xy=(chosen["cx"], chosen["cy"]) if chosen is not None else None,
        blocks=blocks,
        vis=vis,
    )


# ---------------------------------------------------------------------------
# === Persist detection results for the whole dataset ===
# ---------------------------------------------------------------------------
def write_detection_summary(out_dir: str, all_detections: dict):
    """all_detections: {episode_key: DetectionResult}
    Writes:  <out_dir>/detections.csv
             <out_dir>/detections.json
             <out_dir>/frame0_<episode_key>.jpg
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "detections.csv")
    json_path = os.path.join(out_dir, "detections.json")

    rows, json_dump = [], {}
    for key, det in all_detections.items():
        s = det.summary()
        s["key"] = key
        json_dump[key] = s
        rows.append({
            "episode": key,
            "block_x": det.block_xy[0] if det.block_xy else "",
            "block_y": det.block_xy[1] if det.block_xy else "",
            "target_x": det.cross_xy[0] if det.cross_xy else "",
            "target_y": det.cross_xy[1] if det.cross_xy else "",
            "n_blocks": len(det.blocks),
            "vis_path": f"frame0_{key}.jpg" if det.vis is not None else "",
        })
        if det.vis is not None:
            cv2.imwrite(os.path.join(out_dir, f"frame0_{key}.jpg"), det.vis)

    fieldnames = list(rows[0].keys()) if rows else \
        ["episode", "block_x", "block_y", "target_x", "target_y", "n_blocks", "vis_path"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(json_path, "w") as f:
        json.dump(json_dump, f, indent=2)

    return csv_path, json_path