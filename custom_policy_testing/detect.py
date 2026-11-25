"""
detection.py

Wraps the user's existing lighting-invariant cross + block detection.
The detection functions below are copied VERBATIM from hiai.txt.

Public API used by training / modelling:

    detect_frame0(frame_bgr) -> DetectionResult

Detection is always run on the TOP camera view (`observation.images.top`),
once per episode, at frame 0.
"""
from __future__ import annotations
import os
import csv
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# === USER'S ORIGINAL CONSTANTS — DO NOT MODIFY ===
# ---------------------------------------------------------------------------
IMG_SIZE           = (224, 224)
MIN_AREA           = 200
MAX_AREA_FRAC      = 0.09
MAX_TARGET_R       = 26.0
IGNORE_BOTTOM_FRAC = 0.15
# ===========================================================================


# ---------------------------------------------------------------------------
# === USER'S ORIGINAL FUNCTIONS — DO NOT MODIFY ===
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


def candidate_quads(bgr):
    h, w = bgr.shape[:2]
    max_area = MAX_AREA_FRAC * (h * w)

    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)
    _, s_chan, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV))

    chroma_lab = np.hypot(a_chan - 128.0, b_chan - 128.0)
    chroma_score = chroma_lab * 2.2 + s_chan.astype(np.float32) * 0.4
    chroma_u8 = np.clip(chroma_score, 0, 255).astype(np.uint8)
    otsu_c, _ = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_color = cv2.threshold(chroma_u8, max(32.0, float(otsu_c)), 255, cv2.THRESH_BINARY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan.astype(np.uint8))
    se_local = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    local_diff = cv2.morphologyEx(l_eq, cv2.MORPH_BLACKHAT, se_local)
    _, mask_local = cv2.threshold(local_diff, 42, 255, cv2.THRESH_BINARY)

    fg_mask = cv2.bitwise_or(mask_color, mask_local)

    ga = cv2.magnitude(cv2.Scharr(a_chan, cv2.CV_32F, 1, 0), cv2.Scharr(a_chan, cv2.CV_32F, 0, 1))
    gb = cv2.magnitude(cv2.Scharr(b_chan, cv2.CV_32F, 1, 0), cv2.Scharr(b_chan, cv2.CV_32F, 0, 1))
    gl = cv2.magnitude(cv2.Scharr(l_eq.astype(np.float32), cv2.CV_32F, 1, 0),
                       cv2.Scharr(l_eq.astype(np.float32), cv2.CV_32F, 0, 1))
    edge_mag = (ga + gb) * 1.5 + gl * 0.5
    edge_u8 = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sharp_edges = cv2.threshold(edge_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    mx, my_top = int(w * 0.03), int(h * 0.03)
    my_bot = max(my_top, int(round(h * IGNORE_BOTTOM_FRAC)))
    fg_mask[:my_top, :] = 0
    if my_bot > 0:
        fg_mask[h - my_bot:, :] = 0
    fg_mask[:, :mx] = 0
    fg_mask[:, w - mx:] = 0

    fg_carved = cv2.bitwise_and(fg_mask, cv2.bitwise_not(sharp_edges))
    dist = cv2.distanceTransform(fg_carved, cv2.DIST_L2, 5)
    if dist.max() > 0:
        _, sure_fg = cv2.threshold(dist, 0.42 * dist.max(), 255, 0)
    else:
        sure_fg = fg_mask.copy()
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(fg_mask, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(blur, markers)

    cands = []
    for label_id in range(2, num_labels + 1):
        comp_mask = np.uint8(markers == label_id) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(c)
        if not (MIN_AREA * 0.75 <= area <= max_area):
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area == 0 or (area / hull_area) < 0.76:
            continue

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        rot_aspect = min(rw, rh) / max(rw, rh)
        if rot_aspect < 0.50:
            continue

        rectangularity = area / max(1.0, rw * rh)
        if rectangularity < 0.68:
            continue

        box = cv2.boxPoints(rect)
        quad = np.int32(np.round(box)).reshape(-1, 1, 2)

        inner_e = cv2.erode(comp_mask, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(inner_e) == 0:
            inner_e = comp_mask

        halo = cv2.subtract(
            cv2.dilate(comp_mask, np.ones((13, 13), np.uint8)),
            cv2.dilate(comp_mask, np.ones((5, 5), np.uint8)),
        )
        color_contrast = 0.0
        inside_chroma_std = 0.0
        if cv2.countNonZero(halo):
            fg_lab = np.array(cv2.mean(lab, mask=inner_e)[:3]) * np.array([0.8, 2.0, 2.0])
            bg_lab = np.array(cv2.mean(lab, mask=halo)[:3]) * np.array([0.8, 2.0, 2.0])
            color_contrast = float(np.linalg.norm(fg_lab - bg_lab))

            _, sd_a = cv2.meanStdDev(a_chan, mask=inner_e)
            _, sd_b = cv2.meanStdDev(b_chan, mask=inner_e)
            inside_chroma_std = float(np.hypot(sd_a.item(), sd_b.item()))

        if color_contrast < 22.0:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cands.append(
            dict(
                area=area,
                cx=cx,
                cy=cy,
                quad=quad,
                rot_aspect=rot_aspect,
                rectangularity=rectangularity,
                inside_std=inside_chroma_std,
                color_contrast=color_contrast,
            )
        )

    return cands


def select_blocks(cands):
    if not cands:
        return []

    for c in cands:
        aspect_score = c["rot_aspect"] * 30.0
        rect_score = c["rectangularity"] * 30.0
        uniform_score = max(0.0, (25.0 - c["inside_std"]) / 25.0) * 20.0
        contrast_score = min(c["color_contrast"] / 50.0, 1.0) * 20.0
        c["score"] = aspect_score + rect_score + uniform_score + contrast_score

    survivors = [c for c in cands if c["score"] >= 54.0]
    survivors.sort(key=lambda c: (-c["score"], -c["area"]))

    med_side = float(np.sqrt(np.median([c["area"] for c in survivors]))) if survivors else 15.0
    cluster_r = max(10.0, 0.85 * med_side)

    picked = []
    for c in survivors:
        if all(np.hypot(c["cx"] - p["cx"], c["cy"] - p["cy"]) > cluster_r for p in picked):
            picked.append(c)

    return picked


def run_frame0_pipeline(frame_bgr, label=""):
    frame = cv2.resize(frame_bgr, IMG_SIZE)
    vis = frame.copy()

    tx, ty, cross_r, cross_found = detect_cross_by_shape(frame)
    locked_cross = (tx, ty) if cross_found else None

    cands = candidate_quads(frame)
    blocks = select_blocks(cands)
    blocks = [b for b in blocks if b["area"] >= MIN_AREA]

    if IGNORE_BOTTOM_FRAC > 0.0:
        cutoff_y = int(round(IMG_SIZE[1] * (1.0 - IGNORE_BOTTOM_FRAC)))
        blocks = [b for b in blocks if b["cy"] < cutoff_y]
        cv2.line(vis, (0, cutoff_y), (IMG_SIZE[0], cutoff_y), (0, 220, 220), 1)

    if blocks:
        block_half_side = 0.5 * float(np.sqrt(np.median([b["area"] for b in blocks])))
        ignore_r = max(cross_r * 1.35, block_half_side * 1.1)
    else:
        ignore_r = cross_r * 1.45

    ignore_r = float(np.clip(ignore_r, 8.0, MAX_TARGET_R))

    if locked_cross is not None:
        cx_t, cy_t = locked_cross
        blocks = [b for b in blocks if np.hypot(b["cx"] - cx_t, b["cy"] - cy_t) > ignore_r]

    if locked_cross is not None:
        cx_t, cy_t = locked_cross
        r = int(round(ignore_r))
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

    return vis, locked_cross, blocks, len(cands)


# ---------------------------------------------------------------------------
# === NEW: canonical wrapper used by the model ===
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
    """Run detection once on a raw BGR frame (any resolution), typically the
    TOP camera at t=0. Returns block + cross coordinates in 224x224 pixel
    space, plus the annotated visualization."""
    vis, cross, blocks, _ = run_frame0_pipeline(frame_bgr, label="detect")

    chosen = None
    if blocks:
        if cross is not None:
            chosen = min(blocks, key=lambda b: np.hypot(b["cx"] - cross[0],
                                                        b["cy"] - cross[1]))
        else:
            chosen = max(blocks, key=lambda b: b["score"])

    return DetectionResult(
        cross_xy=tuple(cross) if cross is not None else None,
        block_xy=(chosen["cx"], chosen["cy"]) if chosen is not None else None,
        blocks=blocks,
        vis=vis,
    )


# ---------------------------------------------------------------------------
# === NEW: persist detection results for the whole dataset ===
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