"""
Deploy script for VERSION 3 (SSM + object detection, NO GRU).

Architecture confirmed from modeling_custom (V3):
    f_front = ssm_top(encoder_top(frames_front))[-1]          # [B, 1024]
    f_side  = ssm_wrist(encoder_wrist(frames_side))[-1]       # [B, 1024]
    kp_emb  = kp_encoder(cat([kp(4), kp_presence(4)]))        # [B, 64]
    out     = fc(cat([f_front, f_side, state(6), kp_emb]))    # [B, chunk*6]
    out.view(B, chunk, 6)

Detection (from detection.py):
    - Once, on the FIRST frame after connection, on the FRONT camera
      (which corresponds to `observation.images.top` in training).
    - detect_frame0(bgr) -> DetectionResult
    - to_vector()   -> 4-dim in [0,1] (block_x/W, block_y/H, target_x/W, target_y/H)
    - presence_mask() -> 4-dim
    - Toggle with ENABLE_OBJECT_DETECTION.
      * If True : kp vector / presence come from detection
      * If False: kp = zeros(4), presence = zeros(4)  -> matches training dropout=1.0 ablation
    - Prints block position and target position to console.

NUM_FRAMES and ACTION_CHUNK_SIZE are fully configurable. Note V3's forward
ONLY uses the LAST frame of the stacked input, so increasing NUM_FRAMES at
deploy does not change predictions unless you retrain with larger NUM_FRAMES.
"""

"""
top: y = 0
bottom: y = 224
left : x = 0
right: x = 224
"""

import time
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T

from lerobot_policy_custom.modeling_custom import CustomPolicy
from lerobot_policy_custom.detect import detect_frame0
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.opencv import OpenCVCameraConfig

# ==========================================
# --- CONFIGURATION ---
# ==========================================
MODEL_PATH          = "V3_kp_best_8chunk_1frames_l1.pth"   # your best V3 ckpt
ROBOT_PORT          = "COM6"
ROBOT_ID            = "my_awesome_follower_arm"

# MUST match training checkpoint
NUM_FRAMES          = 1        # V3 training used 1 (only last frame is used)
ACTION_CHUNK_SIZE   = 8        # MUST match training
FC_HIDDEN_DIM       = 256
STATE_DIM           = 6
ACTION_DIM          = 6
KEYPOINT_DIM        = 8        # kp(4) + presence(4) fed to kp_encoder
KEYPOINT_EMB_DIM    = 64
KEYPOINT_DROPOUT    = 0.0      # irrelevant at eval, set for config completeness

# Object detection toggle
ENABLE_OBJECT_DETECTION = False   # False -> zeros; mimics training dropout=1.0

MAX_DURATION        = 120
JOINT_KEYS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
              'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']


# ----- internal SimpleConfig (only eval-relevant fields) -----
class SimpleConfig:
    fc_hidden_dim        = FC_HIDDEN_DIM
    state_dim            = STATE_DIM
    action_dim           = ACTION_DIM
    num_frames           = NUM_FRAMES
    action_chunk_size    = ACTION_CHUNK_SIZE
    bn_train_in_layer4   = False
    keypoint_dim         = KEYPOINT_DIM
    keypoint_emb_dim     = KEYPOINT_EMB_DIM
    keypoint_dropout_prob= KEYPOINT_DROPOUT


# ==========================================
# --- V3 PREDICTION (matches modeling_custom V3) ---
# ==========================================
def predict(model, stacked_front, stacked_side, state, kp, kp_presence):
    """
    stacked_front / side: [1, N, 3, 224, 224]
    state:        [1, 6] (already z-scored)
    kp:           [1, 4] in [0,1]
    kp_presence:  [1, 4] in {0,1}
    Returns:      [1, ACTION_CHUNK_SIZE, ACTION_DIM]
    """
    with torch.no_grad():
        # last frame only, matching V3 training pipeline
        kp_front = model._encode_frames(model.encoder_top,   model.ssm_top,
                                        stacked_front)[:, -1]    # [1, 1024]
        kp_side  = model._encode_frames(model.encoder_wrist, model.ssm_wrist,
                                        stacked_side)[:, -1]     # [1, 1024]
        kp_emb   = model.kp_encoder(kp, kp_presence)             # [1, 64]
        combined = torch.cat([kp_front, kp_side, state, kp_emb], dim=1)
        out      = model.fc(combined)                            # [1, chunk*6]
    return out.view(-1, ACTION_CHUNK_SIZE, ACTION_DIM)


# ==========================================
# --- DETECTION PRINTING ---
# ==========================================
def print_detection(det, enabled):
    print("=" * 56)
    print(f"ONE-SHOT OBJECT DETECTION (front camera @ connect) "
          f"[{'ON' if enabled else 'OFF'}]")
    if enabled:
        bx, by = det.block_xy if det.block_xy is not None else ("N/A", "N/A")
        tx, ty = det.cross_xy if det.cross_xy is not None else ("N/A", "N/A")
        print(f"  Block position  : ({bx}, {by})   [224x224 pixel space]")
        print(f"  Target position : ({tx}, {ty})   [224x224 pixel space]")
        print(f"  to_vector()     : {np.round(det.to_vector(), 4)}")
        print(f"  presence_mask() : {det.presence_mask().astype(int)}")
        print(f"  n_blocks found  : {len(det.blocks)}")
    else:
        print("  Detection disabled — kp=zeros(4), presence=zeros(4).")
    print("=" * 56)


# ==========================================
# --- MAIN ---
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Model
    model = CustomPolicy(SimpleConfig())
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print("Checkpoint keys:", list(ckpt.keys()))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Model loaded. chunk={ACTION_CHUNK_SIZE}, frames={NUM_FRAMES}")

    # 2. Stats (V3: state stays 6-dim; detection is NOT folded into state)
    state_mean = ckpt["state_stats"]["mean"].to(device)
    state_std  = ckpt["state_stats"]["std"].to(device)
    action_mean= ckpt["action_stats"]["mean"].to(device)
    action_std = ckpt["action_stats"]["std"].to(device)

    # 3. Image transform (val pipeline from V3 training)
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Cameras + robot
    front_cfg = OpenCVCameraConfig(index_or_path=1, width=640, height=480,
                                   fps=20, fourcc="MJPG")
    side_cfg  = OpenCVCameraConfig(index_or_path=0, width=640, height=480,
                                   fps=20, fourcc="MJPG")
    print(f"Connecting to robot on {ROBOT_PORT}...")
    robot = SO101Follower(SO101FollowerConfig(
        port=ROBOT_PORT, id=ROBOT_ID,
        cameras={"front": front_cfg, "side": side_cfg}))
    robot.connect()
    print("Robot connected!")
    initial_obs = robot.get_observation()
    origin_action_dict = {k: float(initial_obs[k]) for k in JOINT_KEYS}

    # 5. ONE-SHOT DETECTION (front camera, like training's `observation.images.top`)
    kp_vec      = np.zeros(4, dtype=np.float32)
    kp_presence = np.zeros(4, dtype=np.float32)
    if ENABLE_OBJECT_DETECTION:
        # initial_obs["front"] is BGR (cv2 path). detect_frame0 accepts BGR.
        det = detect_frame0(initial_obs["front"])
        print_detection(det, enabled=True)
        kp_vec      = det.to_vector()
        kp_presence = det.presence_mask()
    else:
        print_detection(None, enabled=False)

    kp_vec_t      = torch.tensor(kp_vec,      dtype=torch.float32,
                                 device=device).unsqueeze(0)   # [1,4]
    kp_presence_t = torch.tensor(kp_presence, dtype=torch.float32,
                                 device=device).unsqueeze(0)   # [1,4]

    # 6. Rolling buffers
    buf_front = deque(maxlen=NUM_FRAMES)
    buf_side  = deque(maxlen=NUM_FRAMES)

    start_time = time.time()
    fps_counter, fps_start = 0, time.time()
    current_fps = 0.0

    try:
        while True:
            if time.time() - start_time > MAX_DURATION:
                print(f"\nReached {MAX_DURATION}s. Stopping.")
                break
            t0 = time.time()

            # A. Observation
            obs = robot.get_observation()
            img_f = cv2.cvtColor(obs["front"], cv2.COLOR_BGR2RGB)
            img_s = cv2.cvtColor(obs["side"],  cv2.COLOR_BGR2RGB)
            img_f = transform(torch.from_numpy(img_f).permute(2, 0, 1))
            img_s = transform(torch.from_numpy(img_s).permute(2, 0, 1))

            buf_front.append(img_f)
            buf_side.append(img_s)
            while len(buf_front) < NUM_FRAMES:
                buf_front.appendleft(img_f)
                buf_side.appendleft(img_s)

            stacked_front = torch.stack(list(buf_front), 0).unsqueeze(0).to(device)
            stacked_side  = torch.stack(list(buf_side),  0).unsqueeze(0).to(device)

            # B. State (z-score only the 6 joint dims)
            joint = torch.tensor([obs[k] for k in JOINT_KEYS],
                                 dtype=torch.float32).unsqueeze(0).to(device)
            joint = (joint - state_mean) / (state_std + 1e-8)

            # C. Predict
            pred = predict(model, stacked_front, stacked_side,
                           joint, kp_vec_t, kp_presence_t)   # [1, chunk, 6]

            # D. Unnormalize + execute
            chunk = pred[0].cpu().numpy() * action_std.cpu().numpy() \
                                    + action_mean.cpu().numpy()
            for step_action in chunk:
                act = {k: float(step_action[i]) for i, k in enumerate(JOINT_KEYS)}
                robot.send_action(act)
                time.sleep(0.05)

            # FPS log
            fps_counter += 1
            now = time.time()
            if now - fps_start >= 1.0:
                current_fps = fps_counter / (now - fps_start)
                fps_counter, fps_start = 0, now
            print(f"[{now - start_time:5.1f}s] FPS:{current_fps:4.1f} | "
                  f"A1:{np.round(chunk[0], 2)}")

            time.sleep(max(0, 0.05 - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\nManual stop.")
    finally:
        print("Returning to start position...")
        robot.send_action(origin_action_dict)
        time.sleep(2.0)
        robot.disconnect()
        print("Robot disconnected safely.")


if __name__ == "__main__":
    main()