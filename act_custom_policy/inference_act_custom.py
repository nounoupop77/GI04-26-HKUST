import torch
import torchvision.transforms.v2 as T
import numpy as np
import time
from collections import deque
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from modeling_act_custom import ACTCustomPolicy

# ================= CONFIGURATION — ADJUST THESE =================
WEIGHTS_PATH = Path(
    r"C:\Users\win11\lerobot\src\lerobot\act_custom_policy\best_act_latestdataset_3250.pth"
)

ROBOT_PORT = "COM8"
ROBOT_ID = "my_awesome_follower_arm"
CAMERA_FRONT_INDEX = 2
CAMERA_SIDE_INDEX = 0
FPS = 20
DURATION_SECONDS = 60

MAX_RELATIVE_STEP = 20

CHUNK_SIZE = 100  # fallback only; real value read from checkpoint

DEBUG_FIRST_FRAME = True

# How many recent loop iterations to average for the FPS display
FPS_WINDOW = 30
# =================================================================

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def return_to_safe_start(robot, current_pose, target_pose, steps=50, delay=0.05):
    print("\n[Safety] Returning arm to its initial starting position...")
    for i in range(1, steps + 1):
        interp = current_pose + (target_pose - current_pose) * (i / steps)
        action_dict = {name: float(val) for name, val in zip(JOINT_NAMES, interp)}
        try:
            robot.send_action(action_dict)
        except Exception as e:
            print(f"[Safety Error] Failed to send action: {e}")
        time.sleep(delay)

    print("[Safety] Holding position to stabilize...")
    hold = {name: float(val) for name, val in zip(JOINT_NAMES, target_pose)}
    for _ in range(20):
        try:
            robot.send_action(hold)
        except Exception:
            pass
        time.sleep(0.1)
    print("[Safety] Reached starting position.")


def build_camera_config(front_idx, side_idx, width=640, height=480, fps=20, fourcc="MJPG"):
    return {
        "front": OpenCVCameraConfig(
            index_or_path=front_idx, width=width, height=height, fps=fps, fourcc=fourcc
        ),
        "side": OpenCVCameraConfig(
            index_or_path=side_idx, width=width, height=height, fps=fps, fourcc=fourcc
        ),
    }


def get_obs_image(obs, primary_key, fallback_key):
    candidates = [
        primary_key,
        f"observation.images.{primary_key}",
        fallback_key,
        f"observation.images.{fallback_key}",
    ]
    for k in candidates:
        if k in obs and obs[k] is not None:
            return obs[k], k
    return None, None


def load_model(device, fallback_chunk_size):
    print(f"Loading weights from {WEIGHTS_PATH}...")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS_PATH}")

    raw = torch.load(WEIGHTS_PATH, map_location="cpu")
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if "action_queries" not in state_dict:
        raise KeyError(
            "state_dict has no 'action_queries' key. Is this really the ACT checkpoint?"
        )

    queries_shape = state_dict["action_queries"].shape
    ckpt_chunk_size = int(queries_shape[1])
    hidden_dim = int(queries_shape[2])
    action_dim = int(state_dict["action_head.weight"].shape[0])

    if ckpt_chunk_size != fallback_chunk_size:
        print(
            f"[WARN] Checkpoint chunk_size={ckpt_chunk_size}, "
            f"config CHUNK_SIZE={fallback_chunk_size}. Using checkpoint value."
        )

    if action_dim != len(JOINT_NAMES):
        raise ValueError(
            f"Checkpoint action_dim={action_dim}, robot has {len(JOINT_NAMES)} joints."
        )

    model = ACTCustomPolicy(
        state_dim=len(JOINT_NAMES),
        action_dim=action_dim,
        chunk_size=ckpt_chunk_size,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(
        f"[Model] Loaded on {device} | chunk_size={ckpt_chunk_size} | "
        f"hidden_dim={hidden_dim} | action_dim={action_dim}"
    )
    return model, ckpt_chunk_size


def preprocess_image(img_np, transform):
    tensor = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1)
    return transform(tensor).unsqueeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, chunk_size = load_model(device, fallback_chunk_size=CHUNK_SIZE)

    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

    camera_cfg = build_camera_config(
        CAMERA_FRONT_INDEX, CAMERA_SIDE_INDEX, fps=FPS, fourcc="MJPG"
    )
    robot_cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras=camera_cfg)

    robot = SO101Follower(robot_cfg)
    robot.connect(calibrate=False)
    print(f"[Robot] Connected to {ROBOT_ID} on {ROBOT_PORT}")

    obs = robot.get_observation()
    last_action = np.array([obs[k] for k in JOINT_NAMES], dtype=np.float32)
    initial_start_pose = last_action.copy()
    print(f"[Robot] Initial start position recorded: {initial_start_pose.round(3)}")

    max_steps = int(DURATION_SECONDS * FPS) + chunk_size + 10
    all_time_actions = torch.zeros(
        (max_steps, max_steps + chunk_size, len(JOINT_NAMES)),
        device=device,
    )

    print(
        f"[Inference] Starting loop at target {FPS} Hz for {DURATION_SECONDS}s "
        f"(chunk={chunk_size}, ensemble buffer={max_steps} steps)..."
    )

    # --- FPS / timing trackers ---
    loop_times = deque(maxlen=FPS_WINDOW)
    t_obs = deque(maxlen=FPS_WINDOW)
    t_infer = deque(maxlen=FPS_WINDOW)
    t_send = deque(maxlen=FPS_WINDOW)

    first_frame_logged = False
    start_time = time.time()
    step = 0

    def fmt_fps(dq):
        if len(dq) < 2:
            return float("nan")
        avg = sum(dq) / len(dq)
        return 1.0 / avg if avg > 0 else float("nan")

    try:
        while True:
            loop_start = time.perf_counter()

            # --- A. Observe ---
            t_a = time.perf_counter()
            obs = robot.get_observation()
            t_obs.append(time.perf_counter() - t_a)

            if DEBUG_FIRST_FRAME and not first_frame_logged:
                print(f"[Debug] Observation keys: {list(obs.keys())}")
                missing = [k for k in JOINT_NAMES if k not in obs]
                if missing:
                    print(f"[Debug] MISSING JOINT KEYS: {missing}")

            img_front_raw, front_key = get_obs_image(obs, "front", "top")
            img_side_raw, side_key = get_obs_image(obs, "side", "wrist")

            if img_front_raw is None or img_side_raw is None:
                print(
                    f"[Warning] Camera missing. front_key={front_key}, "
                    f"side_key={side_key}, obs keys={list(obs.keys())}"
                )
                time.sleep(1.0 / FPS)
                step += 1
                if step >= max_steps:
                    break
                continue

            # --- B. Preprocess + model forward ---
            t_b = time.perf_counter()
            img_f = preprocess_image(img_front_raw, transform).to(device)
            img_s = preprocess_image(img_side_raw, transform).to(device)
            joint_angles_np = np.array([obs[k] for k in JOINT_NAMES], dtype=np.float32)
            state = torch.from_numpy(joint_angles_np).unsqueeze(0).to(device)

            with torch.inference_mode():
                pred_chunk, _, _ = model(img_f, img_s, state, actions=None)
            t_infer.append(time.perf_counter() - t_b)

            if step >= max_steps:
                print("[Inference] Ensemble buffer exhausted. Stopping.")
                break
            all_time_actions[step, step:step + chunk_size] = pred_chunk[0]

            past_steps = range(max(0, step - chunk_size + 1), step + 1)
            actions_for_current_step = torch.stack(
                [all_time_actions[p, step] for p in past_steps]
            )
            ensemble_action = actions_for_current_step.mean(dim=0).cpu().numpy()

            delta = ensemble_action - last_action
            delta_clipped = np.clip(delta, -MAX_RELATIVE_STEP, MAX_RELATIVE_STEP)
            safe_action = last_action + delta_clipped
            last_action = safe_action.copy()

            if DEBUG_FIRST_FRAME and not first_frame_logged:
                print(f"[Debug] front_key={front_key}, side_key={side_key}")
                print(f"[Debug] raw pred_chunk[0] = {pred_chunk[0].cpu().numpy().round(3)}")
                print(f"[Debug] raw chunk |max| = {pred_chunk[0].abs().max().item():.3f}")
                print(f"[Debug] delta |max| (pre-clip) = {np.abs(delta).max():.3f}")
                raw_mag = pred_chunk[0].abs().max().item()
                if raw_mag > 20 and MAX_RELATIVE_STEP < 2:
                    print("[Debug] WARN: raw actions look like .pos degrees but "
                          f"MAX_RELATIVE_STEP={MAX_RELATIVE_STEP} is small.")
                if raw_mag < 5 and MAX_RELATIVE_STEP > 2:
                    print("[Debug] WARN: raw actions look like radians but "
                          f"MAX_RELATIVE_STEP={MAX_RELATIVE_STEP} is large.")
                first_frame_logged = True

            # --- C. Send action ---
            t_c = time.perf_counter()
            action_dict = {name: float(val) for name, val in zip(JOINT_NAMES, safe_action)}
            robot.send_action(action_dict)
            t_send.append(time.perf_counter() - t_c)

            # --- Loop timing ---
            step += 1
            loop_duration = time.perf_counter() - loop_start
            loop_times.append(loop_duration)

            if step % 30 == 0 or step <= 3:
                elapsed = time.time() - start_time
                actual_fps = fmt_fps(loop_times)
                avg_obs_ms = 1000 * sum(t_obs) / max(1, len(t_obs))
                avg_inf_ms = 1000 * sum(t_infer) / max(1, len(t_infer))
                avg_snd_ms = 1000 * sum(t_send) / max(1, len(t_send))
                print(
                    f"[Step {step:4d}] t={elapsed:6.1f}s | "
                    f"FPS={actual_fps:5.1f} (target {FPS}) | "
                    f"obs={avg_obs_ms:5.1f}ms infer={avg_inf_ms:5.1f}ms send={avg_snd_ms:5.1f}ms | "
                    f"|delta|max={np.abs(delta_clipped).max():.3f}"
                )

            if DURATION_SECONDS > 0 and (time.time() - start_time) > DURATION_SECONDS:
                print("[Inference] Duration reached. Stopping.")
                break

            # Maintain loop rate
            sleep_time = (1.0 / FPS) - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Inference] Interrupted by user.")
    except Exception as e:
        print(f"\n[Error] Unexpected error during execution: {e}")
        raise
    finally:
        # Compute and print final stats
        total_elapsed = time.time() - start_time
        overall_fps = step / total_elapsed if total_elapsed > 0 else 0.0
        print(f"\n[Inference] Ran {step} steps in {total_elapsed:.1f}s "
              f"(overall FPS={overall_fps:.2f})")

        print("[Safety] Initiating shutdown sequence...")
        try:
            return_to_safe_start(robot, last_action, initial_start_pose)
        except Exception as e:
            print(f"[Safety Warning] Could not perform safe return: {e}")
        finally:
            robot.disconnect()
            print("[Robot] Disconnected safely.")


if __name__ == "__main__":
    main()