import sys, os
# Make `src/` importable without reinstall
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import matplotlib.pyplot as plt
import random, time, pickle, json

from lerobot.datasets import LeRobotDataset
from lerobot_policy_custom.modeling_custom import CustomPolicy
from lerobot_policy_custom.configuration_custom import CustomConfig
from lerobot_policy_custom.detection import (
    detect_frame0, write_detection_summary, DetectionResult,
)

# ==========================================
# --- CONFIGURATION ---
# ==========================================
BASE_DIR = Path("/home/itoaa/newdataset")

NUM_TESTS_TO_LOAD = 30
BATCH_SIZE = 32
MAX_EPOCHS = 40
HEAD_LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
STATE_DIM = 6
ACTION_DIM = 6
SPLIT_MODE = "interleaved" #interleaved/ random
NUM_FRAMES = 1
ACTION_CHUNK_SIZE = 8  
BN_TRAIN_IN_LAYER4 = False
LOSS_TYPE = "L1"  #L1 or MSE or Smooth L1??
UNFREEZE_ENCODER_AFTER_EPOCH = None # None = freeze forever

# ----- keypoint (block + target) -----
KEYPOINT_DROPOUT_PROB = 0.3     # 0.0 = always use; 0.8 = ~20% kept
DETECTION_CACHE_DIR   = "detection_cache"
DETECTION_SUMMARY_DIR = "detection_summary"

CHECKPOINT_PATH = "V3_kp_8chunk_1frames_l1.pth"
BEST_MODEL_PATH = "V3_kp_best_8chunk_1frames_l1.pth"
GRAPH_LOSS_PATH = "V3_kp_loss_8chunk_1frames_l1.png"


# ==========================================
def make_image_transform(is_train: bool):
    base = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    jitter = T.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.15, hue=0.03) if is_train else None
    return base, jitter


def _sample_jitter_params(jitter):
    return T.ColorJitter.get_params(
        jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
    )


def _apply_jitter(img, params):
    fn_idx, b, c, s, h = params
    for fn_id in fn_idx:
        if fn_id == 0 and b is not None:
            img = TF.adjust_brightness(img, b)
        elif fn_id == 1 and c is not None:
            img = TF.adjust_contrast(img, c)
        elif fn_id == 2 and s is not None:
            img = TF.adjust_saturation(img, s)
        elif fn_id == 3 and h is not None:
            img = TF.adjust_hue(img, h)
    return img


# ==========================================
# Convert a LeRobot tensor image -> BGR uint8 for detection
# ==========================================
def _tensor_image_to_bgr(img) -> "np.ndarray":
    import numpy as np
    if torch.is_tensor(img):
        arr = img.detach().cpu().permute(1, 2, 0).numpy()
        if arr.dtype != "uint8":
            arr = (arr * 255.0).clip(0, 255).astype("uint8")
        return arr[:, :, ::-1].copy()
    return img  # already a numpy array


# ==========================================
# DETECTION CACHE — one detection per episode, from observation.images.top
# ==========================================
def _load_or_run_detection(dataset, episode_id: int, cache_dir: str,
                           task_name: str) -> DetectionResult:
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{task_name}_ep{episode_id:04d}"
    pkl_path = os.path.join(cache_dir, key + ".pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if getattr(dataset, "episode_data_index", None) is not None:
        ep_from = dataset.episode_data_index["from"][episode_id].item()
    else:
        ep_from = 0

    item0 = dataset[ep_from]

    # <<< DETECTION FRAME IS ALWAYS THE TOP CAMERA >>>
    assert "observation.images.top" in item0, (
        f"Dataset '{task_name}' (episode {episode_id}) has no "
        f"'observation.images.top' key. Available keys: {list(item0.keys())}"
    )
    img0 = item0["observation.images.top"]
    bgr  = _tensor_image_to_bgr(img0)

    det = detect_frame0(bgr)
    with open(pkl_path, "wb") as f:
        pickle.dump(
            DetectionResult(cross_xy=det.cross_xy, block_xy=det.block_xy,
                            blocks=det.blocks, vis=None),
            f,
        )
    if det.vis is not None:
        import cv2
        cv2.imwrite(os.path.join(cache_dir, key + ".jpg"), det.vis)
    return det


# ==========================================
# DATASET
# ==========================================
class SingleTaskDataset(Dataset):
    def __init__(self, path, is_train, state_mean=None, state_std=None,
                 action_mean=None, action_std=None,
                 num_frames=1, action_chunk_size=1,
                 detection_cache_dir=DETECTION_CACHE_DIR):
        self.ds = LeRobotDataset(repo_id="data", root=path)
        self.base_tf, self.jitter_tf = make_image_transform(is_train)
        self.state_mean, self.state_std = state_mean, state_std
        self.action_mean, self.action_std = action_mean, action_std
        self.num_frames = num_frames
        self.action_chunk_size = action_chunk_size
        self.task_name = Path(path).name

        self._ep_keypoints = {}
        self._ep_presence  = {}
        if getattr(self.ds, "episode_data_index", None) is not None:
            self._ep_from = self.ds.episode_data_index["from"].tolist()
            self._ep_to   = self.ds.episode_data_index["to"].tolist()
            n_eps = len(self._ep_from)
        else:
            self._ep_from = self._ep_to = None
            n_eps = 1

        for ep in range(n_eps):
            det = _load_or_run_detection(self.ds, ep, detection_cache_dir,
                                         self.task_name)
            self._ep_keypoints[ep] = det.to_vector()
            self._ep_presence[ep]  = det.presence_mask()

    def __len__(self):
        return len(self.ds)

    def _episode_of(self, idx):
        if self._ep_from is None:
            return 0
        for ep, (s, e) in enumerate(zip(self._ep_from, self._ep_to)):
            if s <= idx < e:
                return ep
        return 0

    def _episode_bounds(self, ep):
        if self._ep_from is None:
            return 0, len(self.ds)
        return self._ep_from[ep], self._ep_to[ep]

    def __getitem__(self, idx):
        ep = self._episode_of(idx)
        ep_start, ep_end = self._episode_bounds(ep)
        shared_jitter = _sample_jitter_params(self.jitter_tf) if self.jitter_tf else None

        frames_front, frames_side = [], []
        for i in range(self.num_frames - 1, -1, -1):
            past_idx = max(ep_start, idx - i)
            past_item = self.ds[past_idx]
            img_f = past_item.get("observation.images.front",
                                  past_item.get("observation.images.top"))
            img_s = past_item.get("observation.images.side",
                                  past_item.get("observation.images.wrist"))
            img_f = TF.to_dtype(img_f, torch.float32, scale=True)
            img_s = TF.to_dtype(img_s, torch.float32, scale=True)
            if shared_jitter is not None:
                img_f = _apply_jitter(img_f, shared_jitter)
                img_s = _apply_jitter(img_s, shared_jitter)
            frames_front.append(self.base_tf(img_f))
            frames_side.append(self.base_tf(img_s))

        stacked_front = torch.stack(frames_front, dim=0)
        stacked_side  = torch.stack(frames_side,  dim=0)

        state = self.ds[idx]["observation.state"].float()
        if self.state_mean is not None:
            state = (state - self.state_mean) / (self.state_std + 1e-8)

        actions = []
        for k in range(self.action_chunk_size):
            a_idx = min(ep_end - 1, idx + k)
            a = self.ds[a_idx]["action"].float()
            if self.action_mean is not None:
                a = (a - self.action_mean) / (self.action_std + 1e-8)
            actions.append(a)
        action_chunk = torch.stack(actions, dim=0)

        kp = torch.from_numpy(self._ep_keypoints[ep])
        kp_presence = torch.from_numpy(self._ep_presence[ep])

        return {
            "observation.images.front": stacked_front,
            "observation.images.side":  stacked_side,
            "observation.state":        state,
            "observation.keypoint":     kp,
            "observation.keypoint_presence": kp_presence,
            "action":                   action_chunk,
        }


# ==========================================
# STATISTICS
# ==========================================
def compute_dataset_statistics(dataset_paths, state_dim, action_dim):
    state_values, action_values = [], []
    print("Computing training statistics...")
    for path in tqdm(dataset_paths):
        ds = LeRobotDataset(repo_id="data", root=path)
        for index in range(len(ds)):
            item = ds[index]
            state_values.append(item["observation.state"].float())
            action_values.append(item["action"].float())
    states = torch.stack(state_values)
    actions = torch.stack(action_values)
    return {
        "observation.state": {"mean": states.mean(dim=0),
                              "std": states.std(dim=0).clamp_min(1e-6)},
        "action":            {"mean": actions.mean(dim=0),
                              "std": actions.std(dim=0).clamp_min(1e-6)},
    }


# ==========================================
# Optimizer
# ==========================================
def make_optimizer(model, config):
    encoder_parameters = [
        p for p in list(model.encoder_top.layer4.parameters())
                   + list(model.encoder_wrist.layer4.parameters())
        if p.requires_grad
    ]
    head_parameters = [
        p for p in list(model.kp_encoder.parameters())
                   + list(model.fc.parameters())
        if p.requires_grad
    ]
    groups = [{"params": head_parameters, "lr": config.optimizer_lr,
               "weight_decay": config.optimizer_weight_decay}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": config.encoder_lr,
                       "weight_decay": config.optimizer_weight_decay})
    return torch.optim.Adam(groups)


# ==========================================
# Evaluation
# ==========================================
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            pred = model(batch_dev)
            loss = criterion(pred, batch_dev["action"])
            total_loss += loss.item() * batch_dev["action"].shape[0]
            total_samples += batch_dev["action"].shape[0]
    return total_loss / max(total_samples, 1)


# ==========================================
# Summary dump (CSV + JSON + annotated frame0.jpg from the TOP camera)
# ==========================================
def dump_detection_summaries(train_paths, val_paths):
    all_det = {}
    for path in list(train_paths) + list(val_paths):
        ds = LeRobotDataset(repo_id="data", root=path)
        task = Path(path).name
        if getattr(ds, "episode_data_index", None) is not None:
            ep_from = ds.episode_data_index["from"].tolist()
            n_eps = len(ep_from)
        else:
            ep_from = [0]; n_eps = 1

        for ep in range(n_eps):
            key = f"{task}_ep{ep:04d}"
            if key in all_det:
                continue
            item0 = ds[ep_from[ep]]
            # <<< DETECTION FRAME IS ALWAYS THE TOP CAMERA >>>
            assert "observation.images.top" in item0, (
                f"Dataset '{task}' has no 'observation.images.top'. "
                f"Available: {list(item0.keys())}"
            )
            bgr = _tensor_image_to_bgr(item0["observation.images.top"])
            all_det[key] = detect_frame0(bgr)

    csv_p, json_p = write_detection_summary(DETECTION_SUMMARY_DIR, all_det)
    print(f"[detection summary] CSV : {csv_p}")
    print(f"[detection summary] JSON: {json_p}")
    print(f"[detection summary] JPGs: {DETECTION_SUMMARY_DIR}/frame0_*.jpg")


# ==========================================
# main
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Scanning folders...")
    dataset_paths = []
    for i in range(1, NUM_TESTS_TO_LOAD + 1):
        folder = BASE_DIR / f"test{i}"
        if folder.exists():
            dataset_paths.append(folder)
    if not dataset_paths:
        raise ValueError("No datasets found. Check your BASE_DIR path.")

    train_paths, val_paths = [], []
    if SPLIT_MODE == "interleaved":
        print("Using Interleaved Split (1,2,3,4,6,7,8,9 Train, 5,10 Val)...")
        for path in dataset_paths:
            num = int(path.name.replace("test", ""))
            (val_paths if num % 10 in (5, 0) else train_paths).append(path)
    elif SPLIT_MODE == "random":
        random.seed(42); random.shuffle(dataset_paths)
        split_idx = int(len(dataset_paths) * 0.8)
        train_paths, val_paths = dataset_paths[:split_idx], dataset_paths[split_idx:]
    else:
        raise ValueError("Invalid SPLIT_MODE.")

    print(f"Training on {len(train_paths)} folders. Validating on {len(val_paths)} folders.")

    # One-time detection (top camera @ frame 0) + written summaries
    print("\nRunning one-time block + target detection (observation.images.top @ t=0)...")
    dump_detection_summaries(train_paths, val_paths)

    # Statistics
    stats_file = f"dataset_stats_{SPLIT_MODE}_{len(train_paths)}.pkl"
    if os.path.exists(stats_file):
        print("Loading cached statistics...")
        with open(stats_file, "rb") as f:
            raw_stats = pickle.load(f)
    else:
        raw_stats = compute_dataset_statistics(train_paths, STATE_DIM, ACTION_DIM)
        with open(stats_file, "wb") as f:
            pickle.dump(raw_stats, f)

    sm, ss  = raw_stats["observation.state"]["mean"], raw_stats["observation.state"]["std"]
    am, as_ = raw_stats["action"]["mean"], raw_stats["action"]["std"]

    train_dataset = ConcatDataset([
        SingleTaskDataset(p, True, sm, ss, am, as_, NUM_FRAMES, ACTION_CHUNK_SIZE)
        for p in train_paths
    ])
    val_dataset = ConcatDataset([
        SingleTaskDataset(p, False, sm, ss, am, as_, NUM_FRAMES, ACTION_CHUNK_SIZE)
        for p in val_paths
    ])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    config = CustomConfig(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        fc_hidden_dim=256,
        optimizer_lr=HEAD_LEARNING_RATE,
        encoder_lr=ENCODER_LEARNING_RATE,
        optimizer_weight_decay=WEIGHT_DECAY,
        num_frames=NUM_FRAMES,
        action_chunk_size=ACTION_CHUNK_SIZE,
        bn_train_in_layer4=BN_TRAIN_IN_LAYER4,
        keypoint_dim=8,
        keypoint_emb_dim=64,
        keypoint_dropout_prob=KEYPOINT_DROPOUT_PROB,
    )

    model = CustomPolicy(config).to(device)
    optimizer = make_optimizer(model, config)
    criterion = nn.MSELoss() if LOSS_TYPE == "MSE" else nn.L1Loss()

    best_val_loss, start_epoch = float("inf"), 0
    train_losses, val_losses = [], []

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses", [])
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        if UNFREEZE_ENCODER_AFTER_EPOCH is not None and epoch == UNFREEZE_ENCODER_AFTER_EPOCH:
            print("Unfreezing ResNet layer4 ...")
            model.unfreeze_last_block()
            optimizer = make_optimizer(model, config)

        model.train()
        total_train_loss, total_train_samples = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}")
        for batch in progress:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            target = batch_dev["action"]
            optimizer.zero_grad()
            pred = model(batch_dev)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * target.shape[0]
            total_train_samples += target.shape[0]
            progress.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_train_loss = total_train_loss / max(total_train_samples, 1)
        avg_val_loss = evaluate(model, val_loader, device, criterion)
        train_losses.append(avg_train_loss); val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "state_stats": raw_stats["observation.state"],
                "action_stats": raw_stats["action"],
            }, BEST_MODEL_PATH)
            print(f"Validation {LOSS_TYPE} improved! Saved new best to {BEST_MODEL_PATH}.")
        else:
            print(f"Validation {LOSS_TYPE} did not improve. Best remains: {best_val_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "state_stats": raw_stats["observation.state"],
            "action_stats": raw_stats["action"],
        }, CHECKPOINT_PATH)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label=f"Train {LOSS_TYPE}")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label=f"Val {LOSS_TYPE}")
    plt.xlabel("Epoch"); plt.ylabel("Normalized loss"); plt.title("Training and Validation Loss")
    plt.grid(True); plt.legend()
    plt.savefig(GRAPH_LOSS_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()