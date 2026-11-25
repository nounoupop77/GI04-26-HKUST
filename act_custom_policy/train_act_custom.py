import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.v2 as T
import random
import os
import signal
import sys

from lerobot.datasets import LeRobotDataset
from modeling_act_custom import ACTCustomPolicy

# ================= CONFIGURATION =================
BASE_DIR = Path("/home/ckwuae/dataset")
BATCH_SIZE = 32             
EPOCHS = 50
LEARNING_RATE = 1e-5

# ⚠️ REDUCED CHUNK SIZE FOR SPEED
# A chunk of 30 is usually plenty for smooth motion and prevents O(N^2) attention slowdown.
CHUNK_SIZE = 100             

NUM_TESTS_TO_LOAD = 100

SAVE_DIR = Path("outputs/train/custom_act")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = SAVE_DIR / "latest_checkpoint.pth"
BEST_MODEL_PATH = SAVE_DIR / "best_act.pth"
# =================================================

# Global flag for Slurm preemption
PREEMPTED = False
def handle_sigterm(signum, frame):
    global PREEMPTED
    print("\n[Warning] SIGTERM received. Will save checkpoint and exit...")
    PREEMPTED = True
signal.signal(signal.SIGTERM, handle_sigterm)


class ACTDataset(Dataset):
    def __init__(self, paths, is_train=True):
        self.datasets = [LeRobotDataset(repo_id="data", root=p) for p in paths]
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.lengths)
        
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) if is_train else T.Identity(),
            T.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self): 
        return self.total_length

    def __getitem__(self, idx):
        current_idx = idx
        for ds, length in zip(self.datasets, self.lengths):
            if current_idx < length:
                item = ds[current_idx]
                
                # --- ACTION CHUNKING ---
                chunk = []
                for i in range(CHUNK_SIZE):
                    future_idx = min(current_idx + i, length - 1)
                    chunk.append(ds[future_idx]["action"])
                action_chunk = torch.stack(chunk) 

                img_front = item.get("observation.images.front", item.get("observation.images.top"))
                img_side = item.get("observation.images.side", item.get("observation.images.wrist"))

                return {
                    "observation.images.front": self.transform(img_front),
                    "observation.images.side": self.transform(img_side),
                    "observation.state": item["observation.state"],
                    "action": action_chunk
                }
            current_idx -= length
        raise IndexError("Index out of range")


def main():
    print("Scanning folders...")
    dataset_paths = [BASE_DIR / f"test{i}" for i in range(1, NUM_TESTS_TO_LOAD + 1) if (BASE_DIR / f"test{i}").exists()]
    if not dataset_paths: 
        raise ValueError("No datasets found.")

    random.seed(42)
    random.shuffle(dataset_paths)
    split_idx = int(len(dataset_paths) * 0.8)
    train_paths = dataset_paths[:split_idx]
    val_paths = dataset_paths[split_idx:]
    
    print(f"Training on {len(train_paths)} folders | Validating on {len(val_paths)} folders.")

    # --- SPEED FIX 1: DataLoader Optimization ---
    # persistent_workers=True keeps the CPU workers alive between epochs, stopping the 
    # CPU from having to restart them (which causes massive delays).
    train_loader = DataLoader(
        ACTDataset(train_paths, True), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=16, 
        pin_memory=True, 
        prefetch_factor=4,
        persistent_workers=True 
    )
    val_loader = DataLoader(
        ACTDataset(val_paths, False), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACTCustomPolicy(chunk_size=CHUNK_SIZE).to(device)
    
    # --- SPEED FIX 2: Freeze Vision Backbones ---
    # This stops PyTorch from calculating gradients for the 11 million ResNet parameters.
    for param in model.encoder_front.parameters(): 
        param.requires_grad = False
    for param in model.encoder_side.parameters(): 
        param.requires_grad = False
    print("Vision backbones frozen.")

    # --- SPEED FIX 3: torch.compile ---
    # Uses PyTorch 2.0+ to compile the model graph for a 20-30% speed boost.
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile()... (First epoch might be slightly slow)")
        model = torch.compile(model)

    # Only optimize the trainable parameters (Transformer & CVAE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_l1 = nn.L1Loss()

    # --- SPEED FIX 4: Mixed Precision (AMP) ---
    # Uses float16 math to double GPU speed and halve memory usage.
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_loss = float('inf')
    
    # --- CHECKPOINT LOADING ---
    if CHECKPOINT_PATH.exists():
        print(f"Resuming from {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            # Remove the "_orig_mod." prefix that torch.compile adds to state dict keys
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from Epoch {start_epoch}/{EPOCHS}")
        except Exception as e:
            print(f"[Warning] Checkpoint issue: {e}. Starting fresh.")
            start_epoch = 0

    print(f"Training on {device}...")

    for epoch in range(start_epoch, EPOCHS):
        if PREEMPTED: 
            break

        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            if PREEMPTED: 
                break

            # non_blocking=True allows CPU to keep loading while GPU processes
            img_f = batch["observation.images.front"].to(device, non_blocking=True)
            img_s = batch["observation.images.side"].to(device, non_blocking=True)
            state = batch["observation.state"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Autocast enables float16 math
            with torch.amp.autocast('cuda'):
                pred_actions, mu, logvar = model(img_f, img_s, state, actions)
                loss_l1 = criterion_l1(pred_actions, actions)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                
                # Lowered KL weight to 1.0 to prevent mode collapse (freezing)
                loss = loss_l1 + (1.0 * kl_loss) 

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(pbar))

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        
        if not PREEMPTED:
            with torch.no_grad():
                for batch in val_loader:
                    img_f = batch["observation.images.front"].to(device, non_blocking=True)
                    img_s = batch["observation.images.side"].to(device, non_blocking=True)
                    state = batch["observation.state"].to(device, non_blocking=True)
                    actions = batch["action"].to(device, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        # actions=None triggers inference mode (z=0)
                        pred_actions, _, _ = model(img_f, img_s, state, actions=None)
                        loss = criterion_l1(pred_actions, actions)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # --- SAVE BEST MODEL ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"--> New best model! Saving to {BEST_MODEL_PATH}")
                temp_best_path = SAVE_DIR / "temp_best.pth"
                
                # Handle torch.compile prefix for saving
                save_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                torch.save(save_dict, temp_best_path)
                
                if temp_best_path.exists(): 
                    os.replace(temp_best_path, BEST_MODEL_PATH)
        else:
            print(f"Epoch {epoch+1} interrupted.")

        # --- SAFE CHECKPOINT SAVING ---
        temp_checkpoint_path = SAVE_DIR / "temp_checkpoint.pth"
        
        # Handle torch.compile prefix for saving
        save_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': save_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'best_val_loss': best_val_loss
        }, temp_checkpoint_path)
        
        if temp_checkpoint_path.exists(): 
            os.replace(temp_checkpoint_path, CHECKPOINT_PATH)

        if PREEMPTED:
            print("Checkpoint saved. Exiting cleanly due to preemption.")
            sys.exit(0)

    print("Training complete.")

if __name__ == "__main__":
    main()