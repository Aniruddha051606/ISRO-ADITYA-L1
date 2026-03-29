"""
train_lstm.py
=============
Training Pipeline for the CNN-LSTM Solar Flare Sequence Model
Mission: ISRO Aditya-L1

Handles:
  - Sequence dataset construction (sliding window over time-sorted frames)
  - Class-imbalance via WeightedRandomSampler + Focal Loss
  - Mixed-precision training (torch.cuda.amp)
  - LR scheduling with ReduceLROnPlateau
  - Checkpoint saving (best val-AUC)
  - TensorBoard logging
"""

import os
import glob
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model_lstm import SolarFlareSequenceModel, init_weights

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Focal Loss — combats extreme class imbalance better than BCE+pos_weight
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha : Weighting factor for the positive (flare) class.
                Set to (neg_count / pos_count) to match scale_pos_weight logic.
        gamma : Focusing parameter. gamma=2 is the original paper default.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)                   # (B,)
        probs    = torch.sigmoid(logits)
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w  = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce_loss).mean()


# ---------------------------------------------------------------------------
# Dataset: Sliding-window sequence builder
# ---------------------------------------------------------------------------

class SolarSequenceDataset(Dataset):
    """
    Builds overlapping temporal sequences from a sorted list of image paths.

    Expected file naming convention (lexicographic = time order):
        YYYYMMDD_HHMMSS_<label>.png
        e.g. 20240315_143022_0.png  (Quiet Sun)
             20240315_155500_1.png  (Solar Flare)

    Catalog CSV columns:
        filename, label, EXPTIME, SUN_CX, SUN_CY, R_SUN

    Each item returns:
        images_seq : (T, C, H, W)  normalised float tensor
        tabular    : (4,)          normalised float tensor (last frame's header)
        label      : (1,)          float {0.0, 1.0}
    """

    # ImageNet normalisation — valid since EfficientNet was pretrained on it
    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    def __init__(
        self,
        catalog_path:    str,
        image_dir:       str,
        seq_len:         int   = 5,
        augment:         bool  = False,
        tab_means:       Optional[np.ndarray] = None,
        tab_stds:        Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.augment  = augment

        # --- Load catalog ---
        df = pd.read_csv(catalog_path)
        # Sort chronologically so sliding window = temporal order
        df = df.sort_values("filename").reset_index(drop=True)

        self.filenames = df["filename"].tolist()
        self.labels    = df["label"].values.astype(np.float32)

        # Physics features
        tab_cols = ["EXPTIME", "SUN_CX", "SUN_CY", "R_SUN"]
        tab_data = df[tab_cols].values.astype(np.float32)

        # Fit normalisation on training set; use provided stats for val/test
        if tab_means is None:
            self.tab_means = tab_data.mean(axis=0)
            self.tab_stds  = tab_data.std(axis=0) + 1e-8
        else:
            self.tab_means = tab_means
            self.tab_stds  = tab_stds

        self.tab_data  = (tab_data - self.tab_means) / self.tab_stds
        self.image_dir = Path(image_dir)

        # Build valid starting indices: need seq_len consecutive frames
        # Label of a sequence = label of the *last* frame
        self.indices = list(range(seq_len - 1, len(df)))

        # Image transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.NORMALIZE,
        ])
        self.aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            self.NORMALIZE,
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, filename: str) -> torch.Tensor:
        path = self.image_dir / filename
        img  = Image.open(path).convert("RGB")
        if self.augment:
            return self.aug_transform(img)
        return self.base_transform(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end_idx   = self.indices[idx]
        start_idx = end_idx - self.seq_len + 1

        # Collect T frames
        frames = []
        for i in range(start_idx, end_idx + 1):
            frames.append(self._load_image(self.filenames[i]))

        images_seq = torch.stack(frames, dim=0)      # (T, C, H, W)
        tabular    = torch.tensor(
            self.tab_data[end_idx], dtype=torch.float32
        )                                            # (4,)
        label      = torch.tensor(
            [self.labels[end_idx]], dtype=torch.float32
        )                                            # (1,)

        return images_seq, tabular, label


# ---------------------------------------------------------------------------
# Weighted Sampler (oversample minority flare events)
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: SolarSequenceDataset) -> WeightedRandomSampler:
    """
    Compute per-sample weights inversely proportional to class frequency
    and return a WeightedRandomSampler for balanced mini-batches.
    """
    labels      = [dataset.labels[i] for i in dataset.indices]
    class_count = np.bincount(np.array(labels, dtype=int))   # [n_quiet, n_flare]
    total       = len(labels)
    weights     = [total / class_count[int(l)] for l in labels]
    sampler     = WeightedRandomSampler(
        weights     = weights,
        num_samples = total,
        replacement = True,
    )
    log.info(
        f"Sampler | Quiet Sun: {class_count[0]} | Flares: {class_count[1]} | "
        f"Ratio: {class_count[0]/max(class_count[1],1):.1f}:1"
    )
    return sampler


# ---------------------------------------------------------------------------
# Training + Validation Epochs
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    epoch:     int,
    writer:    SummaryWriter,
    grad_clip: float = 1.0,
) -> float:
    """Run one full training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for step, (imgs, tab, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)    # (B, T, C, H, W)
        tab    = tab.to(device,  non_blocking=True)    # (B, 4)
        labels = labels.to(device, non_blocking=True)  # (B, 1)

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass
        with autocast():
            out  = model(imgs, tab)
            loss = criterion(out["logits"], labels)

        scaler.scale(loss).backward()
        # Gradient clipping prevents LSTM exploding gradients
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 20 == 0:
            log.info(f"  Epoch {epoch} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

    mean_loss = total_loss / len(loader)
    writer.add_scalar("Loss/train", mean_loss, epoch)
    return mean_loss


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    writer:    SummaryWriter,
) -> Tuple[float, float]:
    """Evaluate on validation set. Returns (val_loss, val_auc)."""
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for imgs, tab, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        tab    = tab.to(device,    non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            out  = model(imgs, tab)
            loss = criterion(out["logits"], labels)

        total_loss += loss.item()
        all_probs.append(out["probs"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    mean_loss  = total_loss / len(loader)
    all_probs  = np.concatenate(all_probs).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

    writer.add_scalar("Loss/val", mean_loss, epoch)
    writer.add_scalar("AUC/val",  auc,       epoch)

    log.info(f"  Validation | Loss: {mean_loss:.4f} | AUC: {auc:.4f}")
    return mean_loss, auc


# ---------------------------------------------------------------------------
# Final Evaluation (test set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    threshold: float = 0.5,
) -> None:
    """Print full classification report on the test set."""
    model.eval()
    all_probs  = []
    all_labels = []

    for imgs, tab, labels in loader:
        imgs   = imgs.to(device)
        tab    = tab.to(device)
        out    = model(imgs, tab)
        all_probs.append(out["probs"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    preds      = (all_probs >= threshold).astype(int)

    log.info("\n" + "=" * 60)
    log.info("TEST SET EVALUATION")
    log.info("=" * 60)
    log.info(f"\n{classification_report(all_labels, preds, target_names=['Quiet Sun', 'Flare'])}")
    log.info(f"ROC-AUC  : {roc_auc_score(all_labels, all_probs):.4f}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(all_labels, preds)}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN-LSTM Solar Flare Model")
    p.add_argument("--catalog",      default="aditya_l1_catalog.csv",  help="Path to catalog CSV")
    p.add_argument("--image_dir",    default="data/images",             help="Directory of PNG images")
    p.add_argument("--output_dir",   default="checkpoints/lstm",        help="Checkpoint save dir")
    p.add_argument("--seq_len",      type=int,   default=5,             help="Temporal sequence length")
    p.add_argument("--batch_size",   type=int,   default=8,             help="Batch size (sequences are memory-heavy)")
    p.add_argument("--epochs",       type=int,   default=50,            help="Max training epochs")
    p.add_argument("--lr",           type=float, default=1e-4,          help="Initial learning rate")
    p.add_argument("--lstm_hidden",  type=int,   default=512,           help="LSTM hidden dim")
    p.add_argument("--lstm_layers",  type=int,   default=2,             help="LSTM layers")
    p.add_argument("--dropout",      type=float, default=0.4,           help="Dropout probability")
    p.add_argument("--focal_alpha",  type=float, default=0.75,          help="Focal loss alpha (flare weight)")
    p.add_argument("--focal_gamma",  type=float, default=2.0,           help="Focal loss gamma")
    p.add_argument("--workers",      type=int,   default=4,             help="DataLoader workers")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--freeze_cnn",   action="store_true",               help="Freeze CNN backbone initially")
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    # -----------------------------------------------------------------------
    # 1. Catalog split (stratified, no temporal leakage)
    # -----------------------------------------------------------------------
    catalog = pd.read_csv(args.catalog).sort_values("filename").reset_index(drop=True)
    n       = len(catalog)
    # Hold-out last 20% chronologically for test (prevents data leakage)
    test_split  = int(n * 0.80)
    val_split   = int(n * 0.70)

    train_df = catalog.iloc[:val_split]
    val_df   = catalog.iloc[val_split:test_split]
    test_df  = catalog.iloc[test_split:]

    log.info(f"Split | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Save temp CSVs for the subset datasets
    train_csv = os.path.join(args.output_dir, "_train_split.csv")
    val_csv   = os.path.join(args.output_dir, "_val_split.csv")
    test_csv  = os.path.join(args.output_dir, "_test_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)
    test_df.to_csv(test_csv,   index=False)

    # -----------------------------------------------------------------------
    # 2. Datasets & DataLoaders
    # -----------------------------------------------------------------------
    train_ds = SolarSequenceDataset(
        catalog_path = train_csv,
        image_dir    = args.image_dir,
        seq_len      = args.seq_len,
        augment      = True,
    )
    # Pass training normalisation stats to val/test so there's no leakage
    val_ds = SolarSequenceDataset(
        catalog_path = val_csv,
        image_dir    = args.image_dir,
        seq_len      = args.seq_len,
        augment      = False,
        tab_means    = train_ds.tab_means,
        tab_stds     = train_ds.tab_stds,
    )
    test_ds = SolarSequenceDataset(
        catalog_path = test_csv,
        image_dir    = args.image_dir,
        seq_len      = args.seq_len,
        augment      = False,
        tab_means    = train_ds.tab_means,
        tab_stds     = train_ds.tab_stds,
    )

    sampler    = build_weighted_sampler(train_ds)
    train_dl   = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, pin_memory=True)
    val_dl     = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_dl    = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # -----------------------------------------------------------------------
    # 3. Model, Loss, Optimiser, Scheduler
    # -----------------------------------------------------------------------
    model = SolarFlareSequenceModel(
        seq_len        = args.seq_len,
        tabular_dim    = 4,
        lstm_hidden    = args.lstm_hidden,
        lstm_layers    = args.lstm_layers,
        dropout        = args.dropout,
        freeze_backbone= args.freeze_cnn,
    ).to(device)

    # Initialise non-pretrained heads
    model.lstm_enc.apply(init_weights)
    model.tab_mlp.apply(init_weights)
    model.head.apply(init_weights)

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        weight_decay = 1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    scaler = GradScaler()

    # -----------------------------------------------------------------------
    # 4. Training loop
    # -----------------------------------------------------------------------
    best_auc      = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model_lstm.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_dl, criterion, optimizer, scaler, device, epoch, writer
        )
        val_loss, val_auc = validate(
            model, val_dl, criterion, device, epoch, writer
        )
        scheduler.step(val_auc)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch} done in {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_auc":     val_auc,
                "tab_means":   train_ds.tab_means,
                "tab_stds":    train_ds.tab_stds,
                "args":        vars(args),
            }, best_ckpt_path)
            log.info(f"  ★ New best AUC: {best_auc:.4f} — checkpoint saved.")

    # -----------------------------------------------------------------------
    # 5. Final test evaluation using the best checkpoint
    # -----------------------------------------------------------------------
    log.info("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    evaluate_test(model, test_dl, device)
    writer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
