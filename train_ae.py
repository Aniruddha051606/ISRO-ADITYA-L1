"""
train_ae.py
===========
Training Pipeline for the Convolutional Autoencoder Anomaly Detector
Mission: ISRO Aditya-L1

Key Principle:
  The autoencoder is trained ONLY on Class-0 (Quiet Sun) images.
  It learns to reconstruct "normal" solar morphology.
  At inference, any image with high reconstruction MSE is flagged as a
  potential flare event — no explicit flare labels needed during training.

Pipeline stages:
  1. Filter catalog to Class 0 for training
  2. Train the autoencoder to minimise reconstruction MSE
  3. Post-training: calibrate the decision threshold using the full val set
     (including flares) by computing MSE on each image
  4. Evaluate: ROC-AUC, classification report
"""

import os
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model_autoencoder import SolarAutoencoder, AnomalyScorer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — Training: Class 0 Only | Evaluation: All Classes
# ---------------------------------------------------------------------------

class SolarImageDataset(Dataset):
    """
    Single-image dataset for the autoencoder.

    Args:
        catalog_path : CSV with 'filename' and 'label' columns
        image_dir    : Root directory of PNG images
        class_filter : If 0 or 1, only return images of that class.
                       If None, return all images.
        augment      : Apply random augmentations during training.
    """

    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    def __init__(
        self,
        catalog_path: str,
        image_dir:    str,
        class_filter: Optional[int] = None,
        augment:      bool          = False,
    ):
        df = pd.read_csv(catalog_path)

        if class_filter is not None:
            df = df[df["label"] == class_filter].reset_index(drop=True)
            log.info(
                f"Dataset | class_filter={class_filter} | "
                f"{len(df)} images loaded from {catalog_path}"
            )
        else:
            log.info(f"Dataset | All classes | {len(df)} images loaded")

        self.filenames = df["filename"].tolist()
        self.labels    = df["label"].values.astype(np.float32)
        self.image_dir = Path(image_dir)

        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),           # scales to [0, 1] automatically
            # NOTE: We intentionally skip NORMALIZE here so the autoencoder
            # reconstructs true pixel values; Sigmoid output maps to [0,1].
            # If you use a pretrained encoder in the future, add normalisation.
        ])

        self.aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
        ])
        self.augment = augment

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        path  = self.image_dir / self.filenames[idx]
        img   = Image.open(path).convert("RGB")
        label = self.labels[idx]

        if self.augment:
            tensor = self.aug_transform(img)
        else:
            tensor = self.base_transform(img)

        return tensor, label    # (C, H, W), scalar


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss component.

    Pure MSE penalises pixel-wise differences uniformly.
    Adding SSIM preserves structural patterns (sunspots, active regions)
    which is critical for detecting subtle pre-flare morphology.

    Combined loss = MSE + λ * (1 - SSIM)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        # Create a Gaussian kernel
        kernel = self._gaussian_kernel(window_size, sigma)
        # (1, 1, window_size, window_size) — broadcast over channels
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g      = g / g.sum()
        return g.outer(g)

    def _ssim_per_channel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        kernel = self.kernel.repeat(1, 1, 1, 1)
        pad    = self.window_size // 2

        mu_x  = F.conv2d(x, kernel, padding=pad)
        mu_y  = F.conv2d(y, kernel, padding=pad)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_x  = F.conv2d(x ** 2, kernel, padding=pad) - mu_x2
        sig_y  = F.conv2d(y ** 2, kernel, padding=pad) - mu_y2
        sig_xy = F.conv2d(x * y,  kernel, padding=pad) - mu_xy

        numerator   = (2 * mu_xy  + self.C1) * (2 * sig_xy + self.C2)
        denominator = (mu_x2 + mu_y2 + self.C1) * (sig_x + sig_y + self.C2)
        return (numerator / denominator).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        ssim_val   = torch.stack(
            [self._ssim_per_channel(pred[:, c:c+1], target[:, c:c+1])
             for c in range(C)]
        ).mean()
        return 1.0 - ssim_val


class ReconstructionLoss(nn.Module):
    """
    Combined MSE + SSIM loss for training the autoencoder.

    λ_ssim=0.3 is a balanced default; increase to 0.5 to
    weight structural fidelity more heavily.
    """

    def __init__(self, lambda_ssim: float = 0.3):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.ssim_loss   = SSIMLoss()

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> dict:
        mse  = F.mse_loss(recon, target)
        ssim = self.ssim_loss(recon, target)
        total = mse + self.lambda_ssim * ssim
        return {"total": total, "mse": mse, "ssim": ssim}


# ---------------------------------------------------------------------------
# Training Epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     SolarAutoencoder,
    loader:    DataLoader,
    criterion: ReconstructionLoss,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    epoch:     int,
    writer:    SummaryWriter,
) -> float:
    model.train()
    total_loss = 0.0

    for step, (imgs, _) in enumerate(loader):     # Labels ignored during AE training
        imgs = imgs.to(device, non_blocking=True)  # (B, 3, 224, 224)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            recon  = model(imgs)
            losses = criterion(recon, imgs)
            loss   = losses["total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 20 == 0:
            log.info(
                f"  Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {loss.item():.5f} | "
                f"MSE: {losses['mse'].item():.5f} | "
                f"SSIM: {losses['ssim'].item():.4f}"
            )

    mean_loss = total_loss / len(loader)
    writer.add_scalar("AE_Loss/train", mean_loss, epoch)
    return mean_loss


# ---------------------------------------------------------------------------
# Validation Epoch (quiet-sun only — monitors reconstruction quality)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_recon(
    model:     SolarAutoencoder,
    loader:    DataLoader,
    criterion: ReconstructionLoss,
    device:    torch.device,
    epoch:     int,
    writer:    SummaryWriter,
) -> float:
    model.eval()
    total_loss = 0.0

    for imgs, _ in loader:
        imgs = imgs.to(device)
        with autocast():
            recon  = model(imgs)
            losses = criterion(recon, imgs)
        total_loss += losses["total"].item()

    mean_loss = total_loss / len(loader)
    writer.add_scalar("AE_Loss/val_quiet", mean_loss, epoch)
    log.info(f"  Val (quiet-sun) | Recon Loss: {mean_loss:.5f}")
    return mean_loss


# ---------------------------------------------------------------------------
# Threshold Calibration on Full Val Set (all classes)
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_and_evaluate(
    model:      SolarAutoencoder,
    loader:     DataLoader,
    device:     torch.device,
    percentile: float = 95.0,
) -> float:
    """
    1. Compute per-image MSE on the full validation set.
    2. Set threshold at `percentile` of the quiet-sun MSE distribution.
    3. Print ROC-AUC and classification report.
    Returns the calibrated threshold.
    """
    model.eval()
    all_scores = []
    all_labels = []

    for imgs, labels in loader:
        imgs  = imgs.to(device)
        recon = model(imgs)
        mse   = F.mse_loss(recon, imgs, reduction="none").mean(dim=[1, 2, 3])
        all_scores.append(mse.cpu())
        all_labels.append(labels)

    all_scores = torch.cat(all_scores)   # (N,)
    all_labels = torch.cat(all_labels)   # (N,)

    # Set threshold from quiet-sun percentile
    quiet_scores = all_scores[all_labels == 0]
    threshold    = float(torch.quantile(quiet_scores, percentile / 100.0))
    preds        = (all_scores > threshold).long().numpy()
    labels_np    = all_labels.long().numpy()
    scores_np    = all_scores.numpy()

    log.info(f"\n{'='*60}")
    log.info(f"CALIBRATION  (threshold = {threshold:.6f}  @ {percentile}th pctile)")
    log.info(f"{'='*60}")
    log.info(f"Mean MSE — Quiet Sun: {quiet_scores.mean():.6f}")
    log.info(f"Mean MSE — Flares   : {all_scores[all_labels == 1].mean():.6f}")
    log.info(f"\n{classification_report(labels_np, preds, target_names=['Quiet Sun', 'Flare'])}")

    if len(np.unique(labels_np)) > 1:
        auc = roc_auc_score(labels_np, scores_np)
        log.info(f"ROC-AUC: {auc:.4f}")
        log.info(f"Confusion Matrix:\n{confusion_matrix(labels_np, preds)}")

    return threshold


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Convolutional Autoencoder")
    p.add_argument("--catalog",      default="aditya_l1_catalog.csv", help="Full catalog CSV")
    p.add_argument("--image_dir",    default="data/images",            help="PNG image directory")
    p.add_argument("--output_dir",   default="checkpoints/autoencoder",help="Checkpoint dir")
    p.add_argument("--batch_size",   type=int,   default=32,           help="Batch size")
    p.add_argument("--epochs",       type=int,   default=60,           help="Training epochs")
    p.add_argument("--lr",           type=float, default=1e-3,         help="Learning rate")
    p.add_argument("--base_ch",      type=int,   default=64,           help="Base channels in encoder")
    p.add_argument("--latent_ch",    type=int,   default=1024,         help="Bottleneck channels")
    p.add_argument("--lambda_ssim",  type=float, default=0.3,          help="SSIM loss weight")
    p.add_argument("--threshold_p",  type=float, default=95.0,         help="Calibration percentile")
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import random
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    # -----------------------------------------------------------------------
    # 1. Chronological splits (no leakage)
    # -----------------------------------------------------------------------
    catalog = pd.read_csv(args.catalog).sort_values("filename").reset_index(drop=True)
    n       = len(catalog)
    val_cut  = int(n * 0.70)
    test_cut = int(n * 0.80)

    train_df = catalog.iloc[:val_cut]
    val_df   = catalog.iloc[val_cut:test_cut]
    test_df  = catalog.iloc[test_cut:]

    train_csv = os.path.join(args.output_dir, "_train.csv")
    val_csv   = os.path.join(args.output_dir, "_val.csv")
    test_csv  = os.path.join(args.output_dir, "_test.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)
    test_df.to_csv(test_csv,   index=False)

    log.info(
        f"Split | Train (class 0 only): {(train_df['label']==0).sum()} | "
        f"Val: {len(val_df)} | Test: {len(test_df)}"
    )

    # -----------------------------------------------------------------------
    # 2. Datasets & DataLoaders
    # -----------------------------------------------------------------------
    # Training dataset: ONLY Class 0 (Quiet Sun)
    train_ds = SolarImageDataset(
        train_csv, args.image_dir, class_filter=0, augment=True
    )
    # Validation datasets — one quiet-only (for recon monitoring), one full
    val_quiet_ds = SolarImageDataset(
        val_csv, args.image_dir, class_filter=0, augment=False
    )
    val_full_ds = SolarImageDataset(
        val_csv, args.image_dir, class_filter=None, augment=False
    )
    test_full_ds = SolarImageDataset(
        test_csv, args.image_dir, class_filter=None, augment=False
    )

    train_dl     = DataLoader(train_ds,     batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_quiet_dl = DataLoader(val_quiet_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    val_full_dl  = DataLoader(val_full_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    test_full_dl = DataLoader(test_full_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # -----------------------------------------------------------------------
    # 3. Model, Loss, Optimiser, Scheduler
    # -----------------------------------------------------------------------
    model = SolarAutoencoder(
        in_channels     = 3,
        base_channels   = args.base_ch,
        latent_channels = args.latent_ch,
    ).to(device)

    criterion = ReconstructionLoss(lambda_ssim=args.lambda_ssim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler()

    # -----------------------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------------------
    best_val_loss   = float("inf")
    best_ckpt_path  = os.path.join(args.output_dir, "best_autoencoder.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_dl, criterion, optimizer, scaler, device, epoch, writer
        )
        val_loss = validate_recon(
            model, val_quiet_dl, criterion, device, epoch, writer
        )
        scheduler.step()

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch} | {elapsed:.1f}s | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "args":        vars(args),
            }, best_ckpt_path)
            log.info(f"  ★ New best val loss: {best_val_loss:.5f} — saved.")

        # Every 10 epochs: quick anomaly check on full val set
        if epoch % 10 == 0:
            log.info("  Running mid-training calibration check...")
            calibrate_and_evaluate(model, val_full_dl, device, args.threshold_p)

    # -----------------------------------------------------------------------
    # 5. Final calibration + test evaluation
    # -----------------------------------------------------------------------
    log.info("\nLoading best checkpoint...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    log.info("\n--- Validation Set Calibration ---")
    threshold = calibrate_and_evaluate(
        model, val_full_dl, device, args.threshold_p
    )

    log.info("\n--- Test Set Final Evaluation ---")
    calibrate_and_evaluate(model, test_full_dl, device, args.threshold_p)

    # Save threshold alongside the model
    torch.save({
        **ckpt,
        "anomaly_threshold": threshold,
    }, best_ckpt_path)
    log.info(f"Final threshold {threshold:.6f} saved to checkpoint.")

    writer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
