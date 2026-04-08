"""
train_vae.py  ─  Enterprise-Grade VAE Training Pipeline
=========================================================
Mission  : ISRO Aditya-L1 Solar Flare Anomaly Detection
Dataset  : ~77 000 FITS→PNG images

ARCHITECTURAL UPGRADES (v2)
────────────────────────────
  1. Selective Mixed Precision (AMP) & Stability
       • GradScaler + autocast for encoder/decoder forward pass
       • Loss calculation (MSE, SSIM, KL, Physics) explicitly upcast to FP32
       • Latent clamping: mu ∈ [-15, 50], logvar ∈ [-15, 50]
       • Gradient clipping (max_norm=5.0) to prevent exploding gradients

  2. Physics-Informed Loss (PINN)
       • Imports SolarPhysicsLoss from model_pinn.py
       • Total ELBO = MSE + λ_ssim·SSIM + λ_phys·PINN + β·KL

  3. TensorBoard Integration
       • Logs LR, Total Loss, Recon Loss, KL Loss, Physics Loss per epoch
       • Scalar group: "Train/", "Val/", "Hyperparams/"

  4. Early Stopping & Checkpointing
       • EarlyStopping class — patience=8, monitors val_loss
       • Saves best_vae.pt automatically on improvement

  5. Denoising VAE (DVAE)
       • Gaussian noise (μ=0, σ=0.05) injected into encoder input
       • Loss computed against clean originals → forces robust latent space

USAGE:
    python train_vae.py \
        --catalog  aditya_l1_catalog.csv \
        --image_dir data/images \
        --epochs    60 \
        --batch_size 32 \
        --latent_dim 512 \
        --beta_max   1e-4 \
        --lambda_phys 0.1

OUTPUTS (inside --output_dir):
    best_vae.pt            ← best checkpoint + calibrated threshold
    latent_train_mu.npy    ← μ vectors for OCSVM (ensemble_detector.py)
    latent_train_labels.npy
    umap_e<N>.png          ← latent-space visualisations (every 10 epochs)
    tb/                    ← TensorBoard event files
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import os
import time
import random
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Dynamic directory setup  (resolved relative to this file's parent)
# ──────────────────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent
PROJECT_DIR    = _HERE.parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "vae"
LOG_DIR        = PROJECT_DIR / "logs"
DATA_DIR       = PROJECT_DIR / "data"
IMAGE_DIR      = DATA_DIR / "images"
CATALOG_PATH   = PROJECT_DIR / "aditya_l1_catalog.csv"
TB_LOG_DIR     = CHECKPOINT_DIR / "tb"

for _d in (CHECKPOINT_DIR, LOG_DIR, DATA_DIR, IMAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# ──────────────────────────────────────────────────────────────────────────────
# Project modules
# ──────────────────────────────────────────────────────────────────────────────
from model_vae import SolarVAE, VAEAnomalyScorer   # your existing VAE
from model_pinn import SolarPhysicsLoss             # ← UPGRADE 2 (PINN)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train_vae.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  1.  EARLY STOPPING                                          (UPGRADE 4)
# ╚══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Monitors a scalar metric and signals when training should stop.

    Parameters
    ----------
    patience   : int   — epochs to wait after last improvement
    min_delta  : float — minimum change to count as improvement
    mode       : str   — 'min' (loss) or 'max' (AUC / accuracy)
    save_path  : Path  — if provided, saves the best model state dict here
    """

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-6,
        mode: str = "min",
        save_path: Optional[Path] = None,
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.save_path  = save_path

        self.best_score : Optional[float] = None
        self.counter    : int             = 0
        self.stop       : bool            = False

        self._cmp = (lambda a, b: a < b - min_delta) if mode == "min" \
               else (lambda a, b: a > b + min_delta)

    # ------------------------------------------------------------------
    def step(self, score: float, model: nn.Module, extra_state: dict = None) -> bool:
        """
        Call once per epoch.

        Returns True when training should stop.
        Automatically saves best_vae.pt when improvement is detected.
        """
        if self.best_score is None or self._cmp(score, self.best_score):
            log.info(
                f"  ★ EarlyStopping: improvement "
                f"{'' if self.best_score is None else f'{self.best_score:.6f} →'} "
                f"{score:.6f}  (patience counter reset)"
            )
            self.best_score = score
            self.counter    = 0

            if self.save_path is not None:
                payload = {
                    "model_state": model.state_dict(),
                    "best_val_loss": score,
                }
                if extra_state:
                    payload.update(extra_state)
                torch.save(payload, self.save_path)
                log.info(f"  ★ Checkpoint saved → {self.save_path}")
        else:
            self.counter += 1
            log.info(
                f"  EarlyStopping: no improvement "
                f"({self.counter}/{self.patience})"
            )
            if self.counter >= self.patience:
                log.info("  EarlyStopping: patience exhausted — halting training.")
                self.stop = True

        return self.stop


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  2.  VAE LOSS  (FP32 safe + PINN term)        (UPGRADES 1, 2)
# ╚══════════════════════════════════════════════════════════════════════════════

class VAELoss(nn.Module):
    """
    Total ELBO loss:

        L = MSE  +  λ_ssim · SSIM_loss  +  λ_phys · PINN  +  β(t) · KL

    All sub-losses are computed in FP32 regardless of AMP autocast context.

    β-annealing: linear warm-up over `anneal_epochs`, then constant `beta_max`.
    """

    def __init__(
        self,
        beta_max      : float = 1e-4,
        anneal_epochs : int   = 20,
        lambda_ssim   : float = 0.3,
        lambda_phys   : float = 0.1,
        use_perceptual: bool  = True,
        physics_loss  : Optional[nn.Module] = None,
    ):
        super().__init__()
        self.beta_max       = beta_max
        self.anneal_epochs  = anneal_epochs
        self.lambda_ssim    = lambda_ssim
        self.lambda_phys    = lambda_phys
        self.use_perceptual = use_perceptual

        # Physics loss injected from outside so the module can be None-safe
        self.physics_loss: Optional[nn.Module] = physics_loss

        # SSIM window (fixed, not learned)
        self.register_buffer(
            "ssim_window",
            self._gaussian_window(11, 1.5).unsqueeze(0).unsqueeze(0),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g)

    # ------------------------------------------------------------------
    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Structural Similarity loss computed in FP32.
        Returned as (1 - SSIM) so it can be minimised.
        """
        # ── UPGRADE 1: explicit FP32 upcast ──────────────────────────
        pred   = pred.float()
        target = target.float()

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        win = self.ssim_window.expand(pred.size(1), 1, -1, -1)

        mu_p  = F.conv2d(pred,   win, padding=5, groups=pred.size(1))
        mu_t  = F.conv2d(target, win, padding=5, groups=target.size(1))

        mu_p2 = mu_p * mu_p
        mu_t2 = mu_t * mu_t
        mu_pt = mu_p * mu_t

        # Variance — clamped to ≥ 0 to prevent NaN sqrt / negative values
        sig_p  = (F.conv2d(pred * pred,     win, padding=5, groups=pred.size(1))   - mu_p2).clamp(min=0)
        sig_t  = (F.conv2d(target * target, win, padding=5, groups=target.size(1)) - mu_t2).clamp(min=0)
        sig_pt =  F.conv2d(pred * target,   win, padding=5, groups=pred.size(1))   - mu_pt

        ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / \
                   ((mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2))
        return (1.0 - ssim_map).mean()

    # ------------------------------------------------------------------
    def _beta(self, epoch: int) -> float:
        """Linear β warm-up schedule."""
        return min(self.beta_max, self.beta_max * epoch / max(1, self.anneal_epochs))

    # ------------------------------------------------------------------
    def forward(
        self,
        recon   : torch.Tensor,   # decoder output           (B, C, H, W)
        target  : torch.Tensor,   # clean original images    (B, C, H, W)
        mu      : torch.Tensor,   # ← UPGRADE 1: clamped in encoder
        logvar  : torch.Tensor,   # ← UPGRADE 1: clamped in encoder
        epoch   : int = 1,
        raw_imgs: Optional[torch.Tensor] = None,  # for PINN (original flux)
    ) -> Dict[str, torch.Tensor]:

        # ── UPGRADE 1: ensure FP32 throughout loss calc ──────────────
        recon  = recon.float()
        target = target.float()
        mu     = mu.float()
        logvar = logvar.float()

        # ── Reconstruction terms ──────────────────────────────────────
        mse_loss  = F.mse_loss(recon, target, reduction="mean")
        ssim_loss = self._ssim_loss(recon, target)
        recon_loss = mse_loss + self.lambda_ssim * ssim_loss

        # ── KL divergence (FP32-safe) ─────────────────────────────────
        # -0.5 * Σ(1 + logvar - μ² - exp(logvar))
        kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

        # ── UPGRADE 2: Physics-Informed loss ─────────────────────────
        phys_loss = torch.tensor(0.0, device=recon.device, dtype=torch.float32)
        if self.physics_loss is not None:
            imgs_for_pinn = raw_imgs.float() if raw_imgs is not None else target
            phys_loss = self.physics_loss(recon, imgs_for_pinn).float()

        # ── UPGRADE 2: combined ELBO ──────────────────────────────────
        beta  = self._beta(epoch)
        total = recon_loss \
              + self.lambda_phys * phys_loss \
              + beta * kl_loss

        return {
            "total"  : total,
            "recon"  : recon_loss,
            "mse"    : mse_loss,
            "ssim"   : ssim_loss,
            "kl"     : kl_loss,
            "physics": phys_loss,
            "beta"   : torch.tensor(beta, device=recon.device),
        }


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  3.  DATASET
# ╚══════════════════════════════════════════════════════════════════════════════

class SolarImageDataset(Dataset):
    """
    Loads solar PNG images from a CSV catalog.

    Parameters
    ----------
    catalog_path : str / Path — CSV with columns ['filename', 'label']
    image_dir    : str / Path — root directory containing the PNG files
    class_filter : int | None — 0 = quiet sun only, 1 = flare only, None = all
    augment      : bool       — enable geometric / photometric augmentation
    """

    _BASE_TF = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    _AUG_TF = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    def __init__(
        self,
        catalog_path : "str | Path",
        image_dir    : "str | Path",
        class_filter : Optional[int] = None,
        augment      : bool = False,
    ):
        df = pd.read_csv(catalog_path, low_memory=False)
        if class_filter is not None:
            df = df[df["label"] == class_filter].reset_index(drop=True)

        self.filenames = df["filename"].tolist()
        self.labels    = df["label"].values.astype(np.float32)
        self.image_dir = Path(image_dir)
        self.augment   = augment
        self._tf       = self._AUG_TF if augment else self._BASE_TF

        log.info(
            f"Dataset | filter={class_filter} | augment={augment} | "
            f"n={len(df):,} images"
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        img = Image.open(self.image_dir / self.filenames[idx]).convert("RGB")
        return self._tf(img), self.labels[idx]


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  4.  TRAIN ONE EPOCH
# ╚══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model     : SolarVAE,
    loader    : DataLoader,
    criterion : VAELoss,
    optimizer : torch.optim.Optimizer,
    scaler    : GradScaler,
    device    : torch.device,
    epoch     : int,
    writer    : SummaryWriter,
    noise_std : float = 0.05,          # UPGRADE 5: DVAE noise level
) -> Dict[str, float]:
    """
    One training epoch.

    Denoising VAE (UPGRADE 5):
      Gaussian noise is added to imgs → noisy_imgs before the encoder.
      The loss is still computed against the original clean imgs.

    AMP (UPGRADE 1):
      Forward pass runs under autocast (FP16/BF16).
      Loss computation is in FP32 (handled inside VAELoss.forward).
      Gradients are scaled/unscaled via GradScaler.
    """
    model.train()
    running: Dict[str, float] = {
        "total": 0.0, "recon": 0.0, "kl": 0.0, "physics": 0.0
    }
    n_steps = len(loader)

    for step, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)

        # ── UPGRADE 5: inject noise, keep clean target ────────────────
        noise      = torch.randn_like(imgs) * noise_std
        noisy_imgs = (imgs + noise).clamp(0.0, 1.0)   # stay in valid range

        optimizer.zero_grad(set_to_none=True)

        # ── UPGRADE 1: AMP forward pass ───────────────────────────────
        with autocast("cuda"):
            recon, mu, logvar = model(noisy_imgs)  # encode noisy

        # ── UPGRADE 1: latent clamping ────────────────────────────────
        mu     = mu.float().clamp(min=-15.0, max=50.0)
        logvar = logvar.float().clamp(min=-15.0, max=50.0)
        recon  = recon.float()

        # ── UPGRADE 1 + 2: FP32 loss with PINN ───────────────────────
        losses = criterion(
            recon    = recon,
            target   = imgs,          # ← clean original as target (DVAE)
            mu       = mu,
            logvar   = logvar,
            epoch    = epoch,
            raw_imgs = imgs,          # pass to PINN for physics residuals
        )
        loss = losses["total"]

        # ── UPGRADE 1: scaled backward + gradient clipping ───────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        for k in running:
            running[k] += losses[k].item()

        if step % 50 == 0:
            log.info(
                f"  Epoch {epoch:03d} [{step:04d}/{n_steps}] "
                f"total={losses['total'].item():.5f}  "
                f"mse={losses['mse'].item():.5f}  "
                f"ssim={losses['ssim'].item():.5f}  "
                f"kl={losses['kl'].item():.5f}  "
                f"phys={losses['physics'].item():.5f}  "
                f"β={losses['beta'].item():.6f}"
            )

    # ── Average over steps ────────────────────────────────────────────
    avg = {k: v / n_steps for k, v in running.items()}

    # ── UPGRADE 3: TensorBoard ────────────────────────────────────────
    writer.add_scalar("Train/loss_total",   avg["total"],   epoch)
    writer.add_scalar("Train/loss_recon",   avg["recon"],   epoch)
    writer.add_scalar("Train/loss_kl",      avg["kl"],      epoch)
    writer.add_scalar("Train/loss_physics", avg["physics"], epoch)

    return avg


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  5.  VALIDATION EPOCH (quiet-sun images)
# ╚══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_quiet(
    model     : SolarVAE,
    loader    : DataLoader,
    criterion : VAELoss,
    device    : torch.device,
    epoch     : int,
    writer    : SummaryWriter,
) -> float:
    """
    Validates on quiet-sun images only (class 0).
    Returns mean total loss (used by EarlyStopping).
    """
    model.eval()
    total = 0.0

    for imgs, _ in loader:
        imgs = imgs.to(device)

        with autocast("cuda"):
            recon, mu, logvar = model(imgs)

        mu     = mu.float().clamp(min=-15.0, max=50.0)
        logvar = logvar.float().clamp(min=-15.0, max=50.0)
        recon  = recon.float()

        losses = criterion(recon, imgs, mu, logvar, epoch=epoch, raw_imgs=imgs)
        total += losses["total"].item()

    mean_loss = total / len(loader)

    # ── UPGRADE 3: TensorBoard ────────────────────────────────────────
    writer.add_scalar("Val/loss_quiet_total", mean_loss, epoch)
    log.info(f"  Val quiet-sun loss: {mean_loss:.6f}")
    return mean_loss


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  6.  CALIBRATION + EVALUATION
# ╚══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calibrate_evaluate(
    model      : SolarVAE,
    loader     : DataLoader,
    device     : torch.device,
    percentile : float = 95.0,
    tag        : str   = "Val",
    writer     : Optional[SummaryWriter] = None,
    epoch      : int   = 0,
) -> float:
    """
    Computes anomaly scores, derives threshold, logs metrics.
    Returns the calibrated threshold value.
    """
    model.eval()
    all_scores, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        # anomaly_score = pixel-wise MSE between clean and reconstructed
        with autocast("cuda"):
            recon, mu, _ = model(imgs)
        mu    = mu.float().clamp(min=-15.0, max=50.0)
        recon = recon.float()
        scores = F.mse_loss(recon, imgs.float(), reduction="none").mean(dim=[1, 2, 3])
        all_scores.append(scores.cpu())
        all_labels.append(labels)

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).long().numpy()

    quiet_scores = scores[labels == 0]
    threshold    = float(np.percentile(quiet_scores, percentile))
    preds        = (scores > threshold).astype(int)

    log.info(
        f"\n{'='*60}\n"
        f"{tag} SET  |  Threshold @ {percentile}th pctile = {threshold:.6f}\n"
        f"  Quiet MSE : {quiet_scores.mean():.5f} ± {quiet_scores.std():.5f}"
    )

    if (labels == 1).any():
        flare_scores = scores[labels == 1]
        log.info(f"  Flare MSE : {flare_scores.mean():.5f} ± {flare_scores.std():.5f}")

    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, scores)
        log.info(f"  ROC-AUC   : {auc:.4f}")
        log.info(f"\n{classification_report(labels, preds, target_names=['Quiet', 'Flare'])}")
        log.info(f"Confusion matrix:\n{confusion_matrix(labels, preds)}")
        if writer is not None:
            writer.add_scalar(f"{tag}/roc_auc", auc, epoch)

    if writer is not None:
        writer.add_scalar(f"{tag}/anomaly_threshold", threshold, epoch)

    return threshold


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  7.  LATENT VECTOR EXTRACTION  (for OCSVM ensemble)
# ╚══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_latent_vectors(
    model    : SolarVAE,
    loader   : DataLoader,
    device   : torch.device,
    out_path : "str | Path",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the encoder over `loader`, saves μ and labels as .npy files.
    Used by ensemble_detector.py to train the One-Class SVM.
    """
    model.eval()
    all_mu, all_labels = [], []

    for imgs, labels in loader:
        mu, _ = model.encode(imgs.to(device))
        mu = mu.float().clamp(min=-15.0, max=50.0)
        all_mu.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())

    mu_arr  = np.concatenate(all_mu,     axis=0)
    lab_arr = np.concatenate(all_labels, axis=0)

    out_path  = Path(out_path)
    mu_path   = out_path.with_name(out_path.stem + "_mu.npy")
    lab_path  = out_path.with_name(out_path.stem + "_labels.npy")

    np.save(mu_path,  mu_arr)
    np.save(lab_path, lab_arr)
    log.info(f"Latent vectors saved: {mu_arr.shape} → {mu_path}")
    return mu_arr, lab_arr


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  8.  UMAP VISUALISATION  (optional)
# ╚══════════════════════════════════════════════════════════════════════════════

def plot_latent_umap(
    mu_arr   : np.ndarray,
    label_arr: np.ndarray,
    save_path: "str | Path",
    epoch    : int,
    max_pts  : int = 2000,
) -> None:
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        idx      = np.random.choice(len(mu_arr), min(max_pts, len(mu_arr)), replace=False)
        z2d      = umap.UMAP(n_components=2, random_state=42, n_neighbors=30)\
                       .fit_transform(mu_arr[idx])
        labs_sub = label_arr[idx]

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="#080f1c")
        ax.set_facecolor("#080f1c")
        for cls, colour, name in [(0, "#4fc3f7", "Quiet Sun"), (1, "#ff5252", "Flare")]:
            m = labs_sub == cls
            if m.any():
                ax.scatter(z2d[m, 0], z2d[m, 1], c=colour,
                           label=name, alpha=0.65, s=9, linewidths=0)

        ax.set_title(f"VAE Latent Space (UMAP)  —  Epoch {epoch}",
                     color="white", fontsize=13, pad=12)
        ax.legend(facecolor="#0d1b2a", labelcolor="white", framealpha=0.8)
        ax.tick_params(colors="grey")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, facecolor="#080f1c")
        plt.close(fig)
        log.info(f"UMAP saved → {save_path}")

    except ImportError:
        log.warning("umap-learn not installed — skipping UMAP visualisation.")


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  9.  ARGUMENT PARSER
# ╚══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Solar VAE Anomaly Detector (v2)")

    # -- Data
    p.add_argument("--catalog",       default=str(CATALOG_PATH))
    p.add_argument("--image_dir",     default=str(IMAGE_DIR))
    p.add_argument("--output_dir",    default=str(CHECKPOINT_DIR))
    p.add_argument("--log_dir",       default=str(LOG_DIR))

    # -- Training
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--workers",       type=int,   default=4)

    # -- Model
    p.add_argument("--latent_dim",    type=int,   default=512)
    p.add_argument("--base_ch",       type=int,   default=64)

    # -- Loss weights
    p.add_argument("--beta_max",      type=float, default=1e-4,
                   help="Max KL weight (β) after annealing")
    p.add_argument("--anneal_epochs", type=int,   default=20,
                   help="Epochs over which β linearly ramps to beta_max")
    p.add_argument("--lambda_ssim",   type=float, default=0.3)
    p.add_argument("--lambda_phys",   type=float, default=0.1,
                   help="Weight for SolarPhysicsLoss (PINN) term")
    p.add_argument("--threshold_p",   type=float, default=95.0,
                   help="Percentile of quiet-sun scores used as anomaly threshold")

    # -- DVAE noise
    p.add_argument("--noise_std",     type=float, default=0.05,
                   help="Std-dev of Gaussian noise injected for DVAE training")

    # -- Early stopping
    p.add_argument("--patience",      type=int,   default=8,
                   help="EarlyStopping patience (epochs)")

    # -- Flags
    p.add_argument("--no_perceptual", action="store_true",
                   help="Disable perceptual loss (faster, slightly less accurate)")
    p.add_argument("--no_pinn",       action="store_true",
                   help="Disable physics-informed loss (for ablation studies)")

    return p.parse_args()


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  10.  MAIN
# ╚══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")
    log.info(f"Args   : {vars(args)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── UPGRADE 3: TensorBoard writer ────────────────────────────────
    writer = SummaryWriter(log_dir=str(TB_LOG_DIR))
    writer.add_text("config", str(vars(args)), global_step=0)

    # ── Chronological splits (70 / 10 / 20) ─────────────────────────
    catalog  = pd.read_csv(args.catalog, low_memory=False)\
                 .sort_values("filename").reset_index(drop=True)
    n        = len(catalog)
    n_train  = int(n * 0.70)
    n_val    = int(n * 0.80)

    train_df = catalog.iloc[:n_train]
    val_df   = catalog.iloc[n_train:n_val]
    test_df  = catalog.iloc[n_val:]

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_path = out_dir / f"_split_{name}.csv"
        df.to_csv(split_path, index=False)
        log.info(f"Split {name:5s}: {len(df):6,} rows → {split_path}")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds     = SolarImageDataset(out_dir / "_split_train.csv", args.image_dir, class_filter=0,    augment=True)
    val_quiet_ds = SolarImageDataset(out_dir / "_split_val.csv",   args.image_dir, class_filter=0,    augment=False)
    val_full_ds  = SolarImageDataset(out_dir / "_split_val.csv",   args.image_dir, class_filter=None, augment=False)
    test_full_ds = SolarImageDataset(out_dir / "_split_test.csv",  args.image_dir, class_filter=None, augment=False)

    kw = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"),
              persistent_workers=(args.workers > 0))
    train_dl     = DataLoader(train_ds,     args.batch_size, shuffle=True,  **kw)
    val_quiet_dl = DataLoader(val_quiet_ds, args.batch_size, shuffle=False, **kw)
    val_full_dl  = DataLoader(val_full_ds,  args.batch_size, shuffle=False, **kw)
    test_full_dl = DataLoader(test_full_ds, args.batch_size, shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────
    model = SolarVAE(
        in_channels = 3,
        base_ch     = args.base_ch,
        latent_dim  = args.latent_dim,
    ).to(device)
    log.info(
        f"SolarVAE params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # ── UPGRADE 2: Physics-Informed Loss ─────────────────────────────
    physics_loss: Optional[SolarPhysicsLoss] = None
    if not args.no_pinn:
        physics_loss = SolarPhysicsLoss().to(device)
        log.info("SolarPhysicsLoss enabled (PINN)")
    else:
        log.info("SolarPhysicsLoss disabled (--no_pinn flag)")

    # ── Loss, optimiser, scheduler, scaler ───────────────────────────
    criterion = VAELoss(
        beta_max       = args.beta_max,
        anneal_epochs  = args.anneal_epochs,
        lambda_ssim    = args.lambda_ssim,
        lambda_phys    = args.lambda_phys,
        use_perceptual = not args.no_perceptual,
        physics_loss   = physics_loss,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── UPGRADE 1: AMP GradScaler ─────────────────────────────────────
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── UPGRADE 4: Early Stopping ─────────────────────────────────────
    best_ckpt    = out_dir / "best_vae.pt"
    early_stop   = EarlyStopping(
        patience  = args.patience,
        mode      = "min",
        save_path = best_ckpt,
    )

    # ── Training loop ─────────────────────────────────────────────────
    log.info(f"\n{'='*60}\nStarting training for {args.epochs} epochs\n{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log.info(f"\nEpoch {epoch}/{args.epochs}")

        train_avgs = train_epoch(
            model, train_dl, criterion, optimizer,
            scaler, device, epoch, writer,
            noise_std=args.noise_std,          # UPGRADE 5: DVAE
        )

        val_loss = validate_quiet(
            model, val_quiet_dl, criterion, device, epoch, writer
        )

        # ── UPGRADE 3: log LR ──────────────────────────────────────
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Hyperparams/learning_rate", current_lr, epoch)
        log.info(
            f"  LR={current_lr:.2e}  "
            f"train_total={train_avgs['total']:.5f}  "
            f"val_quiet={val_loss:.5f}  "
            f"[{time.time() - t0:.1f}s]"
        )

        scheduler.step()

        # ── UPGRADE 4: early stopping step ────────────────────────
        extra_state = {
            "epoch"      : epoch,
            "args"       : vars(args),
            "train_loss" : train_avgs["total"],
        }
        if early_stop.step(val_loss, model, extra_state):
            log.info("Early stopping triggered — exiting training loop.")
            break

        # ── Every 10 epochs: full calibration + UMAP ──────────────
        if epoch % 10 == 0:
            threshold = calibrate_evaluate(
                model, val_full_dl, device,
                percentile=args.threshold_p, tag="Val",
                writer=writer, epoch=epoch,
            )
            mu_arr, lab_arr = save_latent_vectors(
                model, val_full_dl, device,
                out_dir / f"latent_val_e{epoch}.npy"
            )
            plot_latent_umap(
                mu_arr, lab_arr,
                save_path=out_dir / f"umap_e{epoch}.png",
                epoch=epoch,
            )

    # ── Final evaluation ──────────────────────────────────────────────
    log.info(f"\n{'='*60}\nFinal evaluation — loading {best_ckpt}\n{'='*60}")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_threshold  = calibrate_evaluate(
        model, val_full_dl,  device, args.threshold_p,
        tag="Val",  writer=writer, epoch=args.epochs
    )
    test_threshold = calibrate_evaluate(
        model, test_full_dl, device, args.threshold_p,
        tag="Test", writer=writer, epoch=args.epochs
    )

    # ── Save train latent vectors for OCSVM ──────────────────────────
    train_dl_full = DataLoader(train_ds, args.batch_size, shuffle=False, **kw)
    save_latent_vectors(
        model, train_dl_full, device,
        out_dir / "latent_train.npy"
    )

    # ── Final UMAP ───────────────────────────────────────────────────
    mu_arr, lab_arr = save_latent_vectors(
        model, val_full_dl, device,
        out_dir / "latent_val_final.npy"
    )
    plot_latent_umap(mu_arr, lab_arr, out_dir / "umap_final.png", epoch=args.epochs)

    # ── Re-save checkpoint with calibrated threshold ─────────────────
    torch.save(
        {**ckpt, "anomaly_threshold": val_threshold,
         "test_threshold": test_threshold},
        best_ckpt,
    )
    log.info(
        f"\nFinal checkpoint: {best_ckpt}"
        f"\n  val_threshold  = {val_threshold:.6f}"
        f"\n  test_threshold = {test_threshold:.6f}"
    )

    writer.close()
    log.info("Training complete ✓")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
