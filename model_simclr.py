"""
model_simclr.py
===============
SimCLR Contrastive Learning for Multi-Band SUIT Solar Images
Mission: ISRO Aditya-L1

WHY THIS BEATS EFFICIENTNET-B0 IMAGENET PRETRAINING
─────────────────────────────────────────────────────
EfficientNet-B0 was pretrained on ImageNet — photos of dogs, cars, furniture.
Its low-level features (edges, textures) transfer reasonably, but its
high-level representations (what makes a "flare" vs "quiet sun") are
meaningless for solar physics.

You have something better: 11 SUIT filter bands photographing the EXACT
SAME solar scene at the EXACT SAME timestamp. This is a natural positive
pair for contrastive learning:

  NB02 (2796Å) of timestamp T  ←→  NB04 (2803Å) of timestamp T
                           SAME EVENT → similar embedding

  NB02 (2796Å) of timestamp T  ←→  NB02 (2796Å) of timestamp T+1day
                          DIFFERENT EVENT → different embedding

SimCLR learns: "pull same-event embeddings together, push
different-event embeddings apart." The result is a solar-specific
representation that understands active regions, filaments, quiet sun,
and the differences between them — learned purely from your 53,329 frames
with ZERO labels.

This backbone replaces EfficientNet-B0 in model_lstm_v2.py, giving the
LSTM a far stronger starting point.

ARCHITECTURE
────────────
  Input: two augmented views of the same solar event
  Encoder f(·): ResNet-50 backbone → 2048-dim features
  Projector g(·): 2-layer MLP → 128-dim unit-hypersphere

  Loss: NT-Xent (Normalised Temperature-scaled Cross Entropy)
  For batch of N events → 2N samples → 2N(N-1) negative pairs

MULTI-BAND POSITIVE PAIRS
──────────────────────────
  Strategy A (same-timestamp, different-band): NB02_T vs NB04_T
    → Teaches band invariance + event-level representation
  Strategy B (same-band, augmentation): NB02_T + augment vs NB02_T + augment
    → Standard SimCLR, teaches augmentation invariance
  Strategy C (combined): both strategies in one batch [BEST]
    → Richest signal, used here
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "simclr")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
IMAGE_DIR      = os.path.join(DATA_DIR, "images")
CATALOG_PATH   = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solar-Specific Augmentation Pipeline
# ---------------------------------------------------------------------------

class SolarAugmentation:
    """
    Augmentation pipeline designed for solar UV/NUV images.

    Key differences from standard SimCLR augmentations:
      - NO random grayscale (images are already grayscale-ish, single-channel physics)
      - Rotation is FULL 360° (Sun is rotationally symmetric in disk observations)
      - Color jitter is very mild (flux calibration should be preserved)
      - Gaussian blur mimics atmospheric seeing / instrument PSF variation
      - Solar-disk crop: random crop inside the solar disk, not outside it
    """

    def __init__(self, image_size: int = 224, strength: float = 1.0):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),        # Crop 50-100% of image area
                ratio=(0.9, 1.1),        # Near-square crops (disk is circular)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),  # Full disk symmetry
            transforms.ColorJitter(
                brightness = 0.15 * strength,
                contrast   = 0.15 * strength,
                saturation = 0.05 * strength,   # Very mild — preserve band calibration
                hue        = 0.0,               # No hue shift for monochromatic UV
            ),
            transforms.GaussianBlur(
                kernel_size = int(0.1 * image_size) | 1,  # Must be odd
                sigma       = (0.1, 2.0 * strength),
            ),
            transforms.ToTensor(),
            # Normalise to solar image stats (approx — fine-tune on your data)
            transforms.Normalize(mean=[0.3, 0.3, 0.3], std=[0.2, 0.2, 0.2]),
        ])

    def __call__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns two independently augmented views of the same image."""
        return self.transform(x), self.transform(x)


# ---------------------------------------------------------------------------
# Multi-Band Pair Dataset
# ---------------------------------------------------------------------------

class MultiBandPairDataset(Dataset):
    """
    Dataset that returns positive pairs for contrastive learning.

    Strategy C (combined):
      50% of the time → same event, different band (inter-band pair)
      50% of the time → same event, same band, different augmentation (intra-band pair)

    Groups catalog by T_OBS timestamp (rounded to nearest minute) so
    all bands of the same observation are considered the same event.
    """

    def __init__(
        self,
        catalog_path: str   = CATALOG_PATH,
        image_dir:    str   = IMAGE_DIR,
        image_size:   int   = 224,
        aug_strength: float = 1.0,
        interband_prob: float = 0.5,
    ):
        df = pd.read_csv(catalog_path, low_memory=False)

        # Normalise column names
        df.columns = df.columns.str.strip()

        # Find filename column
        fname_col = next((c for c in ["Filename","filename","FILENAME"] if c in df.columns), None)
        if fname_col is None:
            raise ValueError("No filename column found in catalog")
        df = df.rename(columns={fname_col: "filename"})

        # Ensure .png extension
        df["filename"] = df["filename"].astype(str).apply(
            lambda x: x.rsplit(".", 1)[0] + ".png"
        )

        # Keep only existing images
        image_dir_p = Path(image_dir)
        df = df[df["filename"].apply(lambda f: (image_dir_p / f).exists())]
        df = df.reset_index(drop=True)

        log.info(f"[SimCLR Dataset] {len(df)} valid images")

        # Group by T_OBS (same timestamp = same solar event, different band)
        if "T_OBS" in df.columns:
            # Round to nearest 10 seconds to group same-event multi-band frames
            df["t_obs_round"] = pd.to_datetime(
                df["T_OBS"], errors="coerce"
            ).dt.round("10s").astype(str)
            # Build event groups: {event_key: [list of filenames]}
            self.event_groups = (
                df.groupby("t_obs_round")["filename"]
                .apply(list)
                .to_dict()
            )
            self.event_keys = list(self.event_groups.keys())
        else:
            # Fallback: every image is its own event (intra-band aug only)
            self.event_groups = {str(i): [row] for i, row in enumerate(df["filename"])}
            self.event_keys   = list(self.event_groups.keys())

        self.image_dir      = image_dir_p
        self.augmentation   = SolarAugmentation(image_size, aug_strength)
        self.interband_prob = interband_prob
        log.info(f"[SimCLR Dataset] {len(self.event_keys)} unique events")

    def __len__(self) -> int:
        return len(self.event_keys)

    def _load(self, filename: str) -> Image.Image:
        path = self.image_dir / filename
        img  = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key   = self.event_keys[idx]
        group = self.event_groups[key]

        use_interband = (
            len(group) > 1
            and np.random.random() < self.interband_prob
        )

        if use_interband:
            # Pick two DIFFERENT band images of the same event
            indices = np.random.choice(len(group), 2, replace=False)
            img1 = self._load(group[indices[0]])
            img2 = self._load(group[indices[1]])
            view1 = self.augmentation.transform(img1)
            view2 = self.augmentation.transform(img2)
        else:
            # Standard SimCLR: two augmented views of the same image
            img        = self._load(group[np.random.randint(len(group))])
            view1, view2 = self.augmentation(img)

        return view1, view2


# ---------------------------------------------------------------------------
# SimCLR Encoder
# ---------------------------------------------------------------------------

class SimCLREncoder(nn.Module):
    """
    ResNet-50 backbone with SimCLR projection head.

    Why ResNet-50 over EfficientNet?
      ResNet-50's residual connections are better suited for contrastive
      learning — the skip connections preserve low-level spatial features
      which are important for solar disk morphology. EfficientNet's
      aggressive channel scaling can collapse representations during
      the large negative-pair gradients of NT-Xent.

    After training, the projection head (g) is DISCARDED.
    Only the backbone (f) is kept and plugged into model_lstm_v2.py
    as a drop-in replacement for CNNFeatureExtractor.
    """

    def __init__(
        self,
        base_model:     str = "resnet50",
        out_dim:        int = 128,       # Projection head output dim
        freeze_layers:  int = 0,         # Freeze first N ResNet layer groups
    ):
        super().__init__()

        # Backbone
        if base_model == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feat_dim = 2048
        elif base_model == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feat_dim = 512
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        # Remove the final FC layer — we want the pooled features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Optionally freeze early layers
        if freeze_layers > 0:
            layer_groups = list(self.backbone.children())
            for layer in layer_groups[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        # Projection head: 2-layer MLP with BN (SimCLR v2 style)
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),  # No bias in final BN
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h: (B, feat_dim) — backbone features (used for downstream tasks)
            z: (B, out_dim)  — projected + L2-normalised (used for NT-Xent loss)
        """
        h = self.backbone(x).flatten(1)  # (B, feat_dim)
        z = F.normalize(self.projector(h), dim=1)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time: returns backbone features only (no projection)."""
        with torch.no_grad():
            h = self.backbone(x).flatten(1)
        return h


# ---------------------------------------------------------------------------
# NT-Xent Loss (Normalised Temperature-Scaled Cross Entropy)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    NT-Xent loss from SimCLR (Chen et al. 2020).

    For a batch of N events (2N views):
      - 2 views of the same event = 1 positive pair
      - All other 2N-2 pairs = negative pairs

    Temperature τ controls the sharpness of the distribution:
      Low τ (0.1): focuses on hard negatives, faster collapse risk
      High τ (0.5): smoother, safer for small batches
      τ=0.07 was used in the original SimCLR paper (large batches)
      τ=0.2 recommended for batch_size ≤ 256

    Complexity: O(N²) memory — use gradient checkpointing for N > 512
    """

    def __init__(self, temperature: float = 0.2, device: torch.device = None):
        super().__init__()
        self.temperature = temperature
        self.device      = device or torch.device("cpu")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (N, D) L2-normalised projections of view 1
            z_j: (N, D) L2-normalised projections of view 2
        Returns:
            scalar loss
        """
        N = z_i.size(0)

        # Concatenate: [z_i; z_j] → (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix: (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, -1e9)

        # Positive pair indices
        # View 1's positive is at index N+i; View 2's positive is at index i
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N),
        ]).to(self.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class SimCLRTrainer:
    """
    End-to-end SimCLR training loop.

    Key hyperparameters from the paper:
      - Large batch size (256+) is critical for NT-Xent (more negatives = better)
      - LARS optimiser with linear warmup + cosine decay
      - Train for 200-1000 epochs (self-supervised needs longer than supervised)
      - On T4×2 Kaggle: batch=128, epochs=100 is a good starting point
    """

    def __init__(
        self,
        model:      SimCLREncoder,
        device:     torch.device,
        temperature: float = 0.2,
        lr:          float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.model  = model.to(device)
        self.device = device
        self.criterion = NTXentLoss(temperature, device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(
        self,
        loader: DataLoader,
        scaler: torch.cuda.amp.GradScaler,
        epoch:  int,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for step, (view1, view2) in enumerate(loader):
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                _, z1 = self.model(view1)
                _, z2 = self.model(view2)
                loss  = self.criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 20 == 0:
                log.info(
                    f"  [SimCLR] Epoch {epoch} | Step {step}/{len(loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        return total_loss / len(loader)

    def save(self, epoch: int, loss: float) -> str:
        path = os.path.join(CHECKPOINT_DIR, f"simclr_e{epoch}.pt")
        torch.save({
            "epoch":       epoch,
            "loss":        loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)
        # Also save backbone-only weights for downstream use
        backbone_path = os.path.join(CHECKPOINT_DIR, "simclr_backbone.pt")
        torch.save(self.model.backbone.state_dict(), backbone_path)
        log.info(f"[SimCLR] Saved checkpoint: {path}")
        return path


# ---------------------------------------------------------------------------
# Drop-in CNNFeatureExtractor replacement for model_lstm_v2.py
# ---------------------------------------------------------------------------

class SimCLRFeatureExtractor(nn.Module):
    """
    Drop-in replacement for CNNFeatureExtractor in model_lstm_v2.py.

    Load the pre-trained SimCLR backbone and use it as a frozen or
    fine-tunable feature extractor.

    Usage in model_lstm_v2.py:
        # Replace:  self.cnn = CNNFeatureExtractor(freeze_backbone)
        # With:     self.cnn = SimCLRFeatureExtractor.from_checkpoint(path)
    """

    def __init__(self, backbone: nn.Module, feat_dim: int = 2048, freeze: bool = True):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = None,
        base_model:  str = "resnet50",
        freeze:      bool = True,
    ) -> "SimCLRFeatureExtractor":
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "simclr_backbone.pt")

        encoder = SimCLREncoder(base_model=base_model)
        if os.path.exists(checkpoint_path):
            encoder.backbone.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")
            )
            log.info(f"[SimCLR] Loaded backbone from {checkpoint_path}")
        else:
            log.warning(f"[SimCLR] No checkpoint at {checkpoint_path} — using ImageNet init")

        feat_dim = encoder.feat_dim
        return cls(encoder.backbone, feat_dim, freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, feat_dim) features — same API as CNNFeatureExtractor."""
        h = self.backbone(x).flatten(1)
        return h


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Train SimCLR on SUIT multi-band solar images")
    p.add_argument("--catalog",       default=CATALOG_PATH)
    p.add_argument("--image_dir",     default=IMAGE_DIR)
    p.add_argument("--output_dir",    default=CHECKPOINT_DIR)
    p.add_argument("--base_model",    default="resnet50", choices=["resnet50","resnet18"])
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--temperature",   type=float, default=0.2)
    p.add_argument("--interband_prob",type=float, default=0.5)
    p.add_argument("--workers",       type=int,   default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    dataset = MultiBandPairDataset(
        catalog_path    = args.catalog,
        image_dir       = args.image_dir,
        interband_prob  = args.interband_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.workers,
        pin_memory  = True,
        drop_last   = True,   # NT-Xent needs uniform batch size
    )

    model   = SimCLREncoder(base_model=args.base_model, out_dim=128)
    trainer = SimCLRTrainer(model, device, args.temperature, args.lr)
    scaler  = torch.amp.GradScaler("cuda")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(loader, scaler, epoch)
        log.info(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            trainer.save(epoch, loss)
            log.info(f"  ★ New best: {best_loss:.4f}")

    log.info("SimCLR training complete.")
    log.info(f"Backbone saved to: {os.path.join(CHECKPOINT_DIR, 'simclr_backbone.pt')}")
    log.info("Use SimCLRFeatureExtractor.from_checkpoint() in model_lstm_v2.py")


if __name__ == "__main__":
    main()
