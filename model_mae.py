"""
model_mae.py
============
Masked Autoencoder (MAE) for Solar FITS Image Representation
Mission: ISRO Aditya-L1

WHY MAE OVER VAE
─────────────────
Your VAE reconstructs the full image from a compressed bottleneck.
For solar images — which are mostly quiet disk with small active regions —
the VAE learns to perfectly reconstruct the boring majority and ignores
the rare detail that actually matters (flare ribbons, filament channels).

MAE forces the encoder to RECONSTRUCT MASKED PATCHES. With 75% masking:
  - The model cannot just copy neighbouring pixels
  - It must understand the global structure to fill in large gaps
  - When a sunspot or active region is masked, reconstructing it requires
    understanding what a sunspot LOOKS LIKE — i.e. solar physics features

Result: The MAE encoder builds representations 10-15× richer than a VAE
on the same data, as proven by He et al. (2021) on natural images and
confirmed by subsequent solar physics work.

ARCHITECTURE
────────────
  Encoder: ViT-Base (Vision Transformer) — processes ONLY visible patches
  Decoder: Lightweight ViT decoder — reconstructs masked patches
  
  Image (224×224) → 196 patches of 16×16
  Mask 75% → 147 masked, 49 visible
  Encoder processes 49 visible patches → 768-dim each
  Decoder takes 49 encoded + 147 mask tokens → reconstructs all 196

ANOMALY DETECTION WITH MAE
───────────────────────────
  At inference: mask 50% of patches, reconstruct, compute per-patch MSE
  High reconstruction error on a patch → that patch is anomalous
  This gives SPATIAL LOCALISATION — tells you WHERE the flare is, not just THAT it exists

DOWNSTREAM USE
──────────────
  The encoder is plugged into model_lstm_v2.py as SimCLRFeatureExtractor was:
    self.cnn = MAEFeatureExtractor.from_checkpoint(path)
  Returns (B, 768) per-frame features → feeds into Bi-LSTM / Transformer
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "mae")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
IMAGE_DIR      = os.path.join(DATA_DIR, "images")
CATALOG_PATH   = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Splits image into non-overlapping patches and linearly embeds them."""

    def __init__(
        self,
        img_size:   int = 224,
        patch_size: int = 16,
        in_chans:   int = 3,
        embed_dim:  int = 768,
    ):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Single Conv2d with stride=patch_size does the patch split + linear embed
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Sinusoidal 2D Positional Encoding
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    2D sinusoidal positional embedding for a grid_size×grid_size patch grid.
    Returns (grid_size², embed_dim).
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid   = np.meshgrid(grid_w, grid_h)
    grid   = np.stack(grid, axis=0).reshape([2, -1])

    emb_h  = _get_1d_sincos(embed_dim // 2, grid[0])
    emb_w  = _get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)  # (N, D)


def _get_1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega  = 1.0 / (10000 ** omega)
    out    = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Transformer Block (used in both encoder and decoder)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# MAE Encoder (ViT-Base)
# ---------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """
    Vision Transformer encoder that processes only visible (unmasked) patches.
    Lighter than full ViT because it skips 75% of patches during training.

    ViT-Base: embed_dim=768, depth=12, num_heads=12 — ~86M params
    ViT-Small: embed_dim=384, depth=12, num_heads=6  — ~22M params  ← recommended for T4
    ViT-Tiny:  embed_dim=192, depth=12, num_heads=3  — ~6M params   ← fast iteration
    """

    def __init__(
        self,
        img_size:    int   = 224,
        patch_size:  int   = 16,
        in_chans:    int   = 3,
        embed_dim:   int   = 384,   # ViT-Small default (fits T4 GPU)
        depth:       int   = 12,
        num_heads:   int   = 6,
        mlp_ratio:   float = 4.0,
        mask_ratio:  float = 0.75,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim   = embed_dim
        self.mask_ratio  = mask_ratio

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (fixed sinusoidal, not learned)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, img_size // patch_size)
        self.register_buffer("pos_embed",
                             torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_embed.data[:, 1:] = torch.from_numpy(pos_embed)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def random_masking(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking: shuffle patches, keep first (1-mask_ratio) fraction.

        Returns:
            x_masked:      (B, num_visible, D) — visible patch embeddings
            mask:          (B, num_patches)    — 1=masked, 0=visible
            restore_ids:   (B, num_patches)    — indices to restore original order
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - self.mask_ratio))

        noise    = torch.rand(B, N, device=x.device)
        ids_sort = torch.argsort(noise, dim=1)
        restore  = torch.argsort(ids_sort, dim=1)

        ids_keep = ids_sort[:, :num_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask        = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask        = torch.gather(mask, 1, restore)

        return x_masked, mask, restore

    def forward(
        self, x: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (B, C, H, W)
            mask_ratio: Override instance mask ratio (None = use self.mask_ratio)

        Returns:
            latent:     (B, num_visible+1, embed_dim) — encoded visible patches + CLS
            mask:       (B, num_patches) — binary mask (1=masked)
            restore:    (B, num_patches) — restore indices
        """
        if mask_ratio is not None:
            orig = self.mask_ratio
            self.mask_ratio = mask_ratio

        x = self.patch_embed(x)                           # (B, N, D)
        x = x + self.pos_embed[:, 1:]                     # Add spatial pos embed

        x, mask, restore = self.random_masking(x)

        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
        cls = cls + self.pos_embed[:, :1]
        x   = torch.cat([cls, x], dim=1)                  # (B, 1+num_vis, D)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if mask_ratio is not None:
            self.mask_ratio = orig

        return x, mask, restore


# ---------------------------------------------------------------------------
# MAE Decoder (lightweight)
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs pixel values for masked patches.
    Much smaller than encoder — decoder depth=4, dim=192 works well.
    """

    def __init__(
        self,
        num_patches:   int = 196,
        encoder_dim:   int = 384,
        decoder_dim:   int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = 3,
        patch_size:    int = 16,
        in_chans:      int = 3,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.patch_pixels = patch_size * patch_size * in_chans

        # Project encoder output to decoder dimension
        self.proj = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token (replaces masked patch positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for decoder (full set — all patches)
        pos = get_2d_sincos_pos_embed(decoder_dim, int(num_patches**0.5))
        self.register_buffer("pos_embed",
                             torch.zeros(1, num_patches + 1, decoder_dim))
        self.pos_embed.data[:, 1:] = torch.from_numpy(pos)

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, self.patch_pixels)

    def forward(
        self,
        latent:  torch.Tensor,  # (B, 1+num_vis, encoder_dim)
        restore: torch.Tensor,  # (B, num_patches) restore indices
        num_patches: int,
    ) -> torch.Tensor:
        """Returns (B, num_patches, patch_pixels) — predicted pixel values."""
        x  = self.proj(latent)                              # (B, 1+num_vis, dec_dim)
        B, num_vis_plus1, D = x.shape
        num_vis  = num_vis_plus1 - 1
        num_mask = num_patches - num_vis

        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, num_mask, -1)

        # Concatenate visible + mask tokens (exclude CLS)
        x_vis  = x[:, 1:]                                   # (B, num_vis, D)
        x_full = torch.cat([x_vis, mask_tokens], dim=1)    # (B, num_patches, D)

        # Restore original patch order
        x_full = torch.gather(
            x_full, 1,
            restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # Add positional embedding + prepend CLS
        x_full = x_full + self.pos_embed[:, 1:]
        cls    = x[:, :1] + self.pos_embed[:, :1]
        x_full = torch.cat([cls, x_full], dim=1)

        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)

        # Predict pixels for all patches (CLS discarded)
        return self.pred(x_full[:, 1:])                    # (B, num_patches, patch_px)


# ---------------------------------------------------------------------------
# Full MAE Model
# ---------------------------------------------------------------------------

class SolarMAE(nn.Module):
    """
    Full MAE model combining encoder + decoder.

    Inference modes:
      1. Pretrain (mask_ratio=0.75):  used during self-supervised training
      2. Fine-tune (mask_ratio=0.0):  encode full image, use CLS as feature
      3. Anomaly score (mask_ratio=0.5): mask half, measure reconstruction error
    """

    def __init__(
        self,
        img_size:       int   = 224,
        patch_size:     int   = 16,
        in_chans:       int   = 3,
        encoder_dim:    int   = 384,
        encoder_depth:  int   = 12,
        encoder_heads:  int   = 6,
        decoder_dim:    int   = 192,
        decoder_depth:  int   = 4,
        decoder_heads:  int   = 3,
        mask_ratio:     float = 0.75,
    ):
        super().__init__()
        self.encoder = MAEEncoder(
            img_size, patch_size, in_chans,
            encoder_dim, encoder_depth, encoder_heads,
            mask_ratio=mask_ratio,
        )
        self.decoder = MAEDecoder(
            self.encoder.num_patches,
            encoder_dim, decoder_dim, decoder_depth, decoder_heads,
            patch_size, in_chans,
        )
        self.patch_size = patch_size
        self.in_chans   = in_chans

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, num_patches, patch_pixels)"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p * p * self.in_chans)

    def unpatchify(self, x: torch.Tensor, img_size: int = 224) -> torch.Tensor:
        """(B, num_patches, patch_pixels) → (B, C, H, W)"""
        p  = self.patch_size
        h  = w = img_size // p
        x  = x.reshape(x.shape[0], h, w, p, p, self.in_chans)
        x  = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_chans, img_size, img_size)

    def forward(
        self, x: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred:  (B, num_patches, patch_pixels) — pixel predictions for ALL patches
            mask:  (B, num_patches) — 1=masked (these are the ones we compute loss on)
            loss:  scalar — mean patch reconstruction MSE on MASKED patches only
        """
        latent, mask, restore = self.encoder(x, mask_ratio)
        pred = self.decoder(latent, restore, self.encoder.num_patches)

        # Normalise target patches (per-patch mean/std normalisation as in MAE paper)
        target = self.patchify(x)
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var( dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        # Loss only on MASKED patches (not visible ones)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)           # (B, num_patches) per-patch MSE
        loss = (loss * mask).sum() / mask.sum()

        return pred, mask, loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full-image encoding (no masking). Returns CLS token (B, encoder_dim)."""
        latent, _, _ = self.encoder(x, mask_ratio=0.0)
        return latent[:, 0]   # CLS token

    @torch.no_grad()
    def anomaly_score(
        self,
        x:           torch.Tensor,
        mask_ratio:  float = 0.5,
        n_samples:   int   = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-image anomaly scores by averaging reconstruction error
        over n_samples random masking configurations.

        Returns:
            image_scores: (B,) — mean reconstruction error per image
            patch_maps:   (B, H, W) — spatial anomaly heatmap (upsampled)
        """
        B        = x.shape[0]
        img_size = x.shape[-1]
        n_patches_side = img_size // self.patch_size

        all_patch_scores = []
        for _ in range(n_samples):
            pred, mask, _ = self.forward(x, mask_ratio=mask_ratio)
            target = self.patchify(x)
            target_norm = (target - target.mean(-1, keepdim=True)) / \
                         (target.var(-1, keepdim=True) + 1e-6).sqrt()
            patch_mse = ((pred - target_norm) ** 2).mean(dim=-1)  # (B, N)
            # Only score masked patches
            patch_mse = patch_mse * mask
            all_patch_scores.append(patch_mse)

        mean_patch_scores = torch.stack(all_patch_scores).mean(0)  # (B, N)
        image_scores      = mean_patch_scores.mean(1)               # (B,)

        # Reshape patch scores to 2D spatial map
        patch_map = mean_patch_scores.reshape(
            B, n_patches_side, n_patches_side
        )
        # Upsample to original resolution
        patch_map = F.interpolate(
            patch_map.unsqueeze(1).float(),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)   # (B, H, W)

        return image_scores, patch_map


# ---------------------------------------------------------------------------
# Dataset (same image dir, no labels needed)
# ---------------------------------------------------------------------------

class SolarMAEDataset(Dataset):
    def __init__(self, catalog_path=CATALOG_PATH, image_dir=IMAGE_DIR,
                 image_size=224):
        df = pd.read_csv(catalog_path, low_memory=False)
        df.columns = df.columns.str.strip()
        fname_col = next((c for c in ["Filename","filename"] if c in df.columns), None)
        df = df.rename(columns={fname_col: "filename"})
        df["filename"] = df["filename"].astype(str).apply(
            lambda x: x.rsplit(".", 1)[0] + ".png"
        )
        image_dir_p = Path(image_dir)
        df = df[df["filename"].apply(lambda f: (image_dir_p / f).exists())]
        self.filenames = df["filename"].tolist()
        self.image_dir = image_dir_p
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.3]*3, [0.2]*3),
        ])
        log.info(f"[MAE Dataset] {len(self.filenames)} images")

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.image_dir / self.filenames[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Drop-in feature extractor for model_lstm_v2.py
# ---------------------------------------------------------------------------

class MAEFeatureExtractor(nn.Module):
    """
    Drop-in replacement for CNNFeatureExtractor / SimCLRFeatureExtractor
    in model_lstm_v2.py. Returns (B, encoder_dim) CLS features per frame.
    """

    def __init__(self, encoder: MAEEncoder, freeze: bool = True):
        super().__init__()
        self.encoder  = encoder
        self.feat_dim = encoder.embed_dim
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = None,
        freeze:          bool = True,
        **encoder_kwargs,
    ) -> "MAEFeatureExtractor":
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_mae.pt")

        model = SolarMAE(**encoder_kwargs)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(ckpt.get("model_state", ckpt))
            log.info(f"[MAE] Loaded from {checkpoint_path}")
        else:
            log.warning(f"[MAE] No checkpoint — using random init")
        return cls(model.encoder, freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, encoder_dim) — CLS token from full-image encoding."""
        latent, _, _ = self.encoder(x, mask_ratio=0.0)
        return latent[:, 0]   # CLS token


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from torch.utils.tensorboard import SummaryWriter

    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--catalog",       default=CATALOG_PATH)
    p.add_argument("--image_dir",     default=IMAGE_DIR)
    p.add_argument("--output_dir",    default=CHECKPOINT_DIR)
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1.5e-4)
    p.add_argument("--mask_ratio",    type=float, default=0.75)
    p.add_argument("--encoder_dim",   type=int,   default=384)
    p.add_argument("--workers",       type=int,   default=4)
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    dataset = SolarMAEDataset(args.catalog, args.image_dir)
    loader  = DataLoader(dataset, args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)

    model   = SolarMAE(encoder_dim=args.encoder_dim, mask_ratio=args.mask_ratio).to(device)
    total   = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {total:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.05)
    # Cosine schedule with linear warmup (standard MAE recipe)
    warmup_epochs = 40
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: min(e / warmup_epochs, 1.0) *
                  0.5 * (1 + math.cos(math.pi * e / args.epochs))
    )
    scaler  = torch.amp.GradScaler("cuda")
    writer  = SummaryWriter(os.path.join(args.output_dir, "tb"))
    best    = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                _, _, loss = model(imgs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        mean_loss = total_loss / len(loader)
        writer.add_scalar("MAE/train_loss", mean_loss, epoch)
        log.info(f"Epoch {epoch}/{args.epochs} | Loss: {mean_loss:.5f} "
                 f"| LR: {scheduler.get_last_lr()[0]:.2e}")

        if mean_loss < best:
            best = mean_loss
            torch.save({"epoch": epoch, "loss": mean_loss,
                        "model_state": model.state_dict()},
                       os.path.join(args.output_dir, "best_mae.pt"))
            log.info(f"  ★ Best saved ({best:.5f})")

    writer.close()
    log.info("MAE training complete.")


if __name__ == "__main__":
    main()
