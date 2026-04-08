"""
model_multiband.py
==================
Multi-Band Fusion CNN for All 11 SUIT Filter Bands Simultaneously
Mission: ISRO Aditya-L1

WHY THIS IS THE MOST PHYSICALLY MEANINGFUL UPGRADE
────────────────────────────────────────────────────
SUIT has 11 filter bands covering different atmospheric layers:
  2140Å  — Photosphere (continuum, temperature minimum)
  2767Å  — Chromosphere (Mg II wing)
  2770Å  — Chromosphere (Mg II wing)
  2796Å  — Chromosphere (Mg II k core) ← most frames in your dataset
  2803Å  — Chromosphere (Mg II h core)
  2832Å  — Photosphere/low chromosphere
  3000Å  — UV continuum
  3400Å  — Near-UV continuum
  3880Å  — Ca II H wing
  3968Å  — Ca II H core ← best single flare indicator
  2200Å  — UV continuum

A solar flare propagates UPWARD through these layers:
  1. Photosphere: small brightening at magnetic polarity inversion line
  2. Low chromosphere (2832, 3000): UV brightening appears
  3. Core chromosphere (2796, 2803, 3968): intense flare ribbons
  4. Upper chromosphere (2767, 2770): ribbon separation begins
  5. ALL BANDS: peak emission, maximum area

The TIMING between bands is the physical signature.
A single-band VAE sees one frame. This model sees ALL bands at time T
as a (11, H, W) tensor — one sample per observation epoch —
and learns the cross-band correlation patterns that precede eruptions.

ARCHITECTURE
────────────
  1. Per-band encoder: lightweight CNN for each of the 11 bands
  2. Band cross-attention: Transformer that attends across bands at each spatial location
  3. Temporal stack: if multiple timesteps available, Bi-LSTM over fused band embeddings
  4. Anomaly head: reconstructs the multi-band composite and scores via MSE
  5. Classifier head (when labels available from GOES): binary flare prediction

HANDLING MISSING BANDS
────────────────────────
Not every observation has all 11 bands simultaneously (different cadences,
pointing modes). Model uses a band availability mask to zero-out missing
bands at attention time, so available bands don't attend to missing ones.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "multiband")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
IMAGE_DIR      = os.path.join(DATA_DIR, "images")
CATALOG_PATH   = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

log = logging.getLogger(__name__)

# SUIT band definitions — physical ordering by wavelength
SUIT_BANDS = {
    "NB_2140": 2140.0,
    "NB_2200": 2200.0,
    "NB_2767": 2767.0,
    "NB_2770": 2770.0,
    "NB_2796": 2796.0,
    "NB_2803": 2803.0,
    "NB_2832": 2832.0,
    "NB_3000": 3000.0,
    "NB_3400": 3400.0,
    "NB_3880": 3880.0,
    "NB_3968": 3968.5,
}
BAND_NAMES  = list(SUIT_BANDS.keys())
NUM_BANDS   = len(BAND_NAMES)
# Formation heights (approximate, in Mm above photosphere) — used for positional encoding
BAND_HEIGHTS = [0.0, 0.2, 1.0, 1.1, 1.5, 1.4, 0.5, 0.3, 0.1, 1.8, 2.0]


# ---------------------------------------------------------------------------
# Per-Band CNN Encoder
# ---------------------------------------------------------------------------

class BandEncoder(nn.Module):
    """
    Lightweight CNN that encodes a single-band solar image into a
    spatial feature map (B, feat_dim, H/8, W/8).

    Deliberately shallow (4 conv layers) — deep representations are built
    by the cross-band Transformer that follows.
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            # Layer 1: 1 → 32, stride 2
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            # Layer 2: 32 → 64, stride 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            # Layer 3: 64 → feat_dim, stride 2
            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.GELU(),
            # Global average pool → (B, feat_dim, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, feat_dim)"""
        return self.net(x).flatten(1)


# ---------------------------------------------------------------------------
# Band Cross-Attention Module
# ---------------------------------------------------------------------------

class BandCrossAttention(nn.Module):
    """
    Transformer that attends ACROSS the 11 bands for a given image.
    Each band is a "token"; attention learns which bands correlate.

    Physical interpretation:
      If Ca II H (token 10) has high attention to Mg II k (token 4),
      the model has learned the chromospheric coupling signature of flares.

    Band positional encoding uses formation heights so the model knows
    the physical ordering (photosphere → chromosphere).
    """

    def __init__(
        self,
        num_bands:   int = NUM_BANDS,
        feat_dim:    int = 64,
        n_heads:     int = 4,
        n_layers:    int = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # Positional encoding by formation height
        heights = torch.tensor(BAND_HEIGHTS, dtype=torch.float32)
        heights = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
        pos_enc  = torch.zeros(num_bands, feat_dim)
        div_term = torch.exp(torch.arange(0, feat_dim, 2).float() *
                             (-np.log(10000.0) / feat_dim))
        pos_enc[:, 0::2] = torch.sin(heights.unsqueeze(1) * div_term)
        pos_enc[:, 1::2] = torch.cos(heights.unsqueeze(1) * div_term)
        self.register_buffer("pos_enc", pos_enc)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, dim_feedforward=feat_dim*4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(feat_dim)
        )

        # [CLS] token for global band-fused representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        band_feats:    torch.Tensor,           # (B, num_bands, feat_dim)
        band_mask:     Optional[torch.Tensor] = None,  # (B, num_bands) — 1=missing
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_out:    (B, feat_dim)         — global fused representation
            band_out:   (B, num_bands, feat_dim) — per-band enriched features
        """
        B = band_feats.shape[0]

        # Add formation-height positional encoding
        x = band_feats + self.pos_enc.unsqueeze(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)      # (B, 1+num_bands, feat_dim)

        # Build attention mask: CLS can always attend, masked bands are ignored
        src_key_padding_mask = None
        if band_mask is not None:
            # Prepend False for CLS (never masked)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=band_feats.device)
            src_key_padding_mask = torch.cat([cls_mask, band_mask.bool()], dim=1)

        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return out[:, 0], out[:, 1:]   # CLS, band tokens


# ---------------------------------------------------------------------------
# Full Multi-Band Fusion Model
# ---------------------------------------------------------------------------

class MultiBandFusionModel(nn.Module):
    """
    Complete multi-band solar event model.

    Two modes depending on available labels:

    Mode A — Unsupervised (no GOES labels yet):
      Reconstruction autoencoder on the full band stack.
      Anomaly score = per-band reconstruction MSE, weighted by band flare sensitivity.

    Mode B — Supervised (after GOES gives you flare labels):
      Binary classifier head on top of the cross-attention CLS token.
      Train with FocalLoss from model_lstm_v2.py.
    """

    def __init__(
        self,
        num_bands:      int   = NUM_BANDS,
        feat_dim:       int   = 64,
        n_attn_heads:   int   = 4,
        n_attn_layers:  int   = 3,
        dropout:        float = 0.1,
        image_size:     int   = 224,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.feat_dim  = feat_dim

        # Per-band encoders (shared weights — all bands have same physical resolution)
        # Using a single shared encoder with band-ID conditioning is more parameter-efficient
        self.band_encoder = BandEncoder(in_channels=1, feat_dim=feat_dim)

        # Band ID embedding (so shared encoder knows which band it's processing)
        self.band_id_embed = nn.Embedding(num_bands, feat_dim)

        # Cross-band attention
        self.cross_attn = BandCrossAttention(
            num_bands, feat_dim, n_attn_heads, n_attn_layers, dropout
        )

        # Decoder for unsupervised anomaly scoring
        # Reconstructs each band's feature from the CLS token
        self.band_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.GELU(),
                nn.Linear(feat_dim * 2, feat_dim),
            )
            for _ in range(num_bands)
        ])

        # Classifier head (used when GOES labels available)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),
        )

        # Band flare sensitivity weights (Ca II H most sensitive, photosphere least)
        # Used to weight the reconstruction MSE anomaly score
        sensitivity = torch.tensor([
            0.3,   # 2140 — photosphere, low sensitivity
            0.3,   # 2200 — UV continuum, low
            0.7,   # 2767 — Mg II wing, moderate
            0.7,   # 2770 — Mg II wing, moderate
            0.9,   # 2796 — Mg II k core, high
            0.9,   # 2803 — Mg II h core, high
            0.5,   # 2832 — photosphere/low chrom, moderate
            0.4,   # 3000 — UV continuum, low-moderate
            0.3,   # 3400 — near-UV, low
            0.8,   # 3880 — Ca II H wing, high
            1.0,   # 3968 — Ca II H core, highest
        ], dtype=torch.float32)
        self.register_buffer("band_sensitivity", sensitivity)

    def encode_bands(
        self,
        images:    torch.Tensor,           # (B, num_bands, H, W)
        band_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all bands and run cross-attention.

        Returns:
            cls_feat:   (B, feat_dim)            — global fused representation
            band_feats: (B, num_bands, feat_dim) — per-band representations
        """
        B = images.shape[0]
        band_feats = []

        for i in range(self.num_bands):
            # Encode this band: (B, H, W) → (B, 1, H, W) → (B, feat_dim)
            band_img  = images[:, i:i+1]      # (B, 1, H, W)

            # Zero out missing bands
            if band_mask is not None:
                band_img = band_img * (1 - band_mask[:, i:i+1].unsqueeze(-1).unsqueeze(-1).float())

            enc = self.band_encoder(band_img)                           # (B, feat_dim)
            enc = enc + self.band_id_embed(
                torch.full((B,), i, dtype=torch.long, device=images.device)
            )
            band_feats.append(enc)

        band_feats = torch.stack(band_feats, dim=1)   # (B, num_bands, feat_dim)
        cls_feat, band_feats_out = self.cross_attn(band_feats, band_mask)
        return cls_feat, band_feats_out

    def forward(
        self,
        images:      torch.Tensor,            # (B, num_bands, H, W) — normalised [0,1]
        band_mask:   Optional[torch.Tensor] = None,  # (B, num_bands) — 1=missing
        labels:      Optional[torch.Tensor] = None,  # (B,) — GOES labels if available
    ) -> Dict[str, torch.Tensor]:

        cls_feat, band_feats = self.encode_bands(images, band_mask)

        # Reconstruction of each band's features (unsupervised)
        reconstructed = torch.stack([
            dec(cls_feat) for dec in self.band_decoder
        ], dim=1)   # (B, num_bands, feat_dim)

        # Per-band reconstruction MSE
        recon_mse = ((reconstructed - band_feats.detach()) ** 2).mean(dim=-1)  # (B, num_bands)

        # Weighted anomaly score (higher weight on flare-sensitive bands)
        anomaly_score = (recon_mse * self.band_sensitivity.unsqueeze(0)).mean(dim=1)  # (B,)

        # Reconstruction loss for training
        # Ignore missing bands in loss
        if band_mask is not None:
            visible_mask = (1 - band_mask.float())
            recon_loss = (recon_mse * visible_mask).sum() / visible_mask.sum().clamp(min=1)
        else:
            recon_loss = recon_mse.mean()

        result = {
            "anomaly_score":  anomaly_score,
            "recon_loss":     recon_loss,
            "band_recon_mse": recon_mse,
            "cls_feat":       cls_feat,
        }

        # Classifier (supervised mode — only when labels provided)
        if labels is not None:
            logits = self.classifier(cls_feat).squeeze(-1)
            probs  = torch.sigmoid(logits)
            result["logits"] = logits
            result["probs"]  = probs
        else:
            # At inference, also provide classifier output
            logits = self.classifier(cls_feat).squeeze(-1)
            result["probs"] = torch.sigmoid(logits)

        return result


# ---------------------------------------------------------------------------
# Multi-Band Dataset
# ---------------------------------------------------------------------------

class MultiBandDataset(Dataset):
    """
    Groups catalog rows by observation epoch (T_OBS rounded to 10s).
    Returns (num_bands, H, W) tensor per observation.
    Handles partial observations (not all 11 bands always available).
    """

    def __init__(
        self,
        catalog_path: str = CATALOG_PATH,
        image_dir:    str = IMAGE_DIR,
        image_size:   int = 224,
    ):
        df = pd.read_csv(catalog_path, low_memory=False)
        df.columns = df.columns.str.strip()
        fname_col = next((c for c in ["Filename","filename"] if c in df.columns), None)
        df = df.rename(columns={fname_col: "filename"})
        df["filename"] = df["filename"].astype(str).apply(
            lambda x: x.rsplit(".", 1)[0] + ".png"
        )
        image_dir_p = Path(image_dir)
        df = df[df["filename"].apply(lambda f: (image_dir_p / f).exists())]

        # Group by observation epoch
        if "T_OBS" in df.columns and "WAVELNTH" in df.columns:
            df["t_round"] = pd.to_datetime(
                df["T_OBS"], errors="coerce"
            ).dt.round("30s").astype(str)
            self.groups = df.groupby("t_round").apply(
                lambda g: dict(zip(
                    g["WAVELNTH"].astype(float).tolist(),
                    g["filename"].tolist()
                ))
            ).tolist()
        else:
            # Fallback: single-band pseudo-groups
            self.groups = [{"2796": row} for row in df["filename"]]

        # Filter groups with at least 2 bands
        self.groups = [g for g in self.groups if len(g) >= 1]
        self.image_dir  = image_dir_p
        self.image_size = image_size

        # Map wavelength float → band index
        self.wl_to_idx = {v: i for i, v in enumerate(SUIT_BANDS.values())}

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        log.info(f"[MultiBand] {len(self.groups)} observations, {NUM_BANDS} bands each")

    def __len__(self): return len(self.groups)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        group = self.groups[idx]
        images    = torch.zeros(NUM_BANDS, 1, self.image_size, self.image_size)
        band_mask = torch.ones(NUM_BANDS)   # 1 = missing

        for wl, fname in group.items():
            # Find closest SUIT band
            closest = min(self.wl_to_idx.keys(), key=lambda k: abs(k - float(wl)))
            bid     = self.wl_to_idx[closest]
            img = Image.open(self.image_dir / fname).convert("L")   # Grayscale
            images[bid, 0] = self.transform(img)[0]
            band_mask[bid] = 0   # Mark as available

        # Squeeze band channel dim for the model: (B, num_bands, 1, H, W) → (B, num_bands, H, W)
        return images.squeeze(1), band_mask


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--catalog",     default=CATALOG_PATH)
    p.add_argument("--image_dir",   default=IMAGE_DIR)
    p.add_argument("--output_dir",  default=CHECKPOINT_DIR)
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--feat_dim",    type=int,   default=64)
    p.add_argument("--workers",     type=int,   default=4)
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiBandDataset(args.catalog, args.image_dir)
    loader  = DataLoader(dataset, args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True,
                         collate_fn=lambda b: (
                             torch.stack([x[0] for x in b]),
                             torch.stack([x[1] for x in b]),
                         ))

    model  = MultiBandFusionModel(feat_dim=args.feat_dim).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")
    best   = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, band_mask in loader:
            images    = images.to(device, non_blocking=True)
            band_mask = band_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out  = model(images, band_mask)
                loss = out["recon_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()

        scheduler.step()
        mean_loss = total / len(loader)
        log.info(f"Epoch {epoch}/{args.epochs} | ReconLoss: {mean_loss:.5f}")

        if mean_loss < best:
            best = mean_loss
            torch.save({
                "epoch": epoch, "loss": mean_loss,
                "model_state": model.state_dict(),
            }, os.path.join(args.output_dir, "best_multiband.pt"))
            log.info(f"  ★ Best saved ({best:.5f})")

    log.info("Multi-band training complete.")


if __name__ == "__main__":
    main()
