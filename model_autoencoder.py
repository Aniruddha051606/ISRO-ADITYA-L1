"""
model_autoencoder.py
====================
Convolutional Autoencoder for Solar Flare Anomaly Detection
Mission: ISRO Aditya-L1

Architecture Philosophy:
  Train exclusively on "Quiet Sun" (Class 0) images. The encoder learns
  the latent manifold of normal solar activity. When a flare event is fed
  at inference time, the decoder cannot reconstruct the unfamiliar signal
  accurately → high pixel-wise MSE → anomaly alert.

Architecture:
  Input (3×224×224)
      │
  ┌───▼──────────────────────────────────────┐
  │  ENCODER                                  │
  │  Conv2d → BN → LeakyReLU  ×5 (strided)   │
  │  64 → 128 → 256 → 512 → 1024 channels    │
  └───────────────────────┬──────────────────┘
                          │  Bottleneck
                   (B, 1024, 7, 7)
  ┌───────────────────────▼──────────────────┐
  │  DECODER                                  │
  │  ConvTranspose2d → BN → ReLU  ×5         │
  │  1024 → 512 → 256 → 128 → 64 → 3        │
  └───────────────────────┬──────────────────┘
                          │
                  Output (3×224×224) — Sigmoid activated
                  MSE(Input, Output) = Anomaly Score
"""

import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Encoder Building Block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Strided convolution block (replaces MaxPool for learnable downsampling).

    stride=2 halves the spatial dimensions at each stage.
    LeakyReLU is preferred in encoders to avoid dead neurons.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 4,
        stride:       int = 2,
        padding:      int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Decoder Building Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Transposed convolution block for spatial upsampling.

    ReLU is standard in decoders; the final layer uses Sigmoid.
    """

    def __init__(
        self,
        in_channels:   int,
        out_channels:  int,
        kernel_size:   int = 4,
        stride:        int = 2,
        padding:       int = 1,
        output_padding:int = 0,
        final_layer:   bool = False,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding,
                bias=False,
            )
        ]
        if final_layer:
            layers.append(nn.Sigmoid())   # Output pixel values ∈ [0, 1]
        else:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Convolutional Autoencoder
# ---------------------------------------------------------------------------

class SolarAutoencoder(nn.Module):
    """
    Full Convolutional Autoencoder for Quiet Sun reconstruction.

    Spatial progression (224 → 112 → 56 → 28 → 14 → 7):
      Encoder stages × 5, each halving spatial resolution.
      Decoder stages × 5, each doubling back to 224.

    Bottleneck:
      (B, 1024, 7, 7) = 50,176-dimensional latent space.
      You can optionally add a 1D Linear bottleneck here for even more
      compression if memory is a concern.

    Usage at inference:
        model.eval()
        with torch.no_grad():
            recon = model(img_batch)
        mse_score = F.mse_loss(recon, img_batch, reduction='none')
                              .mean(dim=[1,2,3])  # per-image score
    """

    def __init__(
        self,
        in_channels:      int = 3,
        base_channels:    int = 64,
        latent_channels:  int = 1024,
    ):
        super().__init__()

        C  = base_channels     # 64
        LC = latent_channels   # 1024

        # ---- Encoder ----
        # Input:  (B,  3, 224, 224)
        # After 5 strided-2 convolutions: (B, LC, 7, 7)
        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, C),       # (B,  64, 112, 112)
            EncoderBlock(C,   C * 2),           # (B, 128,  56,  56)
            EncoderBlock(C*2, C * 4),           # (B, 256,  28,  28)
            EncoderBlock(C*4, C * 8),           # (B, 512,  14,  14)
            EncoderBlock(C*8, LC),              # (B,1024,   7,   7)
        )

        # ---- Decoder ----
        # Mirrors the encoder: (B, LC, 7, 7) → (B, 3, 224, 224)
        self.decoder = nn.Sequential(
            DecoderBlock(LC,    C * 8),         # (B, 512,  14,  14)
            DecoderBlock(C*8,   C * 4),         # (B, 256,  28,  28)
            DecoderBlock(C*4,   C * 2),         # (B, 128,  56,  56)
            DecoderBlock(C*2,   C),             # (B,  64, 112, 112)
            DecoderBlock(C, in_channels,        # (B,   3, 224, 224)
                         final_layer=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for conv layers, constant 1 for BN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the bottleneck latent representation.
        Useful for downstream t-SNE / UMAP visualisation.
        """
        return self.encoder(x)   # (B, 1024, 7, 7)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  normalised input image in [0, 1]
        Returns:
            recon: (B, C, H, W)  reconstructed image in [0, 1]
        """
        z     = self.encode(x)
        recon = self.decode(z)
        return recon


# ---------------------------------------------------------------------------
# Per-Image Anomaly Scorer
# ---------------------------------------------------------------------------

class AnomalyScorer:
    """
    Thin wrapper for computing MSE-based anomaly scores at inference time.

    Usage:
        scorer = AnomalyScorer(model, device, threshold=0.02)
        score, is_flare = scorer.score(image_tensor)
    """

    def __init__(
        self,
        model:     SolarAutoencoder,
        device:    torch.device,
        threshold: float = 0.02,   # Tune this on a held-out calibration set
    ):
        self.model     = model.to(device).eval()
        self.device    = device
        self.threshold = threshold

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) normalised image batch on CPU
        Returns:
            mse_scores : (B,) per-image MSE reconstruction error
            predictions: (B,) binary tensor — 1 = Flare anomaly
        """
        import torch.nn.functional as F
        x     = x.to(self.device)
        recon = self.model(x)
        # Per-image mean squared error across all pixels and channels
        mse_scores  = F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])
        predictions = (mse_scores > self.threshold).long()
        return mse_scores.cpu(), predictions.cpu()

    def set_threshold(self, calibration_mse: torch.Tensor, percentile: float = 95.0) -> None:
        """
        Automatically set threshold from the MSE distribution of quiet-sun
        calibration images (e.g. 95th percentile).

        Args:
            calibration_mse: 1-D tensor of MSE scores from Class-0 images
            percentile: Upper tail percentile to use as decision boundary
        """
        self.threshold = float(torch.quantile(calibration_mse, percentile / 100.0))
        print(f"[AnomalyScorer] Threshold set to {self.threshold:.6f} "
              f"({percentile}th percentile of quiet-sun MSE distribution)")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_autoencoder.py] Running sanity check on: {device}")

    model = SolarAutoencoder(in_channels=3, base_channels=64, latent_channels=1024).to(device)

    # Dummy batch of quiet-sun images
    dummy_imgs = torch.rand(4, 3, 224, 224).to(device)

    with torch.no_grad():
        recon = model(dummy_imgs)

    print(f"  Input shape  : {dummy_imgs.shape}")
    print(f"  Output shape : {recon.shape}")
    assert recon.shape == dummy_imgs.shape, "Shape mismatch!"
    assert recon.min() >= 0.0 and recon.max() <= 1.0, "Sigmoid output out of range!"

    # Bottleneck shape
    latent = model.encode(dummy_imgs)
    print(f"  Bottleneck   : {latent.shape}")   # (4, 1024, 7, 7)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print("[model_autoencoder.py] Sanity check PASSED ✓")
