"""
model_vae.py
============
Variational Autoencoder (VAE) for Solar Anomaly Detection
Mission: ISRO Aditya-L1 — SUIT Payload

UPGRADE OVER model_autoencoder.py
──────────────────────────────────
  Old:  Deterministic bottleneck → raw MSE score → static threshold
  New:  Probabilistic latent space (μ, σ) → ELBO loss → z-score anomaly

Why VAE beats plain AE for this mission:
  • The encoder outputs a Gaussian distribution N(μ, σ²) rather than a
    single point. Flare images produce a high σ (model is uncertain) AND
    high reconstruction error → two independent anomaly signals.
  • The KL-divergence term regularises the latent space into a unit
    Gaussian, making the anomaly score statistically interpretable:
    a 3-sigma latent deviation = genuine out-of-distribution event.
  • Rolling calibration replaces the hardcoded threshold=0.02, adapting
    to each SUIT filter band (NB3 → NB8 have different flux profiles).

Architecture:
  Input (3×224×224)
        │
  ┌─────▼──────────────────────────────────┐
  │  ENCODER  (5× strided ConvBlocks)      │
  │  3 → 64 → 128 → 256 → 512 → 1024      │
  └────────────────┬───────────────────────┘
                   │ Flatten → (B, 1024×7×7)
          ┌────────▼──────────┐
          │  Linear → μ (512) │  ← mean vector
          │  Linear → logσ²   │  ← log-variance vector
          └────────┬──────────┘
                   │  Reparameterisation: z = μ + ε·σ
  ┌────────────────▼───────────────────────┐
  │  DECODER  (5× ConvTransposeBlocks)     │
  │  512 → 1024 → 512 → 256 → 128 → 64 →3 │
  └────────────────┬───────────────────────┘
                   │
            Output (3×224×224)

Loss = MSE + λ_ssim·SSIM + β·KL(N(μ,σ²) ‖ N(0,I))

Anomaly Score = α·MSE + (1-α)·SSIM_loss + γ·KL_score
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
# Resolves all paths relative to the project root (two levels up from this file),
# so the codebase works unchanged on the Ubuntu server, Kaggle, or any local machine.

PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "vae")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(DATA_DIR,       exist_ok=True)


# ---------------------------------------------------------------------------
# Encoder / Decoder Building Blocks (identical to original for compatibility)
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k=4, s=2, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k=4, s=2, p=1,
                 out_p=0, final=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, k, s, p, out_p, bias=False)]
        if final:
            layers.append(nn.Sigmoid())
        else:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


# ---------------------------------------------------------------------------
# SSIM Loss (pulled from train_ae.py into the model module for reuse)
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """Structural Similarity loss. Penalises loss of spatial structure."""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.ws = window_size
        self.pad = window_size // 2
        kernel = self._gaussian(window_size, sigma)
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))
        self.C1, self.C2 = 0.01**2, 0.03**2

    @staticmethod
    def _gaussian(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        return g.outer(g / g.sum())

    def _ssim_channel(self, x, y):
        K = self.kernel
        mu_x = F.conv2d(x, K, padding=self.pad)
        mu_y = F.conv2d(y, K, padding=self.pad)
        sig_x  = F.conv2d(x**2, K, padding=self.pad) - mu_x**2
        sig_y  = F.conv2d(y**2, K, padding=self.pad) - mu_y**2
        sig_xy = F.conv2d(x*y,  K, padding=self.pad) - mu_x*mu_y
        num = (2*mu_x*mu_y + self.C1) * (2*sig_xy + self.C2)
        den = (mu_x**2 + mu_y**2 + self.C1) * (sig_x + sig_y + self.C2)
        return (num / den).mean()

    def forward(self, pred, target):
        C = pred.shape[1]
        ssim = torch.stack([
            self._ssim_channel(pred[:, c:c+1], target[:, c:c+1])
            for c in range(C)
        ]).mean()
        return 1.0 - ssim


# ---------------------------------------------------------------------------
# Perceptual Loss using VGG-16 feature maps
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Compares deep VGG-16 feature activations between recon and target.
    Captures high-level structural patterns (active region morphology,
    limb brightening) that MSE and SSIM both miss.
    Uses only relu2_2 and relu3_3 — lightweight enough for training loops.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        # Block 1-2: relu1_2, relu2_2
        self.slice1 = nn.Sequential(*list(vgg.features)[:9]).eval()
        # Block 3:   relu3_3
        self.slice2 = nn.Sequential(*list(vgg.features)[9:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        # Normalise to ImageNet stats before feeding VGG
        mean = torch.tensor([0.485, 0.456, 0.406],
                             device=pred.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                             device=pred.device).view(1, 3, 1, 1)
        p = (pred   - mean) / std
        t = (target - mean) / std
        p1, t1 = self.slice1(p), self.slice1(t)
        p2, t2 = self.slice2(p1), self.slice2(t1)
        return F.mse_loss(p1, t1) + F.mse_loss(p2, t2)


# ---------------------------------------------------------------------------
# VAE Encoder — outputs (μ, log σ²) instead of a single point
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        C = base_ch
        self.conv = nn.Sequential(
            EncoderBlock(in_channels, C),       # (B, 64, 112, 112)
            EncoderBlock(C,     C*2),            # (B, 128, 56, 56)
            EncoderBlock(C*2,   C*4),            # (B, 256, 28, 28)
            EncoderBlock(C*4,   C*8),            # (B, 512, 14, 14)
            EncoderBlock(C*8,   C*16),           # (B,1024,  7,  7)
        )
        self.flat_dim = C * 16 * 7 * 7          # 1024 * 49 = 50,176

        # Separate heads for μ and log σ²
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)             # (B, flat_dim)
        return self.fc_mu(h), self.fc_logvar(h) # (B, D), (B, D)


# ---------------------------------------------------------------------------
# VAE Decoder — mirrors encoder, takes sampled z back to image space
# ---------------------------------------------------------------------------

class VAEDecoder(nn.Module):
    def __init__(self, base_ch=64, latent_dim=512):
        super().__init__()
        C  = base_ch
        LC = C * 16  # 1024
        self.flat_dim = LC * 7 * 7

        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.conv = nn.Sequential(
            DecoderBlock(LC,   C*8),             # (B, 512, 14, 14)
            DecoderBlock(C*8,  C*4),             # (B, 256, 28, 28)
            DecoderBlock(C*4,  C*2),             # (B, 128, 56, 56)
            DecoderBlock(C*2,  C),               # (B,  64,112,112)
            DecoderBlock(C,    3, final=True),   # (B,   3,224,224)
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 1024, 7, 7)
        return self.conv(h)


# ---------------------------------------------------------------------------
# Full Solar VAE
# ---------------------------------------------------------------------------

class SolarVAE(nn.Module):
    """
    Variational Autoencoder for Quiet Sun reconstruction + anomaly scoring.

    Two anomaly signals:
      1. Reconstruction score  — high MSE/SSIM on flare pixels
      2. Latent deviation score — KL divergence of encoded z from N(0,I)
                                  flares land far outside the learned manifold

    Usage (inference):
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(imgs)
            scores = model.anomaly_score(imgs)  # combined score tensor (B,)
    """

    def __init__(self, in_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, base_ch, latent_dim)
        self.decoder = VAEDecoder(base_ch, latent_dim)
        self.latent_dim = latent_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick:  z = μ + ε·σ,  ε ~ N(0, I)
        Keeps the gradient flowing through μ and σ during backprop.
        At eval time this still samples — use .encode() for deterministic μ.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ, log σ²) — deterministic at inference."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            recon   : (B, 3, 224, 224)
            mu      : (B, latent_dim)
            logvar  : (B, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def anomaly_score(
        self,
        x: torch.Tensor,
        alpha: float = 0.5,
        gamma: float = 0.2,
    ) -> torch.Tensor:
        """
        Combined anomaly score (B,) — higher = more anomalous.

        score = alpha*(MSE + SSIM_loss) + gamma*KL_per_image

        Args:
            alpha : weight on reconstruction component
            gamma : weight on latent KL component
        """
        mu, logvar = self.encode(x)
        z = mu  # use mean (deterministic) at inference
        recon = self.decode(z)

        mse  = F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])
        ssim_fn = SSIMLoss().to(x.device)
        ssim_loss = torch.stack([
            ssim_fn(recon[i:i+1], x[i:i+1]) for i in range(x.size(0))
        ])
        recon_score = mse + ssim_loss

        # Per-image KL: 0.5 * sum(μ² + σ² - log σ² - 1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        kl = kl / self.latent_dim   # normalise by latent dim

        return alpha * recon_score + gamma * kl


# ---------------------------------------------------------------------------
# VAE Loss Function — ELBO + SSIM + Perceptual
# ---------------------------------------------------------------------------

class VAELoss(nn.Module):
    """
    Total Loss = Reconstruction + β·KL

    Reconstruction = MSE + λ_ssim·SSIM + λ_perc·Perceptual

    β-annealing: start with β=0 (pure reconstruction), ramp up to β_max
    over `anneal_epochs`. This prevents posterior collapse on solar images
    where the pixel-level detail is the primary training signal.
    """

    def __init__(
        self,
        beta_max:      float = 1e-4,
        anneal_epochs: int   = 20,
        lambda_ssim:   float = 0.3,
        lambda_perc:   float = 0.05,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.beta_max      = beta_max
        self.anneal_epochs = anneal_epochs
        self.lambda_ssim   = lambda_ssim
        self.lambda_perc   = lambda_perc
        self.ssim_loss     = SSIMLoss()
        self.perc_loss     = PerceptualLoss() if use_perceptual else None

    def beta(self, epoch: int) -> float:
        """Linear β warm-up over anneal_epochs."""
        return min(1.0, epoch / max(self.anneal_epochs, 1)) * self.beta_max

    def forward(
        self,
        recon:  torch.Tensor,
        target: torch.Tensor,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
        epoch:  int = 0,
    ) -> Dict[str, torch.Tensor]:

        mse  = F.mse_loss(recon, target)
        ssim = self.ssim_loss(recon, target)
        perc = self.perc_loss(recon, target) if self.perc_loss else torch.zeros(1, device=recon.device)

        # KL divergence: mean over batch, mean over latent dims
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        recon_loss = mse + self.lambda_ssim * ssim + self.lambda_perc * perc
        beta_t     = self.beta(epoch)
        total      = recon_loss + beta_t * kl

        return {
            "total":  total,
            "recon":  recon_loss,
            "mse":    mse,
            "ssim":   ssim,
            "perc":   perc,
            "kl":     kl,
            "beta":   torch.tensor(beta_t),
        }


# ---------------------------------------------------------------------------
# Rolling-Calibration Anomaly Scorer
# ---------------------------------------------------------------------------

class VAEAnomalyScorer:
    """
    Wraps SolarVAE with rolling-window threshold calibration.

    Instead of a fixed threshold, a deque of recent quiet-sun scores
    is maintained. Threshold = 95th percentile of the rolling window.
    This auto-adapts to different SUIT filter bands (NB3 vs NB8).

    Usage:
        scorer = VAEAnomalyScorer(model, device, window=500)
        score, is_flare = scorer.score(image_tensor)
        scorer.calibrate(quiet_sun_tensor)  # called by watchdog on Class-0
    """

    def __init__(
        self,
        model:      SolarVAE,
        device:     torch.device,
        threshold:  float = 0.05,
        window:     int   = 500,
        percentile: float = 95.0,
    ):
        self.model      = model.to(device).eval()
        self.device     = device
        self.threshold  = threshold
        self.percentile = percentile
        self._buffer    = deque(maxlen=window)   # rolling quiet-sun scores
        self._ckpt_dir  = CHECKPOINT_DIR

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:  x: (B, 3, 224, 224)  normalised to [0,1]
        Returns:
            scores      : (B,) combined anomaly score
            predictions : (B,) binary — 1 = Flare
        """
        x = x.to(self.device)
        scores = self.model.anomaly_score(x)
        preds  = (scores > self.threshold).long()
        return scores.cpu(), preds.cpu()

    @torch.no_grad()
    def calibrate(self, quiet_sun_batch: torch.Tensor) -> None:
        """Feed a batch of confirmed quiet-sun images to update the threshold."""
        quiet_sun_batch = quiet_sun_batch.to(self.device)
        scores = self.model.anomaly_score(quiet_sun_batch)
        self._buffer.extend(scores.cpu().tolist())

        if len(self._buffer) >= 10:
            buf = np.array(self._buffer)
            self.threshold = float(np.percentile(buf, self.percentile))

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_vae.py] Device: {device}")

    model = SolarVAE(in_channels=3, base_ch=64, latent_dim=512).to(device)
    imgs  = torch.rand(4, 3, 224, 224).to(device)

    # Forward pass
    recon, mu, logvar = model(imgs)
    assert recon.shape == imgs.shape
    assert recon.min() >= 0 and recon.max() <= 1.0

    # Loss
    criterion = VAELoss(beta_max=1e-4, anneal_epochs=20)
    losses    = criterion(recon, imgs, mu, logvar, epoch=5)
    print(f"  Loss breakdown: { {k: f'{v.item():.5f}' for k,v in losses.items()} }")

    # Anomaly scorer
    scorer = VAEAnomalyScorer(model, device)
    scores, preds = scorer.score(imgs)
    print(f"  Anomaly scores : {scores.numpy()}")
    print(f"  Predictions    : {preds.numpy()}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters     : {total:,}")
    print("[model_vae.py] PASSED ✓")
