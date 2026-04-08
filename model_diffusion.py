"""
model_diffusion.py
==================
Denoising Diffusion Probabilistic Model (DDPM) for Solar Flare Augmentation
Mission: ISRO Aditya-L1

WHY DIFFUSION FOR THIS MISSION
────────────────────────────────
Once GOES gives you 50-100 labeled flare frames, that's still too few
to train a supervised CNN-LSTM reliably. Standard augmentation (flip,
rotate) doesn't help because it only creates trivially similar copies.

A DDPM generates brand-new, physically plausible flare images by:
  1. Learning the distribution of your real solar images
  2. Conditioning on flare class (Class 0 = quiet, Class 1 = flare)
  3. Generating hundreds of synthetic flares that match the
     morphological statistics of real ones

This is the same technique used to augment rare medical conditions
(cancer lesions, retinal diseases) — exact same problem as yours:
rare positive class, cannot collect more data, need to synthesise.

ARCHITECTURE: Conditional DDPM (Ho et al. 2020 + Classifier-Free Guidance)
────────────────────────────────────────────────────────────────────────────
  Forward process: x₀ → x₁ → ... → xT (gradually add Gaussian noise)
  Reverse process: xT → ... → x₁ → x₀ (learn to denoise, conditioned on class)

  U-Net denoiser with:
    - Time-step embedding (tells model what noise level it's at)
    - Class conditioning (tells model flare vs quiet)
    - Residual connections + GroupNorm
    - Self-attention at low spatial resolutions

  Classifier-Free Guidance:
    - 15% of training: drop class conditioning (unconditional)
    - At inference: blend conditional and unconditional predictions
    - guidance_scale=7.5 → strong flare morphology
    - guidance_scale=1.0 → more diverse, less sharp

TRAINING STRATEGY
─────────────────
  Phase 1 (before GOES labels): Train unconditionally on all 53K frames
    → learns general solar morphology distribution
  Phase 2 (after GOES labels): Fine-tune with class conditioning
    → learns the conditional flare distribution
  This is called pretraining + fine-tuning with diffusion — identical to
  how Stable Diffusion is trained (large unsupervised → conditional fine-tune).
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "diffusion")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
IMAGE_DIR      = os.path.join(DATA_DIR, "images")
SYNTHETIC_DIR  = os.path.join(DATA_DIR, "synthetic_flares")
CATALOG_PATH   = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------

class CosineNoiseSchedule:
    """
    Cosine variance schedule (Nichol & Dhariwal 2021).
    Better than the linear schedule for solar images because:
    - Preserves more signal at small timesteps
    - Avoids too-rapid destruction of the thin bright flare ribbons
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        t      = torch.linspace(0, T, T + 1)
        f      = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]

        self.register_schedule(alphas_cumprod)

    def register_schedule(self, alphas_cumprod: torch.Tensor):
        betas              = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.betas         = betas.clamp(0, 0.999)
        self.alphas        = 1.0 - self.betas
        self.alphas_cumprod      = alphas_cumprod[1:]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        # Posterior variance for DDPM sampling
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) /
            (1 - self.alphas_cumprod).clamp(min=1e-8)
        )

    def to(self, device):
        for attr in ["betas","alphas","alphas_cumprod","alphas_cumprod_prev",
                     "sqrt_alphas_cumprod","sqrt_one_minus_alphas_cumprod",
                     "posterior_variance"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward process: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise


# ---------------------------------------------------------------------------
# U-Net Building Blocks
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    assert dim % 2 == 0
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with timestep + class conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 class_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop  = nn.Dropout(dropout)

        self.time_proj  = nn.Linear(time_dim, out_ch * 2)   # scale + shift
        if class_dim > 0:
            self.class_proj = nn.Linear(class_dim, out_ch * 2)
        else:
            self.class_proj = None

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                c_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Time conditioning
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        # Class conditioning
        if c_emb is not None and self.class_proj is not None:
            c_scale, c_shift = self.class_proj(c_emb).chunk(2, dim=-1)
            h = h * (1 + c_scale[:, :, None, None]) + c_shift[:, :, None, None]

        h = self.drop(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention for spatial feature maps."""

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)   # (B, H*W, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, class_dim, use_attn=False):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, time_dim, class_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t, c=None):
        x = self.attn(self.res(x, t, c))
        return self.down(x), x   # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, class_dim, use_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res  = ResBlock(in_ch + skip_ch, out_ch, time_dim, class_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t, c=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.res(x, t, c))


# ---------------------------------------------------------------------------
# Solar Denoising U-Net
# ---------------------------------------------------------------------------

class SolarUNet(nn.Module):
    """
    U-Net denoiser for solar images.
    Input: noisy image xₜ + timestep t + optional class label c
    Output: predicted noise ε (same shape as input image)

    Designed for 64×64 or 128×128 — solar images are downsampled for
    diffusion training to manage T4 memory; final size upsampled by VAE decoder
    or simple bilinear interpolation.
    """

    def __init__(
        self,
        in_channels:  int = 3,
        base_ch:      int = 64,
        ch_mult:      Tuple = (1, 2, 4, 4),
        time_dim:     int = 256,
        num_classes:  int = 2,    # 0=quiet, 1=flare
        dropout:      float = 0.1,
        attn_resols:  Tuple = (2,),   # Add attention at these U-Net levels
    ):
        super().__init__()
        self.time_dim    = time_dim
        self.num_classes = num_classes

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Class embedding (+ null class for classifier-free guidance)
        class_dim = time_dim
        self.class_emb = nn.Embedding(num_classes + 1, class_dim)
        # Index num_classes = null / unconditional token

        # Channel sizes at each level
        chs = [base_ch * m for m in ch_mult]

        # Input projection
        self.in_proj = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        # Encoder (downsampling)
        self.downs = nn.ModuleList()
        in_ch = chs[0]
        for i, out_ch in enumerate(chs[1:]):
            use_attn = i in attn_resols
            self.downs.append(
                DownBlock(in_ch, out_ch, time_dim, class_dim, use_attn)
            )
            in_ch = out_ch

        # Bottleneck
        self.mid_res1 = ResBlock(in_ch, in_ch, time_dim, class_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_res2 = ResBlock(in_ch, in_ch, time_dim, class_dim)

        # Decoder (upsampling)
        self.ups = nn.ModuleList()
        skip_chs = list(reversed(chs[:-1]))
        for i, (skip_ch, out_ch) in enumerate(zip(skip_chs, reversed(chs[:-1]))):
            use_attn = (len(chs) - 2 - i) in attn_resols
            self.ups.append(
                UpBlock(in_ch, skip_ch, out_ch, time_dim, class_dim, use_attn)
            )
            in_ch = out_ch

        # Output projection
        self.out_norm = nn.GroupNorm(min(32, in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, in_channels, 3, padding=1)

    def forward(
        self,
        x:       torch.Tensor,        # (B, C, H, W) noisy image
        t:       torch.Tensor,        # (B,) integer timesteps
        c:       Optional[torch.Tensor] = None,  # (B,) class labels (None=unconditional)
    ) -> torch.Tensor:
        """Returns predicted noise ε of same shape as x."""
        B = x.shape[0]

        # Time embedding
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Class embedding (null token if c is None)
        if c is None:
            c_idx = torch.full((B,), self.num_classes, dtype=torch.long, device=x.device)
        else:
            c_idx = c.long()
        c_emb = self.class_emb(c_idx)

        # U-Net
        h = self.in_proj(x)
        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb, c_emb)
            skips.append(skip)

        h = self.mid_res2(self.mid_attn(self.mid_res1(h, t_emb, c_emb)), t_emb, c_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, t_emb, c_emb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# DDPM Trainer / Sampler
# ---------------------------------------------------------------------------

class SolarDDPM:
    """
    Wraps SolarUNet with training and sampling utilities.

    Classifier-Free Guidance:
      During training: randomly replace class label with null token (prob=0.15)
      At sampling: blend conditional and unconditional predictions
        ε_guided = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
    """

    def __init__(
        self,
        model:     SolarUNet,
        device:    torch.device,
        T:         int   = 1000,
        cfg_prob:  float = 0.15,  # Probability of dropping class label during training
    ):
        self.model     = model.to(device)
        self.device    = device
        self.cfg_prob  = cfg_prob
        self.schedule  = CosineNoiseSchedule(T).to(device)
        self.T         = T

    def training_loss(
        self,
        x0: torch.Tensor,
        c:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDPM training loss: predict the noise added at a random timestep."""
        B  = x0.shape[0]
        t  = torch.randint(0, self.T, (B,), device=self.device)

        xt, noise = self.schedule.q_sample(x0, t)

        # Classifier-free guidance: randomly drop class labels
        if c is not None and self.cfg_prob > 0:
            drop_mask = torch.rand(B, device=self.device) < self.cfg_prob
            c = c.clone()
            c[drop_mask] = self.model.num_classes   # Null token

        predicted_noise = self.model(xt, t, c)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        n_samples:       int,
        img_size:        int             = 64,
        class_label:     Optional[int]   = 1,     # 1=flare, 0=quiet
        guidance_scale:  float           = 7.5,
        channels:        int             = 3,
        show_progress:   bool            = True,
    ) -> torch.Tensor:
        """
        Generate synthetic solar images using DDPM reverse process.

        Args:
            class_label: 0=quiet sun, 1=solar flare (None=unconditional)
            guidance_scale: 1.0=no guidance, 7.5=strong conditioning
        """
        self.model.eval()
        x = torch.randn(n_samples, channels, img_size, img_size, device=self.device)

        c_cond   = None
        c_uncond = None
        if class_label is not None:
            c_cond   = torch.full((n_samples,), class_label,
                                  dtype=torch.long, device=self.device)
            c_uncond = torch.full((n_samples,), self.model.num_classes,
                                  dtype=torch.long, device=self.device)

        sched = self.schedule
        steps = range(self.T - 1, -1, -1)
        if show_progress:
            try:
                from tqdm import tqdm
                steps = tqdm(steps, desc="Sampling")
            except ImportError:
                pass

        for t_val in steps:
            t_batch = torch.full((n_samples,), t_val, device=self.device, dtype=torch.long)

            # Classifier-free guidance
            if c_cond is not None and guidance_scale > 1.0:
                eps_cond   = self.model(x, t_batch, c_cond)
                eps_uncond = self.model(x, t_batch, c_uncond)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.model(x, t_batch, c_cond)

            # DDPM reverse step
            alpha      = sched.alphas[t_val]
            alpha_bar  = sched.alphas_cumprod[t_val]
            sigma      = sched.posterior_variance[t_val].sqrt()

            # Predict x₀ from noise prediction
            x0_pred = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            if t_val > 0:
                noise = torch.randn_like(x)
                x = (alpha_bar.sqrt() * x0_pred +
                     (1 - alpha_bar).sqrt() * eps +
                     sigma * noise)
            else:
                x = x0_pred

        return (x.clamp(-1, 1) + 1) / 2   # Normalise to [0, 1]

    def save_synthetic_dataset(
        self,
        n_flares:   int = 500,
        n_quiet:    int = 500,
        img_size:   int = 128,
        output_dir: str = SYNTHETIC_DIR,
    ) -> None:
        """
        Generate and save a synthetic balanced dataset.
        Saves PNGs + a catalog CSV with labels.
        """
        from torchvision.utils import save_image
        import pandas as pd

        rows = []
        for cls, name, n in [(1, "flare", n_flares), (0, "quiet", n_quiet)]:
            log.info(f"Generating {n} synthetic {name} images...")
            imgs = self.sample(n, img_size, class_label=cls, guidance_scale=7.5)
            for i, img in enumerate(imgs):
                fname = f"synthetic_{name}_{i:04d}.png"
                path  = os.path.join(output_dir, fname)
                save_image(img, path)
                rows.append({"filename": fname, "label": cls, "source": "synthetic"})

        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "synthetic_catalog.csv"), index=False
        )
        log.info(f"Synthetic dataset saved to {output_dir}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import pandas as pd
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--catalog",     default=CATALOG_PATH)
    p.add_argument("--image_dir",   default=IMAGE_DIR)
    p.add_argument("--output_dir",  default=CHECKPOINT_DIR)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--img_size",    type=int,   default=64)    # 64px for T4 memory
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--workers",     type=int,   default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DiffusionDataset(Dataset):
        def __init__(self, catalog, image_dir, size):
            df = pd.read_csv(catalog, low_memory=False)
            df.columns = df.columns.str.strip()
            fname_col = next((c for c in ["Filename","filename"] if c in df.columns), None)
            df = df.rename(columns={fname_col: "filename"})
            df["filename"] = df["filename"].astype(str).apply(
                lambda x: x.rsplit(".", 1)[0] + ".png"
            )
            image_dir_p = Path(image_dir)
            df = df[df["filename"].apply(lambda f: (image_dir_p / f).exists())]
            self.filenames = df["filename"].tolist()
            self.labels    = df.get("label", pd.Series([0]*len(df))).tolist()
            self.image_dir = image_dir_p
            self.tf = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),  # [-1, 1] for DDPM
            ])
        def __len__(self): return len(self.filenames)
        def __getitem__(self, i):
            img = Image.open(self.image_dir / self.filenames[i]).convert("RGB")
            return self.tf(img), torch.tensor(self.labels[i], dtype=torch.long)

    dataset = DiffusionDataset(args.catalog, args.image_dir, args.img_size)
    loader  = DataLoader(dataset, args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)

    model  = SolarUNet(base_ch=args.base_ch)
    ddpm   = SolarDDPM(model, device)
    optim  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best   = float("inf")

    log.info(f"U-Net params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x0, c in loader:
            x0 = x0.to(device); c = c.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = ddpm.training_loss(x0, c)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item()

        mean_loss = total / len(loader)
        log.info(f"Epoch {epoch}/{args.epochs} | DDPM Loss: {mean_loss:.5f}")

        if mean_loss < best:
            best = mean_loss
            torch.save({"epoch": epoch, "loss": mean_loss,
                        "model_state": model.state_dict()},
                       os.path.join(args.output_dir, "best_ddpm.pt"))

        # Every 10 epochs: generate sample flares
        if epoch % 10 == 0:
            log.info("Generating sample flares...")
            samples = ddpm.sample(4, args.img_size, class_label=1, guidance_scale=7.5)
            from torchvision.utils import save_image
            save_image(samples, os.path.join(args.output_dir, f"samples_e{epoch}.png"),
                       nrow=2)

    log.info("DDPM training complete.")


if __name__ == "__main__":
    main()
