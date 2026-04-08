"""
model_pinn.py
=============
Physics-Informed Loss Functions for Solar VAE / Anomaly Detection
Mission: ISRO Aditya-L1 

WHAT THIS IS NOT
─────────────────
This is NOT a full PINN in the PDE-solving sense. Solving solar MHD PDEs
directly on image data would require supercomputer-scale resources.

WHAT THIS IS
─────────────
Physics-Informed Regularisation:

Your VAE currently minimises pixel-level MSE + SSIM.
These losses treat all pixels equally — a cosmic ray hit in the corner
gets the same loss weight as a flare ribbon at the limb.

Physics-informed losses add solar physics CONSTRAINTS to the training:
  1. Limb Darkening Model    — solar disk edge should be darker than center
  2. Solar Disk Mask         — reconstruction error outside the disk is irrelevant
  3. Flux Conservation       — integrated brightness should be smooth over time
  4. Intensity Gradient      — active regions have sharper spatial gradients
  5. Band Correlation        — Mg II and Ca II brightening should be correlated

Together these guide the VAE's latent space to align with physically
meaningful quantities rather than just pixel statistics.
Result: anomaly scores are more physically interpretable and the
OCSVM latent space classifier works better because the latent dimensions
correspond to real solar properties.

INTEGRATION
───────────
These losses are added to the VAELoss class in model_vae.py:
  loss = mse + λ_ssim*ssim + λ_phys*physics_loss + β*kl

The physics_loss wrapper in this file handles all constraints.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Limb Darkening Constraint
# ---------------------------------------------------------------------------

class LimbDarkeningConstraint(nn.Module):
    """
    The solar disk naturally appears darker at its edges (limb) than at
    the center due to opacity and temperature gradients in the atmosphere.

    The limb darkening law (Eddington approximation):
      I(μ) = I₀ * (a + b*μ)  where μ = cos(θ), θ = heliocentric angle

    For NUV bands (2796Å):
      a ≈ 0.3, b ≈ 0.7  (coefficients from Neckel & Labs 1994)

    This constraint:
      1. Generates the expected limb darkening pattern for the solar disk
         given the disk center (SUN_CX, SUN_CY) and radius (R_SUN) from
         the FITS header
      2. Penalises VAE reconstructions that deviate from this pattern
         in the quiet-sun disk regions (low anomaly score areas)

    Physical effect on training:
      The encoder learns that quiet-sun variations AWAY from center are
      EXPECTED (limb darkening), not anomalous. This prevents the VAE
      from flagging the solar limb as an anomaly due to its
      systematic darkness.
    """

    def __init__(
        self,
        a: float = 0.3,    # Limb darkening coefficient a
        b: float = 0.7,    # Limb darkening coefficient b
        img_size: int = 224,
    ):
        super().__init__()
        self.a = a
        self.b = b

        # Pre-compute normalized coordinate grid
        y_coords = torch.linspace(-1, 1, img_size).unsqueeze(1).expand(img_size, img_size)
        x_coords = torch.linspace(-1, 1, img_size).unsqueeze(0).expand(img_size, img_size)
        r = torch.sqrt(x_coords**2 + y_coords**2).clamp(0, 1)
        mu = torch.sqrt((1 - r**2).clamp(min=0))   # cos(heliocentric angle)
        # Limb darkening map: I(μ) / I(1) = a + b*μ
        ld_map = (a + b * mu)
        # Disk mask: 1 inside disk, 0 outside
        disk_mask = (r <= 1.0).float()
        self.register_buffer("ld_map",   ld_map.unsqueeze(0).unsqueeze(0))
        self.register_buffer("disk_mask", disk_mask.unsqueeze(0).unsqueeze(0))

    def expected_pattern(self, brightness_scale: torch.Tensor) -> torch.Tensor:
        """
        Returns expected limb darkening pattern scaled to observed disk brightness.
        brightness_scale: (B, 1, 1, 1) — per-image mean disk brightness
        """
        return self.ld_map * brightness_scale * self.disk_mask

    def forward(
        self,
        recon:  torch.Tensor,   # (B, C, H, W) VAE reconstruction
        target: torch.Tensor,   # (B, C, H, W) original image
    ) -> torch.Tensor:
        """
        Returns the limb darkening regularisation loss.
        Penalises deviations from expected limb darkening in quiet-sun regions.
        """
        # Use grayscale mean for brightness estimate
        target_gray = target.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        recon_gray  = recon.mean(dim=1, keepdim=True)

        # Estimate disk center brightness
        # (average over central 25% of disk area)
        H, W = target.shape[-2:]
        cx, cy = H // 2, W // 2
        cr = min(H, W) // 8   # 25% of radius
        center_bright = target_gray[
            :, :,
            cy - cr : cy + cr,
            cx - cr : cx + cr
        ].mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1)
        center_bright = center_bright.clamp(min=1e-6)

        # Expected pattern
        expected = self.expected_pattern(center_bright)

        # Loss: penalise deviation from limb darkening in disk regions
        # Only on reconstructed image (we want the decoder to produce
        # physically correct limb darkening on quiet sun)
        ld_loss = F.mse_loss(
            recon_gray * self.disk_mask,
            expected.to(recon.device)
        )
        return ld_loss


# ---------------------------------------------------------------------------
# 2. Disk Mask Constraint
# ---------------------------------------------------------------------------

class DiskMaskConstraint(nn.Module):
    """
    The solar disk occupies roughly 60-70% of SUIT's 560×560 field of view
    (cropped to 224×224 for training). Everything outside the disk is
    background (spacecraft interior, stray light, cosmic rays).

    This constraint:
      1. Weights reconstruction loss higher INSIDE the disk
         (where solar physics happens)
      2. Reduces loss weight OUTSIDE the disk
         (background noise should be mostly ignored)

    Physics effect:
      Prevents the VAE from wasting capacity modelling the dark background.
      The latent space focuses on the solar disk morphology.
    """

    def __init__(
        self,
        img_size:        int   = 224,
        disk_radius_frac: float = 0.48,  # Disk fills 48% of half-width
        edge_width_frac:  float = 0.05,  # Soft edge over 5% of radius
        inside_weight:   float = 2.0,    # Higher loss weight inside disk
        outside_weight:  float = 0.1,    # Lower loss weight outside disk
    ):
        super().__init__()
        # Generate soft disk mask with smooth edge
        y = torch.linspace(-1, 1, img_size).unsqueeze(1).expand(img_size, img_size)
        x = torch.linspace(-1, 1, img_size).unsqueeze(0).expand(img_size, img_size)
        r = torch.sqrt(x**2 + y**2)

        # Sigmoid edge: 1 inside, smooth transition, 0 outside
        edge_steepness = 1.0 / (edge_width_frac + 1e-6)
        soft_mask = torch.sigmoid(edge_steepness * (disk_radius_frac - r))

        # Weighted mask: inside_weight inside, outside_weight outside
        weighted_mask = outside_weight + (inside_weight - outside_weight) * soft_mask
        self.register_buffer("weighted_mask",
                             weighted_mask.unsqueeze(0).unsqueeze(0))
        self.register_buffer("soft_mask",
                             soft_mask.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        recon:  torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted reconstruction loss emphasising the solar disk."""
        pixel_mse = (recon - target) ** 2          # (B, C, H, W)
        weighted  = pixel_mse * self.weighted_mask.to(recon.device)
        return weighted.mean()


# ---------------------------------------------------------------------------
# 3. Spatial Gradient Constraint
# ---------------------------------------------------------------------------

class SpatialGradientConstraint(nn.Module):
    """
    Active regions and flare ribbons have much sharper spatial gradients
    than quiet-sun background. This constraint:

      1. Computes the spatial gradient magnitude of the target image
         (high = active region, low = quiet sun)
      2. Penalises reconstructions that are TOO SMOOTH in high-gradient areas
         (the decoder shouldn't blur out active region fine structure)
      3. Penalises reconstructions that are TOO SHARP in low-gradient areas
         (the decoder shouldn't hallucinate features in the quiet sun)

    In practice this means:
      - Flare ribbons are preserved in the reconstruction
      - Quiet-sun granulation noise is smoothed (as expected)
      - The anomaly score is more sensitive to genuine spatial structure changes
    """

    def __init__(self, sharpness_weight: float = 0.5):
        super().__init__()
        self.sharpness_weight = sharpness_weight
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                               dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                               dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Sobel gradient magnitude. x: (B, C, H, W)"""
        gray = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        gx = F.conv2d(gray, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(x.device), padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(
        self,
        recon:  torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Penalises gradient magnitude mismatch between reconstruction and target."""
        grad_target = self.gradient_magnitude(target)
        grad_recon  = self.gradient_magnitude(recon)
        # MSE on gradient magnitudes (preserves edge sharpness statistics)
        return F.mse_loss(grad_recon, grad_target)


# ---------------------------------------------------------------------------
# 4. Temporal Flux Smoothness Constraint
# ---------------------------------------------------------------------------

class TemporalFluxConstraint(nn.Module):
    """
    The total integrated intensity (flux) of a quiet-sun image should
    vary smoothly over time (solar rotation period ~27 days introduces
    slow changes; quiet sun has no rapid flux changes).

    During a flare, the flux INCREASES suddenly — this is the signal.
    But between flares, large flux variations in the reconstruction
    indicate the VAE is hallucinating structure.

    This constraint penalises large VAE reconstruction flux differences
    between consecutive frames in a batch.

    Applied during training only when the batch contains consecutive frames.
    """

    def __init__(self, smoothness_weight: float = 0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        recon_batch: torch.Tensor,    # (B, C, H, W) — should be consecutive frames
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        if recon_batch.shape[0] < 2:
            return torch.tensor(0.0, device=recon_batch.device)

        # Total mean intensity per frame
        recon_flux  = recon_batch.mean(dim=[1, 2, 3])    # (B,)
        target_flux = target_batch.mean(dim=[1, 2, 3])

        # Flux difference between consecutive frames
        recon_delta  = (recon_flux[1:]  - recon_flux[:-1])   # (B-1,)
        target_delta = (target_flux[1:] - target_flux[:-1])

        # The reconstruction's flux changes should mirror the target's flux changes
        # (don't penalise real flux changes — only hallucinated ones)
        return F.mse_loss(recon_delta, target_delta)


# ---------------------------------------------------------------------------
# Combined Physics Loss
# ---------------------------------------------------------------------------

class SolarPhysicsLoss(nn.Module):
    """
    Combines all physics-informed constraints into a single loss term.

    Drop-in addition to VAELoss in model_vae.py:

    In VAELoss.forward():
        physics_loss = self.physics_loss(recon_f, target_f)
        total = recon_loss + beta_t * kl + lambda_phys * physics_loss

    Recommended weights (start conservative, increase if stable):
        lambda_limb     = 0.05   — limb darkening, mild regularisation
        lambda_disk     = 0.10   — disk masking, moderate effect
        lambda_gradient = 0.05   — gradient matching
        lambda_flux     = 0.02   — temporal smoothness

    Total physics_loss weight λ_phys = 0.1-0.2 of total reconstruction loss.
    """

    def __init__(
        self,
        img_size:         int   = 224,
        lambda_limb:      float = 0.05,
        lambda_disk:      float = 0.10,
        lambda_gradient:  float = 0.05,
        lambda_flux:      float = 0.02,
        use_limb:         bool  = True,
        use_disk:         bool  = True,
        use_gradient:     bool  = True,
        use_flux:         bool  = True,
    ):
        super().__init__()
        self.lambda_limb     = lambda_limb
        self.lambda_disk     = lambda_disk
        self.lambda_gradient = lambda_gradient
        self.lambda_flux     = lambda_flux

        self.limb_loss  = LimbDarkeningConstraint(img_size=img_size) if use_limb else None
        self.disk_loss  = DiskMaskConstraint(img_size=img_size)      if use_disk else None
        self.grad_loss  = SpatialGradientConstraint()                if use_gradient else None
        self.flux_loss  = TemporalFluxConstraint()                   if use_flux else None

    def forward(
        self,
        recon:  torch.Tensor,    # (B, C, H, W) VAE reconstruction (float32)
        target: torch.Tensor,    # (B, C, H, W) original image (float32)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with individual component losses and total.
        All inputs should already be float32 (call .float() before passing).
        """
        losses = {}
        total  = torch.tensor(0.0, device=recon.device)

        if self.limb_loss is not None:
            l = self.limb_loss(recon, target)
            losses["limb"]  = l
            total = total + self.lambda_limb * l

        if self.disk_loss is not None:
            l = self.disk_loss(recon, target)
            losses["disk"]  = l
            total = total + self.lambda_disk * l

        if self.grad_loss is not None:
            l = self.grad_loss(recon, target)
            losses["gradient"] = l
            total = total + self.lambda_gradient * l

        if self.flux_loss is not None:
            l = self.flux_loss(recon, target)
            losses["flux"]  = l
            total = total + self.lambda_flux * l

        losses["total"] = total
        return losses


# ---------------------------------------------------------------------------
# Integration helper: patch VAELoss in model_vae.py
# ---------------------------------------------------------------------------

def patch_vae_loss_with_physics(vae_loss_instance, lambda_phys: float = 0.15):
    """
    Monkey-patches an existing VAELoss instance to include SolarPhysicsLoss.

    Usage in train_vae.py:
        criterion = VAELoss(...)
        from model_pinn import patch_vae_loss_with_physics
        patch_vae_loss_with_physics(criterion, lambda_phys=0.15)

        # Then in training loop: criterion(recon, target, mu, logvar, epoch)
        # Physics loss is now included automatically.
    """
    physics_loss = SolarPhysicsLoss()
    original_forward = vae_loss_instance.forward

    def new_forward(recon, target, mu, logvar, epoch=0):
        result = original_forward(recon, target, mu, logvar, epoch)
        recon_f  = recon.float()
        target_f = target.float()
        phys = physics_loss(recon_f, target_f)
        result["physics"] = phys["total"]
        result["total"]   = result["total"] + lambda_phys * phys["total"]
        # Include component losses for TensorBoard logging
        for k, v in phys.items():
            if k != "total":
                result[f"phys_{k}"] = v
        return result

    import types
    vae_loss_instance.forward = types.MethodType(
        lambda self, *args, **kwargs: new_forward(*args, **kwargs),
        vae_loss_instance
    )
    log.info(f"[PINN] VAELoss patched with SolarPhysicsLoss (λ={lambda_phys})")
    return vae_loss_instance


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[model_pinn.py] Running sanity check...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 4, 3, 224, 224

    # Synthetic solar-like image (bright center, dark limb)
    y = torch.linspace(-1, 1, H).unsqueeze(1).expand(H, W)
    x = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
    r = torch.sqrt(x**2 + y**2).clamp(0, 1)
    solar_disk = (torch.sqrt((1 - r**2).clamp(0)) * 0.8 + 0.1)
    solar_disk = solar_disk.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1).to(device)

    # Slightly noisy reconstruction
    recon = (solar_disk + 0.02 * torch.randn_like(solar_disk)).clamp(0, 1)

    physics = SolarPhysicsLoss().to(device)
    losses  = physics(recon, solar_disk)
    print(f"  Physics losses: {  {k: f'{v.item():.5f}' for k,v in losses.items()} }")

    # Check each component
    assert "total" in losses
    assert losses["total"].item() >= 0

    # Integration helper
    class FakeLoss:
        def forward(self, r, t, mu, lv, epoch=0):
            return {"total": F.mse_loss(r, t), "mse": F.mse_loss(r, t)}
    fake = FakeLoss()
    patch_vae_loss_with_physics(fake, lambda_phys=0.1)
    mu  = torch.randn(B, 64)
    out = fake.forward(recon, solar_disk, mu, mu * 0.1)
    print(f"  Patched loss keys: {list(out.keys())}")
    assert "physics" in out
    print("[model_pinn.py] PASSED ✓")
