"""
wavelet_features.py
===================
2D Wavelet Feature Extraction Pre-Processing for Solar FITS Images
Mission: ISRO Aditya-L1

WHY WAVELETS BEFORE THE CNN?
──────────────────────────────
Solar flares produce HIGH-FREQUENCY SPATIAL BURSTS:
  • Impulsive phase: sub-second brightening in localised pixels
  • Flare ribbons: thin sharp edges (high-frequency detail)
  • Type II radio bursts: coherent structure across spatial scales

A standard CNN processes spatial information at multiple scales through
its pooling layers, but it's not explicitly designed to separate
"background solar activity" (smooth, low-frequency) from
"flare signatures" (sharp, high-frequency).

2D Discrete Wavelet Transform (DWT) decomposes each frame into:
  ┌─────────────┬─────────────┐
  │   LL (cA)   │   LH (cH)  │
  │ Approximation│ Horizontal │
  │  (low-freq) │  detail    │
  ├─────────────┼─────────────┤
  │   HL (cV)   │   HH (cD)  │
  │  Vertical   │  Diagonal  │
  │   detail    │   detail   │
  └─────────────┴─────────────┘

The DETAIL BANDS (LH, HL, HH) carry flare signatures.
Passing these through the CNN alongside the original image
gives it explicit access to multi-scale spatial structure.

PRACTICAL OUTPUTS:
  1. WaveletAugmentor  — adds wavelet detail bands as extra input channels
                         (3 RGB + 3 detail = 6-channel input to the CNN)
  2. WaveletFeatureExtractor — produces 12 scalar statistics for XGBoost
  3. WaveletPreprocessor     — replaces raw PNG with wavelet-decomposed PNG
                               saved to processed_images/wavelet/
"""

import os
import numpy as np
import pywt
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_IMG_DIR = os.path.join(PROJECT_DIR, "processed_images")
WAVELET_DIR       = os.path.join(PROCESSED_IMG_DIR, "wavelet")
WAVELET_CSV       = os.path.join(PROJECT_DIR, "aditya_l1_wavelet_features.csv")
LOG_DIR           = os.path.join(PROJECT_DIR, "logs")

os.makedirs(WAVELET_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Single-level 2D DWT decomposition helper
# ---------------------------------------------------------------------------

def dwt2_image(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Apply 2D DWT to a single-channel grayscale image.

    Args:
        image   : (H, W) float32 in [0, 1]
        wavelet : 'haar' (fastest), 'db4' (smoother), 'sym4' (symmetric)
        level   : Decomposition level

    Returns:
        cA          : Approximation coefficients (H/2, W/2)
        (cH, cV, cD): Horizontal, Vertical, Diagonal detail coefficients
    """
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs
    return cA, (cH, cV, cD)


def dwt2_multichannel(
    image_rgb: np.ndarray,
    wavelet: str = "haar",
) -> Dict[str, np.ndarray]:
    """
    Apply DWT to each RGB channel independently.

    Returns dict with keys: cA_r, cA_g, cA_b, cH_r, cH_g, cH_b, etc.
    """
    result = {}
    for i, ch_name in enumerate(["r", "g", "b"]):
        cA, (cH, cV, cD) = dwt2_image(image_rgb[:, :, i].astype(np.float32), wavelet)
        result[f"cA_{ch_name}"] = cA
        result[f"cH_{ch_name}"] = cH
        result[f"cV_{ch_name}"] = cV
        result[f"cD_{ch_name}"] = cD
    return result


# ---------------------------------------------------------------------------
# 2.  WaveletAugmentor — adds detail bands as extra CNN input channels
# ---------------------------------------------------------------------------

class WaveletChannelAugmentor:
    """
    Produces a (6, 224, 224) tensor: 3 original RGB channels +
    3 fused detail band channels (magnitude of cH, cV, cD averaged over RGB).

    This lets the existing EfficientNet-B0 (or any CNN) receive wavelet
    information with ZERO architectural changes — just change in_channels to 6.

    The detail channels are:
      ch4: sqrt(cH²_mean_across_rgb)   — horizontal edge energy
      ch5: sqrt(cV²_mean_across_rgb)   — vertical edge energy
      ch6: sqrt(cD²_mean_across_rgb)   — diagonal/texture energy

    Usage:
        aug = WaveletChannelAugmentor()
        tensor_6ch = aug.augment(image_tensor_3ch)   # (3,224,224) → (6,224,224)
    """

    def __init__(self, wavelet: str = "db4", target_size: int = 224):
        self.wavelet     = wavelet
        self.target_size = target_size

    def augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: (3, H, W) float32 in [0, 1]
        Returns:
            (6, H, W) — original RGB + 3 wavelet detail channels
        """
        img_np = tensor.permute(1, 2, 0).numpy()    # (H, W, 3)
        H, W   = img_np.shape[:2]

        detail_maps = []
        for i in range(3):
            _, (cH, cV, cD) = dwt2_image(img_np[:, :, i], self.wavelet)
            # Combine all detail bands into one energy map
            energy = np.sqrt(cH**2 + cV**2 + cD**2 + 1e-8)
            detail_maps.append(energy)

        # Average detail energy across RGB channels → 3 maps (one per detail band)
        cH_all = []; cV_all = []; cD_all = []
        for i in range(3):
            _, (cH, cV, cD) = dwt2_image(img_np[:, :, i], self.wavelet)
            cH_all.append(cH); cV_all.append(cV); cD_all.append(cD)

        detail_cH = np.mean(np.stack(cH_all), axis=0)
        detail_cV = np.mean(np.stack(cV_all), axis=0)
        detail_cD = np.mean(np.stack(cD_all), axis=0)

        extra_channels = []
        for band in [detail_cH, detail_cV, detail_cD]:
            # Resize back to original resolution
            band_rs = cv2.resize(np.abs(band), (W, H), interpolation=cv2.INTER_LINEAR)
            # Normalise to [0, 1]
            band_rs = (band_rs - band_rs.min()) / (band_rs.max() - band_rs.min() + 1e-8)
            extra_channels.append(torch.from_numpy(band_rs).float())

        # Stack: (3, H, W) original + (3, H, W) wavelet details
        wavelet_stack = torch.stack(extra_channels)            # (3, H, W)
        return torch.cat([tensor, wavelet_stack], dim=0)        # (6, H, W)

    def augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Augments a batch (B, 3, H, W) → (B, 6, H, W)."""
        return torch.stack([self.augment(batch[i]) for i in range(batch.size(0))])


# ---------------------------------------------------------------------------
# 3.  WaveletFeatureExtractor — scalar statistics for XGBoost / EDA
# ---------------------------------------------------------------------------

class WaveletFeatureExtractor:
    """
    Extracts 12 physics-motivated scalar statistics from the wavelet
    decomposition of a FITS image.

    These features are appended to the aditya_l1_catalog.csv by
    an upgraded version of process_fits.py and fed to train_xgboost.py.

    Features (per frame):
        detail_energy_H, V, D      — Total energy in each detail band
        detail_max_H, V, D         — Peak coefficient (localized brightening)
        detail_entropy_H, V, D     — Shannon entropy of detail band
        approx_energy              — Energy in approximation band (DC component)
        detail_to_approx_ratio     — Ratio: flares push this high
        high_freq_fraction         — Fraction of pixels > 2σ in detail bands
    """

    def __init__(self, wavelet: str = "db4", levels: int = 2):
        self.wavelet = wavelet
        self.levels  = levels

    def _entropy(self, coeffs: np.ndarray) -> float:
        """Shannon entropy of absolute wavelet coefficients."""
        flat = np.abs(coeffs).flatten()
        flat = flat / (flat.sum() + 1e-10)
        flat = flat[flat > 0]
        return float(-np.sum(flat * np.log2(flat + 1e-10)))

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Args:
            image: (H, W) or (H, W, 3) — float32 in [0, 1]
        Returns:
            dict of 12 scalar features
        """
        if image.ndim == 3:
            gray = image.mean(axis=2)
        else:
            gray = image.astype(np.float32)

        cA, (cH, cV, cD) = dwt2_image(gray, self.wavelet)

        e_H = float(np.sum(cH**2))
        e_V = float(np.sum(cV**2))
        e_D = float(np.sum(cD**2))
        e_A = float(np.sum(cA**2))
        detail_energy = e_H + e_V + e_D

        # High-frequency fraction (proxy for flare ribbon density)
        all_detail = np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
        threshold  = all_detail.mean() + 2 * all_detail.std()
        hf_frac    = float((np.abs(all_detail) > threshold).mean())

        return {
            "wav_detail_energy_H":   e_H,
            "wav_detail_energy_V":   e_V,
            "wav_detail_energy_D":   e_D,
            "wav_max_H":             float(np.abs(cH).max()),
            "wav_max_V":             float(np.abs(cV).max()),
            "wav_max_D":             float(np.abs(cD).max()),
            "wav_entropy_H":         self._entropy(cH),
            "wav_entropy_V":         self._entropy(cV),
            "wav_entropy_D":         self._entropy(cD),
            "wav_approx_energy":     e_A,
            "wav_detail_to_approx":  detail_energy / (e_A + 1e-8),
            "wav_hf_fraction":       hf_frac,
        }

    def extract_from_path(self, image_path: str) -> Dict[str, float]:
        """Load a PNG and extract features."""
        img = cv2.imread(str(image_path))
        if img is None:
            return {k: 0.0 for k in [
                "wav_detail_energy_H","wav_detail_energy_V","wav_detail_energy_D",
                "wav_max_H","wav_max_V","wav_max_D",
                "wav_entropy_H","wav_entropy_V","wav_entropy_D",
                "wav_approx_energy","wav_detail_to_approx","wav_hf_fraction"
            ]}
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return self.extract(img)


# ---------------------------------------------------------------------------
# 4.  WaveletPreprocessor — batch processes all PNGs and saves decomposed images
# ---------------------------------------------------------------------------

class WaveletPreprocessor:
    """
    Runs as a daemon alongside process_fits.py.
    Watches processed_images/ and for each new PNG:
      1. Computes wavelet decomposition
      2. Saves a false-colour detail-band composite PNG
      3. Appends scalar features to wavelet_features.csv

    The false-colour composite uses:
      R=cH (horizontal edges), G=cV (vertical edges), B=cD (diagonal/texture)
    This makes flare ribbons and active regions immediately visible as
    bright multi-colour features on a dark background.
    """

    def __init__(
        self,
        input_dir:  str = PROCESSED_IMG_DIR,
        output_dir: str = WAVELET_DIR,
        csv_path:   str = WAVELET_CSV,
        wavelet:    str = "db4",
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.csv_path   = csv_path
        self.extractor  = WaveletFeatureExtractor(wavelet)

    def process_image(self, png_path: str) -> Optional[Dict]:
        """Process a single PNG. Returns feature dict or None on failure."""
        try:
            img = cv2.imread(str(png_path))
            if img is None: return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Extract per-channel detail bands for visualisation
            all_cH, all_cV, all_cD = [], [], []
            for i in range(3):
                _, (cH, cV, cD) = dwt2_image(img_rgb[:, :, i], self.extractor.wavelet)
                all_cH.append(np.abs(cH))
                all_cV.append(np.abs(cV))
                all_cD.append(np.abs(cD))

            H_vis = np.mean(all_cH, axis=0)
            V_vis = np.mean(all_cV, axis=0)
            D_vis = np.mean(all_cD, axis=0)

            def norm(a):
                return (255 * (a - a.min()) / (a.max() - a.min() + 1e-8)).astype(np.uint8)

            composite = cv2.merge([norm(H_vis), norm(V_vis), norm(D_vis)])  # BGR
            h, w = img.shape[:2]
            composite = cv2.resize(composite, (w // 2, h // 2))             # Half res

            stem    = Path(png_path).stem
            out_png = self.output_dir / f"{stem}_wavelet.png"
            cv2.imwrite(str(out_png), composite)

            features = self.extractor.extract(img_rgb)
            features["filename"] = Path(png_path).name
            return features

        except Exception as e:
            log.warning(f"[Wavelet] Failed on {png_path}: {e}")
            return None

    def run_batch(self, png_list: List[str]) -> None:
        """Process a list of PNGs and append features to CSV."""
        import pandas as pd
        rows = []
        for p in png_list:
            feat = self.process_image(p)
            if feat: rows.append(feat)

        if rows:
            df = pd.DataFrame(rows)
            if os.path.exists(self.csv_path):
                df.to_csv(self.csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(self.csv_path, index=False)
            log.info(f"[Wavelet] Appended {len(rows)} rows to {self.csv_path}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[wavelet_features.py] Running sanity check...")

    # Synthetic solar image with a bright "flare" patch
    h, w = 224, 224
    img  = np.random.uniform(0.2, 0.4, (h, w, 3)).astype(np.float32)
    cv2.circle(img, (100, 100), 15, (1.0, 0.9, 0.7), -1)   # Bright flare spot

    # WaveletFeatureExtractor
    extractor = WaveletFeatureExtractor(wavelet="db4")
    feats     = extractor.extract(img)
    print(f"  Scalar features ({len(feats)}):")
    for k, v in feats.items():
        print(f"    {k}: {v:.5f}")

    # WaveletChannelAugmentor
    tensor = torch.from_numpy(img.transpose(2, 0, 1))       # (3, H, W)
    aug    = WaveletChannelAugmentor(wavelet="db4")
    out    = aug.augment(tensor)
    assert out.shape == (6, h, w), f"Expected (6,{h},{w}), got {out.shape}"
    print(f"  Augmented tensor shape: {out.shape}")
    print(f"  Channel value ranges: {[(f'{out[i].min():.2f}', f'{out[i].max():.2f}') for i in range(6)]}")

    # DWT coefficients
    gray = img.mean(axis=2)
    cA, (cH, cV, cD) = dwt2_image(gray, "haar")
    print(f"  DWT (Haar) | cA:{cA.shape} cH:{cH.shape} cV:{cV.shape} cD:{cD.shape}")

    print("[wavelet_features.py] PASSED ✓")

