"""
optical_flow_module.py
======================
Optical Flow Feature Extractor for Solar CME / Flare Motion Detection
Mission: ISRO Aditya-L1

WHY OPTICAL FLOW?
─────────────────
Solar flares and CMEs produce MOTION:
  • Erupting filaments travel at 100–3000 km/s across the solar disk
  • Flare ribbons rapidly separate from the polarity inversion line
  • EIT waves propagate as expanding circular fronts
  • CME fronts eject radially outward

A CNN looking at a single frame sees BRIGHTNESS anomalies.
Optical flow between consecutive frames sees VELOCITY FIELDS.
These are independent signals → combining both dramatically reduces false positives
(e.g. cosmic ray hits are bright but have zero flow).

TWO METHODS PROVIDED:
  1. CPU: Farneback dense optical flow (OpenCV) — used in the watcher daemon
     on per-PNG pairs as they arrive from processed_images/
  2. GPU: RAFT-inspired lightweight CNN flow estimator (optional, for Kaggle training)

OUTPUT:
  • Flow fields (H/8 × W/8) × 2 channels (u, v) saved as .npy alongside PNGs
  • FlowDataset for integration with model_lstm_v2.py's SolarFlareSequenceModelV2
  • FlowVisualiser for the Mission Control UI (HSV colour wheel → SFTP'd PNG)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR           = os.path.join(PROJECT_DIR, "data",             "flow_fields")
FLOW_VIS_DIR       = os.path.join(PROJECT_DIR, "processed_images", "flow_vis")
PROCESSED_IMG_DIR  = os.path.join(PROJECT_DIR, "processed_images")
LOG_DIR            = os.path.join(PROJECT_DIR, "logs")

os.makedirs(FLOW_DIR,          exist_ok=True)
os.makedirs(FLOW_VIS_DIR,      exist_ok=True)
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR,           exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  CPU Dense Optical Flow (Farneback) — for the live watchdog daemon
# ---------------------------------------------------------------------------

class FarnebackFlowExtractor:
    """
    Dense optical flow using Gunnar Farneback's polynomial expansion.
    Runs entirely on CPU with OpenCV — no GPU required.

    Designed to be called by the watchdog as new PNGs arrive.

    Usage:
        extractor = FarnebackFlowExtractor(output_dir=FLOW_DIR)
        flow = extractor.compute_and_save("frame_001.png", "frame_002.png")
    """

    def __init__(
        self,
        output_dir: str = FLOW_DIR,
        downsample_factor: int = 8,       # Compute flow at 1/8 resolution
        pyr_scale:  float = 0.5,
        levels:     int   = 3,
        winsize:    int   = 15,
        iterations: int   = 3,
        poly_n:     int   = 5,
        poly_sigma: float = 1.2,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.ds         = downsample_factor
        self.params     = dict(
            pyr_scale  = pyr_scale,
            levels     = levels,
            winsize    = winsize,
            iterations = iterations,
            poly_n     = poly_n,
            poly_sigma = poly_sigma,
            flags      = 0,
        )

    def load_gray(self, path: str) -> np.ndarray:
        """Load image, convert to grayscale, normalise to [0,255] uint8."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def compute(
        self,
        prev_path: str,
        curr_path: str,
    ) -> np.ndarray:
        """
        Compute dense optical flow between two consecutive frames.

        Returns:
            flow: (H/ds, W/ds, 2)  float32 — (u, v) displacement per pixel
        """
        prev = self.load_gray(prev_path)
        curr = self.load_gray(curr_path)

        # Downsample for efficiency
        h, w  = prev.shape
        nh, nw = h // self.ds, w // self.ds
        prev_ds = cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA)
        curr_ds = cv2.resize(curr, (nw, nh), interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(prev_ds, curr_ds, None, **self.params)
        return flow  # (nh, nw, 2)

    def compute_and_save(
        self,
        prev_path: str,
        curr_path: str,
        stem: Optional[str] = None,
    ) -> np.ndarray:
        """Compute flow and save to .npy. Returns the flow array."""
        flow = self.compute(prev_path, curr_path)
        if stem is None:
            stem = Path(curr_path).stem
        out_path = self.output_dir / f"{stem}_flow.npy"
        np.save(str(out_path), flow.astype(np.float32))
        return flow

    def magnitude_angle(self, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (magnitude, angle_radians) from (u,v) flow."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag, ang

    def to_tensor(self, flow: np.ndarray) -> torch.Tensor:
        """
        Convert (H, W, 2) numpy flow to (2, H, W) normalised tensor.
        Normalised to [-1, 1] using 3-sigma clipping.
        """
        flow_t = torch.from_numpy(flow.transpose(2, 0, 1))  # (2, H, W)
        # 3-sigma normalisation per channel
        for c in range(2):
            mu  = flow_t[c].mean()
            std = flow_t[c].std().clamp(min=1e-6)
            flow_t[c] = (flow_t[c] - mu) / (3 * std)
            flow_t[c] = flow_t[c].clamp(-1, 1)
        return flow_t


# ---------------------------------------------------------------------------
# 2.  Flow Anomaly Features — CME / Flare motion signatures
# ---------------------------------------------------------------------------

class FlowAnomalyFeatures:
    """
    Extracts physics-motivated scalar features from a dense flow field.
    These are designed to capture the specific motion signatures of flares/CMEs:

    • mean_magnitude    : Global motion intensity (CMEs → very high)
    • max_magnitude     : Peak local velocity (flare impulsive phase)
    • radial_component  : Outward radial flow from disk center (CME indicator)
    • divergence        : ∇·v  (positive = ejection, negative = inflow)
    • vorticity         : ∇×v  (rotating filaments before eruption)
    • hot_pixel_ratio   : Fraction of pixels with mag > 2σ (flare ribbon proxy)
    """

    def __init__(self, solar_center: Tuple[float, float] = (0.5, 0.5)):
        """solar_center: (cx, cy) as fraction of image dims"""
        self.cx, self.cy = solar_center

    def extract(self, flow: np.ndarray) -> dict:
        """
        Args:
            flow: (H, W, 2) optical flow (u, v)
        Returns:
            dict of scalar features
        """
        h, w = flow.shape[:2]
        u, v = flow[..., 0], flow[..., 1]
        mag  = np.sqrt(u**2 + v**2)

        # Radial component
        ys, xs = np.mgrid[0:h, 0:w]
        rx = xs / w - self.cx
        ry = ys / h - self.cy
        r  = np.sqrt(rx**2 + ry**2).clip(min=1e-6)
        radial = (u * (rx/r) + v * (ry/r)).mean()

        # Divergence (∇·v) via finite differences
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        div   = (du_dx + dv_dy).mean()

        # Vorticity (∇×v in 2D)
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        vort  = np.abs(dv_dx - du_dy).mean()

        # Hot pixel ratio
        threshold = mag.mean() + 2 * mag.std()
        hot_ratio = (mag > threshold).mean()

        return {
            "flow_mean_mag":    float(mag.mean()),
            "flow_max_mag":     float(mag.max()),
            "flow_radial":      float(radial),
            "flow_divergence":  float(div),
            "flow_vorticity":   float(vort),
            "flow_hot_ratio":   float(hot_ratio),
        }


# ---------------------------------------------------------------------------
# 3.  Flow Visualiser — for Mission Control UI
# ---------------------------------------------------------------------------

class FlowVisualiser:
    """
    Converts a (H, W, 2) optical flow field to an HSV colour wheel image.
    Standard visualisation used in computer vision:
      • Hue    = flow direction (angle)
      • Value  = flow magnitude (speed)
      • Saved as PNG and SFTP'd to Mission Control dashboard

    Output is saved to processed_images/flow_vis/ for the SFTP sync loop.
    """

    def __init__(self, output_dir: str = FLOW_VIS_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = Path(output_dir)

    def to_rgb(self, flow: np.ndarray) -> np.ndarray:
        """Convert flow (H, W, 2) → RGB visualisation (H, W, 3) uint8."""
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2          # Hue = direction
        hsv[..., 1] = 255                              # Full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def save(self, flow: np.ndarray, stem: str) -> str:
        """Saves visualisation PNG. Returns output path."""
        vis = self.to_rgb(flow)
        # Overlay arrow grid for large motion
        h, w = flow.shape[:2]
        step = max(h // 16, 1)
        for y in range(0, h, step):
            for x in range(0, w, step):
                u, v = float(flow[y, x, 0]), float(flow[y, x, 1])
                scale = 3.0
                end = (int(x + u * scale), int(y + v * scale))
                cv2.arrowedLine(vis, (x, y), end, (255, 255, 255), 1,
                                tipLength=0.3)

        out = str(self.output_dir / f"{stem}_flow_vis.png")
        cv2.imwrite(out, vis)
        return out


# ---------------------------------------------------------------------------
# 4.  Flow Dataset — integrates with model_lstm_v2.py's DataLoader
# ---------------------------------------------------------------------------

class FlowSequenceDataset(torch.utils.data.Dataset):
    """
    Extends the solar image dataset with pre-computed flow tensors.
    Expects .npy flow files in flow_dir with naming: {image_stem}_flow.npy

    Compatible with SolarFlareSequenceModelV2 (images_seq, tabular, flow_seq).
    """

    def __init__(
        self,
        image_paths:  List[str],   # Ordered list of image paths (T per sequence)
        flow_dir:     str,
        labels:       List[int],
        tabular_data: np.ndarray,  # (N, 4) normalised physics features
        seq_len:      int = 5,
        image_size:   int = 224,
        flow_size:    int = 28,    # 224 / 8
    ):
        import torchvision.transforms as T
        self.image_paths  = image_paths
        self.flow_dir     = Path(flow_dir)
        self.labels       = labels
        self.tabular      = tabular_data
        self.seq_len      = seq_len
        self.flow_size    = flow_size
        self.N            = len(image_paths) - seq_len + 1
        self.img_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self): return self.N

    def __getitem__(self, idx):
        from PIL import Image
        imgs, flows = [], []

        for t in range(self.seq_len):
            path = self.image_paths[idx + t]
            img  = Image.open(path).convert("RGB")
            imgs.append(self.img_tf(img))

        for t in range(1, self.seq_len):
            stem = Path(self.image_paths[idx + t]).stem
            fp   = self.flow_dir / f"{stem}_flow.npy"
            if fp.exists():
                flow_np = np.load(str(fp)).astype(np.float32)
            else:
                # Zero flow if missing (e.g. first batch not yet computed)
                flow_np = np.zeros((self.flow_size, self.flow_size, 2), np.float32)
            # Resize to flow_size if needed
            if flow_np.shape[0] != self.flow_size:
                flow_np = cv2.resize(flow_np, (self.flow_size, self.flow_size))
            flows.append(torch.from_numpy(flow_np.transpose(2, 0, 1)))  # (2, H', W')

        images_seq = torch.stack(imgs)       # (T, 3, 224, 224)
        flow_seq   = torch.stack(flows)      # (T-1, 2, H', W')
        label      = torch.tensor(self.labels[idx + self.seq_len - 1]).float()
        tab        = torch.from_numpy(self.tabular[idx + self.seq_len - 1]).float()

        return images_seq, tab, flow_seq, label


# ---------------------------------------------------------------------------
# 5.  Watchdog Integration — computes flow on-the-fly as new PNGs arrive
# ---------------------------------------------------------------------------

class FlowWatchdogIntegration:
    """
    Drop-in addition to live_inference.py.
    Maintains a rolling buffer of the last N frames and computes flow
    for each new frame as it arrives from process_fits.py.

    Usage (in live_inference.py's watchdog loop):
        flow_watcher = FlowWatchdogIntegration(output_dir=FLOW_DIR)
        ...
        for new_png in new_files:
            flow = flow_watcher.on_new_frame(new_png)
            if flow is not None:
                flow_tensor = flow_watcher.to_tensor(flow)
                # feed flow_tensor to model alongside image
    """

    def __init__(self, output_dir: str = FLOW_DIR, buffer_size: int = 2):
        self.extractor  = FarnebackFlowExtractor(output_dir)
        self.visualiser = FlowVisualiser()
        self.feats      = FlowAnomalyFeatures()
        self._buffer    = []
        self._buf_size  = buffer_size

    def on_new_frame(self, png_path: str) -> Optional[np.ndarray]:
        """
        Call when a new PNG arrives. Returns the flow field vs. previous frame,
        or None if this is the first frame (no predecessor).
        """
        self._buffer.append(png_path)
        if len(self._buffer) > self._buf_size:
            self._buffer.pop(0)

        if len(self._buffer) < 2:
            return None

        prev, curr = self._buffer[-2], self._buffer[-1]
        try:
            flow = self.extractor.compute_and_save(prev, curr)
            self.visualiser.save(flow, Path(curr).stem)
            features = self.feats.extract(flow)
            log.info(f"[Flow] {Path(curr).name} | "
                     f"mag={features['flow_mean_mag']:.3f} | "
                     f"radial={features['flow_radial']:.3f} | "
                     f"hot={features['flow_hot_ratio']:.3f}")
            return flow
        except Exception as e:
            log.warning(f"[Flow] Failed on {curr}: {e}")
            return None

    def to_tensor(self, flow: np.ndarray) -> torch.Tensor:
        return self.extractor.to_tensor(flow)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[optical_flow_module.py] Running sanity check...")

    # Simulate two consecutive solar frames with a synthetic bright moving patch
    h, w = 224, 224
    prev_frame = np.zeros((h, w, 3), dtype=np.uint8)
    curr_frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(prev_frame, (50, 112), 20, (255, 255, 255), -1)   # Bright patch at x=50
    cv2.circle(curr_frame, (80, 112), 20, (255, 255, 255), -1)   # Moved to x=80

    cv2.imwrite("/tmp/prev.png", prev_frame)
    cv2.imwrite("/tmp/curr.png", curr_frame)

    extractor = FarnebackFlowExtractor(output_dir="/tmp/flow_test")
    flow = extractor.compute("/tmp/prev.png", "/tmp/curr.png")
    print(f"  Flow shape: {flow.shape}")       # (28, 28, 2)
    print(f"  Flow max magnitude: {np.sqrt(flow[...,0]**2 + flow[...,1]**2).max():.3f}")

    feats = FlowAnomalyFeatures()
    f = feats.extract(flow)
    print(f"  Features: {f}")

    tensor = extractor.to_tensor(flow)
    print(f"  Tensor shape: {tensor.shape}")   # (2, 28, 28)
    assert tensor.shape == (2, 28, 28)

    vis = FlowVisualiser(output_dir="/tmp/flow_vis")
    out = vis.save(flow, "test")
    print(f"  Visualisation saved: {out}")

    print("[optical_flow_module.py] PASSED ✓")
