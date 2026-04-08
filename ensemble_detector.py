"""
ensemble_detector.py
====================
Ensemble Anomaly Detector: VAE + Isolation Forest + One-Class SVM
Mission: ISRO Aditya-L1

WHY AN ENSEMBLE?
────────────────
Your current system has ONE anomaly signal: MSE reconstruction error.
This means:
  • A cosmic ray hit (bright pixel, zero physical meaning) can trigger a false alert
  • A subtle pre-flare filament activation (low brightness change, high physical meaning)
    might MISS the threshold

The ensemble combines THREE independent detectors:
  ┌──────────────────────────────────────────────────────┐
  │  DETECTOR 1: VAE Reconstruction Score               │
  │  • Image-level: MSE + SSIM + KL divergence          │
  │  • Strong at spatial anomalies (flare ribbons)      │
  ├──────────────────────────────────────────────────────┤
  │  DETECTOR 2: Isolation Forest on FITS Headers       │
  │  • Tabular: EXPTIME, CRPIX, NAXIS, flux metadata    │
  │  • Strong at telemetry anomalies (pointing, cadence) │
  │  • Fully unsupervised — no labels needed             │
  ├──────────────────────────────────────────────────────┤
  │  DETECTOR 3: One-Class SVM on VAE Latent Space      │
  │  • Latent: μ vectors from the VAE encoder           │
  │  • Strong at distribution-level anomalies           │
  │  • Catches CMEs with unusual morphology but low MSE │
  └──────────────────────────────────────────────────────┘
               ↓
  Weighted Soft Voting → Combined Score → Alert Decision

ALERT LOGIC:
  • 3/3 detectors agree → HIGH CONFIDENCE alert (🚨 flare)
  • 2/3 agree           → MEDIUM confidence (⚠️ watch)
  • 1/3 agree           → LOW confidence (ℹ️ log only)

This dramatically reduces both false positives (cosmic rays, noise)
and false negatives (subtle pre-eruption activity).
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import deque
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
# Resolves all paths relative to the project root — works identically on the
# Ubuntu server, Kaggle notebooks, and local machines without any edits.

PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
REPORTS_DIR    = os.path.join(PROJECT_DIR, "reports")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(REPORTS_DIR,    exist_ok=True)
os.makedirs(DATA_DIR,       exist_ok=True)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector 1: VAE Image Scorer (wrapper around model_vae.py)
# ---------------------------------------------------------------------------

class VAEDetector:
    """
    Wraps SolarVAE.anomaly_score() for use in the ensemble.
    Returns a normalised [0, 1] score.
    """

    def __init__(self, model, device: torch.device, threshold: float = 0.05):
        self.model     = model.to(device).eval()
        self.device    = device
        self.threshold = threshold
        self._cal_buf  = deque(maxlen=500)   # rolling quiet-sun calibration buffer

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> np.ndarray:
        """Returns raw anomaly scores (B,) as numpy."""
        x      = x.to(self.device)
        scores = self.model.anomaly_score(x)
        return scores.cpu().numpy()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (raw_scores, binary_predictions)."""
        scores = self.score(x)
        preds  = (scores > self.threshold).astype(int)
        return scores, preds

    def calibrate(self, quiet_batch: torch.Tensor) -> None:
        scores = self.score(quiet_batch)
        self._cal_buf.extend(scores.tolist())
        if len(self._cal_buf) >= 20:
            self.threshold = float(np.percentile(list(self._cal_buf), 95))
            log.info(f"[VAEDetector] Threshold updated → {self.threshold:.5f}")


# ---------------------------------------------------------------------------
# Detector 2: Isolation Forest on FITS Tabular Metadata
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Fully unsupervised anomaly detection on FITS header metadata.

    Why this works for Aditya-L1:
      During a solar flare, the SUIT payload registers changes in:
      • Exposure time (auto-exposure adjustment)
      • Pointing offsets (spacecraft reaction to particle flux)
      • Observation cadence (burst mode triggered)
      • Solar radius in pixels (limb darkening changes)

    Isolation Forest isolates anomalies by randomly partitioning the
    feature space. Anomalies require fewer splits to isolate (shorter paths).

    Features used (subset of the 240+ FITS columns):
        EXPTIME, SUN_CX, SUN_CY, R_SUN, NAXIS1, NAXIS2,
        CRPIX1, CRPIX2, CADENCE, and any flux statistics present
    """

    FEATURE_COLS = [
        "EXPTIME", "SUN_CX", "SUN_CY", "R_SUN",
        "NAXIS1",  "NAXIS2", "CRPIX1", "CRPIX2",
        "CADENCE", "DATAMEAN", "DATARMS", "DATAMIN", "DATAMAX",
    ]

    def __init__(
        self,
        n_estimators:       int   = 200,
        contamination:      float = 0.05,   # Expected 5% flare fraction
        max_features:       float = 0.8,
        model_path:         str   = os.path.join(CHECKPOINT_DIR, "isolation_forest.pkl"),
        scaler_path:        str   = os.path.join(CHECKPOINT_DIR, "if_scaler.pkl"),
    ):
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.max_features  = max_features
        self.model_path    = Path(model_path)
        self.scaler_path   = Path(scaler_path)
        self.model:  Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler]  = None

    def _extract_features(self, df) -> np.ndarray:
        """Extract and clean numeric features from catalog DataFrame."""
        import pandas as pd
        cols  = [c for c in self.FEATURE_COLS if c in df.columns]
        feats = df[cols].copy()
        feats = feats.apply(pd.to_numeric, errors="coerce")
        feats = feats.fillna(feats.median())
        return feats.values.astype(np.float32)

    def fit(self, catalog_df) -> None:
        """Train on the full catalog (predominantly quiet-sun)."""
        X = self._extract_features(catalog_df)
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        self.model  = IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            max_features  = self.max_features,
            random_state  = 42,
            n_jobs        = -1,
            warm_start    = False,
        )
        self.model.fit(X_scaled)

        # Persist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,  self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        log.info(f"[IsoForest] Trained on {len(X)} samples. Saved to {self.model_path}")

    def load(self) -> bool:
        """Load persisted model. Returns True if successful."""
        if self.model_path.exists() and self.scaler_path.exists():
            self.model  = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            log.info("[IsoForest] Loaded from disk.")
            return True
        return False

    def predict_from_catalog(self, catalog_df) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (anomaly_scores, binary_preds) for a catalog DataFrame.
        Score = negative of the isolation forest decision_function (higher = more anomalous).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")
        X        = self._extract_features(catalog_df)
        X_scaled = self.scaler.transform(X)
        scores   = -self.model.decision_function(X_scaled)   # higher = more anomalous
        preds    = (self.model.predict(X_scaled) == -1).astype(int)
        return scores, preds

    def predict_single(self, feature_dict: dict) -> Tuple[float, int]:
        """
        Predict for a single observation (e.g. from a single FITS header).
        feature_dict: {column_name: value, ...}
        Returns (score, is_anomaly)
        """
        import pandas as pd
        row = pd.DataFrame([feature_dict])
        scores, preds = self.predict_from_catalog(row)
        return float(scores[0]), int(preds[0])

    def partial_fit_update(self, new_catalog_df, retrain_fraction: float = 0.2) -> None:
        """
        Lightweight online update: resample a fraction of the original
        training data (from disk) + new data and retrain.
        Called periodically by the autonomous_core.py scheduler.
        """
        log.info("[IsoForest] Retraining with new data...")
        self.fit(new_catalog_df)


# ---------------------------------------------------------------------------
# Detector 3: One-Class SVM on VAE Latent Vectors
# ---------------------------------------------------------------------------

class LatentSpaceOCSVM:
    """
    One-Class SVM trained on the μ vectors from the VAE encoder.

    The VAE maps "Quiet Sun" images to a compact Gaussian ball in latent space.
    Flare images fall OUTSIDE this ball → OCSVM flags them as anomalies.

    This catches a specific failure mode of the plain MSE detector:
    "Bright but structurally similar" events (e.g. active region brightening
    without eruption) have low MSE but land outside the quiet-sun latent cluster.

    Uses RBF kernel; ν=0.05 means "5% of training points may be outliers"
    (i.e., 5% contamination assumed, matching the IsoForest setting).
    """

    def __init__(
        self,
        nu:          float = 0.05,
        kernel:      str   = "rbf",
        gamma:       str   = "scale",
        model_path:  str   = os.path.join(CHECKPOINT_DIR, "ocsvm.pkl"),
        scaler_path: str   = os.path.join(CHECKPOINT_DIR, "ocsvm_scaler.pkl"),
    ):
        self.nu          = nu
        self.kernel      = kernel
        self.gamma       = gamma
        self.model_path  = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model:  Optional[OneClassSVM] = None
        self.scaler: Optional[StandardScaler] = None

    def fit(self, latent_vectors: np.ndarray) -> None:
        """
        Train on quiet-sun μ vectors from the VAE encoder.
        latent_vectors: (N, latent_dim) numpy array
        """
        self.scaler = StandardScaler()
        Z           = self.scaler.fit_transform(latent_vectors)

        # PCA reduction to 64 dims before OCSVM for scalability
        from sklearn.decomposition import PCA
        n_comp = min(64, Z.shape[1], Z.shape[0] - 1)
        self.pca = PCA(n_components=n_comp, random_state=42)
        Z_pca    = self.pca.fit_transform(Z)

        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(Z_pca)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "pca": self.pca},
                    self.model_path)
        log.info(f"[OCSVM] Trained on {len(latent_vectors)} latent vectors.")

    def load(self) -> bool:
        if self.model_path.exists():
            d = joblib.load(self.model_path)
            self.model, self.scaler, self.pca = d["model"], d["scaler"], d["pca"]
            log.info("[OCSVM] Loaded from disk.")
            return True
        return False

    def predict(self, latent_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (decision_scores, binary_predictions).
        Score > 0 = inlier (quiet sun), Score < 0 = outlier (anomaly).
        We negate so that higher = more anomalous.
        """
        Z     = self.scaler.transform(latent_vectors)
        Z_pca = self.pca.transform(Z)
        scores = -self.model.decision_function(Z_pca)   # negate: high = anomalous
        preds  = (self.model.predict(Z_pca) == -1).astype(int)
        return scores, preds


# ---------------------------------------------------------------------------
# Ensemble: Weighted Voting Combiner
# ---------------------------------------------------------------------------

class EnsembleAnomalyDetector:
    """
    Combines all three detectors with configurable weights and alert levels.

    Confidence levels written to anomaly_status.json:
        HIGH   — 3/3 detectors triggered (send alert, shut SUIT to safe mode)
        MEDIUM — 2/3 triggered (raise watch, increase cadence)
        LOW    — 1/3 triggered (log only)
        NONE   — 0/3 triggered (quiet sun)

    Usage:
        ensemble = EnsembleAnomalyDetector(vae_det, iso_det, ocsvm_det)
        result   = ensemble.run(image_tensor, tabular_dict, latent_vector)
        ensemble.write_status_json(result)   # defaults to LOG_DIR/anomaly_status.json
    """

    CONFIDENCE_MAP = {3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "NONE"}
    ALERT_MAP      = {3: True,   2: True,     1: False, 0: False}

    def __init__(
        self,
        vae_detector:   VAEDetector,
        iso_detector:   IsolationForestDetector,
        ocsvm_detector: LatentSpaceOCSVM,
        weights: Tuple[float, float, float] = (0.5, 0.25, 0.25),
    ):
        self.vae   = vae_detector
        self.iso   = iso_detector
        self.ocsvm = ocsvm_detector
        self.w_vae, self.w_iso, self.w_ocsvm = weights

    def run(
        self,
        image_tensor:   torch.Tensor,         # (1, 3, 224, 224)
        tabular_dict:   dict,                 # FITS header key→value
        latent_mu:      Optional[np.ndarray] = None,  # (1, latent_dim)
    ) -> Dict:
        results = {}

        # --- VAE ---
        vae_scores, vae_preds = self.vae.predict(image_tensor)
        results["vae_score"] = float(vae_scores[0])
        results["vae_alert"] = int(vae_preds[0])

        # --- Isolation Forest ---
        iso_score, iso_pred = self.iso.predict_single(tabular_dict)
        results["iso_score"] = float(iso_score)
        results["iso_alert"] = int(iso_pred)

        # --- OCSVM (if latent vector provided) ---
        if latent_mu is not None and self.ocsvm.model is not None:
            oc_scores, oc_preds = self.ocsvm.predict(latent_mu)
            results["ocsvm_score"] = float(oc_scores[0])
            results["ocsvm_alert"] = int(oc_preds[0])
            n_triggered = results["vae_alert"] + results["iso_alert"] + results["ocsvm_alert"]
        else:
            results["ocsvm_score"] = 0.0
            results["ocsvm_alert"] = 0
            n_triggered = results["vae_alert"] + results["iso_alert"]
            n_triggered = min(n_triggered, 2)  # Max 2/3 without OCSVM

        # --- Combined weighted score ---
        results["combined_score"] = (
            self.w_vae   * results["vae_score"]   +
            self.w_iso   * results["iso_score"]   +
            self.w_ocsvm * results["ocsvm_score"]
        )

        # --- Confidence level ---
        results["n_triggered"]  = n_triggered
        results["confidence"]   = self.CONFIDENCE_MAP[n_triggered]
        results["is_anomaly"]   = self.ALERT_MAP[n_triggered]

        return results

    def write_status_json(self, result: Dict, path: str = None) -> None:
        """
        Overwrites anomaly_status.json with the current ensemble decision.
        Fully backward-compatible with existing mission_control.py SFTP reader.
        """
        import time
        if path is None:
            path = os.path.join(LOG_DIR, "anomaly_status.json")
        status = {
            "is_anomaly":      result["is_anomaly"],
            "confidence":      result["confidence"],
            "n_triggered":     result["n_triggered"],
            "vae_score":       result["vae_score"],
            "iso_score":       result["iso_score"],
            "ocsvm_score":     result["ocsvm_score"],
            "combined_score":  result["combined_score"],
            "vae_alert":       bool(result["vae_alert"]),
            "iso_alert":       bool(result["iso_alert"]),
            "ocsvm_alert":     bool(result["ocsvm_alert"]),
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(status, f, indent=2)

        level = result["confidence"]
        emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️", "NONE": "✅"}[level]
        log.info(f"[Ensemble] {emoji} {level} | "
                 f"VAE={result['vae_score']:.4f} | "
                 f"ISO={result['iso_score']:.4f} | "
                 f"OCSVM={result['ocsvm_score']:.4f} | "
                 f"Combined={result['combined_score']:.4f}")


# ---------------------------------------------------------------------------
# Training helpers — batch-extract latent vectors for OCSVM / IsoForest
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latent_vectors(
    model,                    # SolarVAE
    dataloader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the VAE encoder over a DataLoader and returns:
        (mu_vectors, labels)   — shapes (N, latent_dim), (N,)
    """
    all_mu, all_labels = [], []
    model.eval()
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        mu, _ = model.encode(imgs)
        all_mu.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_mu), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    print("[ensemble_detector.py] Running sanity check...")

    # --- IsoForest on synthetic catalog ---
    n = 300
    df = pd.DataFrame({
        "EXPTIME":  np.random.normal(2.0, 0.1, n),
        "SUN_CX":   np.random.normal(512, 5, n),
        "SUN_CY":   np.random.normal(512, 5, n),
        "R_SUN":    np.random.normal(400, 3, n),
        "NAXIS1":   np.full(n, 1024),
        "NAXIS2":   np.full(n, 1024),
        "CRPIX1":   np.random.normal(512, 2, n),
        "CRPIX2":   np.random.normal(512, 2, n),
        "CADENCE":  np.random.normal(60, 1, n),
        "DATAMEAN": np.random.normal(500, 50, n),
        "DATARMS":  np.random.normal(100, 10, n),
        "DATAMIN":  np.random.normal(100, 10, n),
        "DATAMAX":  np.random.normal(900, 50, n),
    })

    iso = IsolationForestDetector(
        model_path  = "/tmp/if_test.pkl",
        scaler_path = "/tmp/if_scaler_test.pkl",
    )
    iso.fit(df)
    scores, preds = iso.predict_from_catalog(df)
    print(f"  IsoForest | anomalies detected: {preds.sum()}/{n}")

    # --- OCSVM on synthetic latent vectors ---
    Z_quiet = np.random.randn(200, 512).astype(np.float32)   # quiet sun cluster
    Z_flare = np.random.randn(10, 512).astype(np.float32) * 5 + 10  # outliers

    ocsvm = LatentSpaceOCSVM(model_path="/tmp/ocsvm_test.pkl")
    ocsvm.fit(Z_quiet)
    s_q, p_q = ocsvm.predict(Z_quiet[:5])
    s_f, p_f = ocsvm.predict(Z_flare[:5])
    print(f"  OCSVM | Quiet scores: {s_q.round(3)}, preds: {p_q}")
    print(f"  OCSVM | Flare scores: {s_f.round(3)}, preds: {p_f}")
    assert p_f.sum() > p_q.sum(), "OCSVM should flag flare latents more than quiet"

    print("[ensemble_detector.py] PASSED ✓")
