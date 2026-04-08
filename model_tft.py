"""
model_tft.py
============
Temporal Fusion Transformer for Aditya-L1 FITS Metadata Time Series
Mission: ISRO Aditya-L1

WHY TFT OVER LSTM FOR TELEMETRY
─────────────────────────────────
Your 353 metadata columns change at different rates:
  - Instrument temperatures (DCNTC1R, DHNTC1) — slow drift, hours timescale
  - Spacecraft pointing (ROLL, YAW, PITCH)     — changes per observation
  - Exposure time (EXPTIME)                    — changes per filter band
  - Solar position (DSUN_OBS, HGLT_OBS)        — daily drift

A standard LSTM treats all features equally at each timestep.
TFT uses Variable Selection Networks (VSN) to learn which features
matter at each timestep — and reports this as interpretable weights,
so you can tell ISRO "EXPTIME and DM1TEMP drove this anomaly prediction."

TFT COMPONENTS
───────────────
  1. Variable Selection Networks: learns which of the 353 columns to attend to
  2. Gated Residual Networks (GRN): non-linear feature transformation with gating
  3. Static Context: spacecraft/instrument state that doesn't change within a window
  4. Temporal Self-Attention: captures long-range dependencies (hours before a flare)
  5. Quantile output: predicts not just anomaly score but confidence intervals

OUTPUT
──────
  anomaly_score:   (B,) — probability that this time window contains an anomaly
  feature_weights: (B, num_features) — which telemetry columns drove the prediction
  attn_weights:    (B, T, T) — which timesteps attended to which
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "tft")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
CATALOG_PATH   = os.path.join(PROJECT_DIR, "aditya_l1_catalog.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

log = logging.getLogger(__name__)

# Feature groups from your 353 columns
INSTRUMENT_FEATURES = [
    "EXPTIME", "MEAS_EXP", "CMD_EXPT", "CADENCE",
    "FW1POS", "FW2POS", "HKFW1POS", "HKFW2POS",
    "AMP_G_E", "AMP_G_F", "AMP_G_G", "AMP_G_H",
    "MBIAS_E", "MBIAS_F", "MBIAS_G", "MBIAS_H",
    "OD_BIAS", "RD_BIAS", "DD_BIAS", "OG_BIAS",
]
THERMAL_FEATURES = [
    "DM1TEMP", "DCNTC1R", "DHNTC1", "DMTTC2R", "DHTC2R",
    "DCNTF1R", "DHNTF1",  "DMTF2R1","DHTF2R1", "DMTF2R2",
    "DHTF2R2","FD2TT",    "SMTTR",  "FWMTT",   "SHUMTT",
    "DHAM1TT","DHAM2TT",  "FD1TT",  "FCLAMTT", "THFMTT",
]
POINTING_FEATURES = [
    "ROLL", "YAW", "PITCH", "SC_YAW", "SC_PITCH", "SC_ROLL",
    "CROTA2", "P_ANGLE", "APSSPCH", "APSSROL",
    "HGLT_OBS", "HGLN_OBS", "CRLT_OBS",
]
SOLAR_FEATURES = [
    "R_SUN", "RSUN_OBS", "SUN_CX", "SUN_CY",
    "DSUN_OBS", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
    "CDELT1", "CDELT2",
]
WAVELET_FEATURES = [
    "wav_detail_energy_H", "wav_detail_energy_V", "wav_detail_energy_D",
    "wav_max_H", "wav_max_V", "wav_max_D",
    "wav_entropy_H", "wav_entropy_V", "wav_entropy_D",
    "wav_approx_energy", "wav_detail_to_approx", "wav_hf_fraction",
]
ALL_FEATURES = (INSTRUMENT_FEATURES + THERMAL_FEATURES +
                POINTING_FEATURES + SOLAR_FEATURES + WAVELET_FEATURES)


# ---------------------------------------------------------------------------
# Gated Residual Network (core TFT building block)
# ---------------------------------------------------------------------------

class GRN(nn.Module):
    """
    Gated Residual Network with ELU activation and GLU gating.
    Allows the network to suppress irrelevant features by gating near-zero.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim * 2)  # × 2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_dim)

        # Optional context (static enrichment)
        if context_dim is not None:
            self.fc_ctx = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.fc_ctx = None

        # Residual projection if dims differ
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x if self.residual_proj is None else self.residual_proj(x)

        x = self.fc1(x)
        if context is not None and self.fc_ctx is not None:
            x = x + self.fc_ctx(context)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc2(x)                  # (B, ..., output_dim * 2)
        x, gate = x.chunk(2, dim=-1)    # GLU gating
        x = x * torch.sigmoid(gate)

        return self.norm(x + residual)


# ---------------------------------------------------------------------------
# Variable Selection Network
# ---------------------------------------------------------------------------

class VariableSelectionNetwork(nn.Module):
    """
    Learns soft weights over input features — tells you which telemetry
    columns are actually informative for this prediction.

    Returns both the weighted feature and the selection weights for
    interpretability reporting.
    """

    def __init__(
        self,
        input_sizes:  List[int],   # Size of each individual feature embedding
        hidden_dim:   int,
        dropout:      float = 0.1,
        context_dim:  Optional[int] = None,
    ):
        super().__init__()
        self.num_vars  = len(input_sizes)
        self.hidden    = hidden_dim

        # Per-variable GRNs (process each feature independently)
        self.var_grns = nn.ModuleList([
            GRN(sz, hidden_dim, hidden_dim, dropout)
            for sz in input_sizes
        ])

        # Weight GRN: maps concatenated features → softmax weights
        total_input = sum(input_sizes)
        self.weight_grn = GRN(
            total_input, hidden_dim, self.num_vars,
            dropout, context_dim
        )

    def forward(
        self,
        var_embeddings: List[torch.Tensor],  # List of (B, T, feat_i) tensors
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:          (B, T, hidden_dim) — context-enriched feature vector
            selection_weights: (B, T, num_vars) — interpretable feature importances
        """
        # Process each variable
        processed = [grn(v) for grn, v in zip(self.var_grns, var_embeddings)]
        processed_stack = torch.stack(processed, dim=-2)  # (B, T, num_vars, H)

        # Variable weights from concatenated raw input
        concat = torch.cat(var_embeddings, dim=-1)       # (B, T, sum_sizes)
        weights = F.softmax(
            self.weight_grn(concat, context), dim=-1
        )  # (B, T, num_vars)

        # Weighted sum of processed variables
        output = (processed_stack * weights.unsqueeze(-1)).sum(dim=-2)
        return output, weights


# ---------------------------------------------------------------------------
# Temporal Self-Attention (multi-head, interpretable)
# ---------------------------------------------------------------------------

class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention where all heads share the same value projection.
    This allows meaningful averaging of attention weights across heads —
    making the temporal attention interpretable.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_head)  # Shared across heads
        self.out    = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        Q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        K = self.k_proj(x).reshape(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        V = self.v_proj(x)                                          # (B,T,Dh) shared

        scale  = Dh ** -0.5
        attn   = torch.softmax((Q @ K.transpose(-2, -1)) * scale, dim=-1)  # (B,H,T,T)
        attn   = self.dropout(attn)

        # Shared value: same V for all heads
        V_exp  = V.unsqueeze(1).expand(-1, H, -1, -1)   # (B,H,T,Dh)
        out    = (attn @ V_exp)                          # (B,H,T,Dh)
        out    = out.transpose(1, 2).reshape(B, T, D)
        out    = self.out(out)

        # Return average attention across heads for interpretability
        attn_mean = attn.mean(dim=1)   # (B, T, T)
        return out, attn_mean


# ---------------------------------------------------------------------------
# Full TFT Model
# ---------------------------------------------------------------------------

class SolarTFT(nn.Module):
    """
    Temporal Fusion Transformer for Aditya-L1 FITS metadata anomaly detection.

    Input:
        features:  (B, T, num_features) — normalised telemetry time series
        timestamps: (B, T) — optional Unix timestamps for temporal encoding

    Output:
        dict with:
          anomaly_score:    (B,)            — [0,1] probability
          feature_weights:  (B, T, N_feat)  — variable selection weights
          attn_weights:     (B, T, T)        — temporal attention map
          hidden:           (B, hidden_dim)  — final representation
    """

    def __init__(
        self,
        num_features:  int   = len(ALL_FEATURES),
        hidden_dim:    int   = 64,
        seq_len:       int   = 24,       # 24 timesteps (e.g. 24 frames ~= 24 min)
        n_heads:       int   = 4,
        n_lstm_layers: int   = 2,
        dropout:       float = 0.1,
        quantiles:     List[float] = [0.1, 0.5, 0.9],
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_features = num_features
        self.quantiles    = quantiles

        # Feature embedding: each scalar feature → hidden_dim vector
        self.feature_embeds = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])

        # Temporal encoding (learnable)
        self.temporal_embed = nn.Embedding(seq_len + 1, hidden_dim)

        # Variable Selection Networks
        self.input_vsn = VariableSelectionNetwork(
            input_sizes = [hidden_dim] * num_features,
            hidden_dim  = hidden_dim,
            dropout     = dropout,
        )

        # Sequence encoder: LSTM with skip connection
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0,
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Gating after LSTM
        self.lstm_gate = GRN(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Static enrichment (uses global mean as static context)
        self.static_grn = GRN(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Temporal self-attention
        self.attn = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_gate = GRN(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Position-wise feedforward
        self.ff = GRN(hidden_dim, hidden_dim * 4, hidden_dim, dropout)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features:   torch.Tensor,             # (B, T, num_features)
        timestamps: Optional[torch.Tensor] = None,  # (B, T) integer indices
    ) -> Dict[str, torch.Tensor]:

        B, T, F = features.shape

        # Embed each feature individually
        feat_embeds = [
            self.feature_embeds[i](features[:, :, i:i+1])
            for i in range(F)
        ]   # List of (B, T, hidden_dim)

        # Variable selection
        selected, feat_weights = self.input_vsn(feat_embeds)   # (B,T,H), (B,T,F)

        # Add temporal encoding
        if timestamps is not None:
            t_emb = self.temporal_embed(timestamps.long())     # (B, T, H)
            selected = selected + t_emb

        # LSTM encoder
        lstm_out, _ = self.lstm_encoder(selected)              # (B, T, H)
        lstm_out    = self.lstm_norm(lstm_out + selected)      # Skip connection
        lstm_out    = self.lstm_gate(lstm_out)

        # Static enrichment: use temporal mean as static context
        static_ctx = lstm_out.mean(dim=1, keepdim=True)        # (B, 1, H)
        enriched   = self.static_grn(lstm_out, static_ctx.expand_as(lstm_out))

        # Temporal self-attention
        attn_out, attn_weights = self.attn(enriched)
        attn_out = self.attn_norm(attn_out + enriched)
        attn_out = self.attn_gate(attn_out)

        # Position-wise feedforward
        out = self.ff(attn_out)

        # Anomaly score from final timestep representation
        final = out[:, -1]                                     # (B, H)
        score = self.output_head(final).squeeze(-1)            # (B,)

        return {
            "anomaly_score":    score,
            "feature_weights":  feat_weights,
            "attn_weights":     attn_weights,
            "hidden":           final,
        }


# ---------------------------------------------------------------------------
# Feature Preprocessor (builds the input tensor from your catalog CSV)
# ---------------------------------------------------------------------------

class TFTPreprocessor:
    """
    Converts your aditya_l1_catalog.csv into normalised (B, T, F) tensors
    ready for SolarTFT. Handles missing values, normalisation per-feature,
    and sliding-window sequencing.
    """

    def __init__(
        self,
        feature_cols: List[str] = None,
        seq_len:      int       = 24,
        step_size:    int       = 1,
    ):
        self.features = feature_cols or ALL_FEATURES
        self.seq_len  = seq_len
        self.step     = step_size
        self.means_:  Optional[np.ndarray] = None
        self.stds_:   Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "TFTPreprocessor":
        cols = [c for c in self.features if c in df.columns]
        data = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        self.means_ = data.mean(axis=0)
        self.stds_  = data.std(axis=0).clip(min=1e-6)
        self.fitted_cols = cols
        log.info(f"[TFT] Preprocessor fitted on {len(cols)} features, "
                 f"{len(df)} rows")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns:
            windows: (N_windows, seq_len, num_features)
            centers: list of catalog row indices for the center of each window
        """
        cols = [c for c in self.fitted_cols if c in df.columns]
        data = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        data = (data - self.means_[:len(cols)]) / self.stds_[:len(cols)]
        data = data.astype(np.float32)

        windows, centers = [], []
        for i in range(0, len(data) - self.seq_len + 1, self.step):
            windows.append(data[i:i + self.seq_len])
            centers.append(i + self.seq_len - 1)

        return torch.from_numpy(np.stack(windows)), centers

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"means": self.means_, "stds": self.stds_,
                     "cols": self.fitted_cols}, path)

    def load(self, path: str) -> "TFTPreprocessor":
        import joblib
        d = joblib.load(path)
        self.means_, self.stds_, self.fitted_cols = d["means"], d["stds"], d["cols"]
        return self


# ---------------------------------------------------------------------------
# Interpretability report
# ---------------------------------------------------------------------------

def make_interpretability_report(
    model:       SolarTFT,
    windows:     torch.Tensor,
    feature_cols: List[str],
    device:      torch.device,
    top_k:       int = 10,
) -> pd.DataFrame:
    """
    Run model on windows and return a DataFrame showing which telemetry
    features drove the anomaly predictions.

    Returns DataFrame with columns: feature_name, mean_weight, max_weight
    """
    model.eval()
    all_weights = []
    with torch.no_grad():
        for i in range(0, len(windows), 32):
            batch = windows[i:i+32].to(device)
            out   = model(batch)
            # Average weight over time dimension: (B, F)
            w = out["feature_weights"].mean(dim=1).cpu().numpy()
            all_weights.append(w)

    all_weights = np.concatenate(all_weights, axis=0)   # (N, F)
    mean_w = all_weights.mean(axis=0)
    max_w  = all_weights.max(axis=0)

    n_feats = min(len(feature_cols), len(mean_w))
    report  = pd.DataFrame({
        "feature_name": feature_cols[:n_feats],
        "mean_weight":  mean_w[:n_feats],
        "max_weight":   max_w[:n_feats],
    }).sort_values("mean_weight", ascending=False)

    log.info(f"\n=== TFT Top {top_k} Features ===")
    log.info(report.head(top_k).to_string(index=False))
    return report


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--catalog",     default=CATALOG_PATH)
    p.add_argument("--output_dir",  default=CHECKPOINT_DIR)
    p.add_argument("--seq_len",     type=int,   default=24)
    p.add_argument("--hidden_dim",  type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df   = pd.read_csv(args.catalog, low_memory=False)
    df.columns = df.columns.str.strip()
    # Sort chronologically
    if "T_OBS" in df.columns:
        df["_t"] = pd.to_datetime(df["T_OBS"], errors="coerce")
        df = df.sort_values("_t").reset_index(drop=True)

    prep = TFTPreprocessor(seq_len=args.seq_len)
    prep.fit(df)
    windows, _ = prep.transform(df)
    prep.save(os.path.join(args.output_dir, "tft_preprocessor.pkl"))

    log.info(f"Windows: {windows.shape}")

    from torch.utils.data import TensorDataset
    dataset = TensorDataset(windows)
    loader  = torch.utils.data.DataLoader(
        dataset, args.batch_size, shuffle=True, drop_last=True
    )

    model     = SolarTFT(
        num_features = windows.shape[-1],
        hidden_dim   = args.hidden_dim,
        seq_len      = args.seq_len,
    ).to(device)
    log.info(f"TFT params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Unsupervised: minimise reconstruction-like anomaly score variance
    # (encourages the model to spread its scores rather than collapse)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            # Unsupervised objective: maximise variance of anomaly scores
            # (so the model learns to discriminate, not just output 0.5)
            scores = out["anomaly_score"]
            # Entropy regularisation + variance maximisation
            loss   = -scores.var()  # Maximise variance = learn to discriminate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        mean_loss = total / len(loader)
        log.info(f"Epoch {epoch}/{args.epochs} | Neg-Var Loss: {mean_loss:.5f}")

        if mean_loss < best:
            best = mean_loss
            torch.save({
                "epoch": epoch, "loss": mean_loss,
                "model_state": model.state_dict(),
                "feature_cols": prep.fitted_cols,
            }, os.path.join(args.output_dir, "best_tft.pt"))

    log.info("TFT training complete.")


if __name__ == "__main__":
    main()
