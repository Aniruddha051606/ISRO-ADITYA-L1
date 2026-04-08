"""
model_lstm_v2.py
================
Upgraded CNN-LSTM with Attention Pooling + Transformer Temporal Encoder
Mission: ISRO Aditya-L1

UPGRADES OVER model_lstm.py
────────────────────────────
  1. Attention Pooling over ALL LSTM timesteps (replaces last-step slice)
     • The model learns WHICH frame in the 5-frame window triggered the flare
     • Attention weights are exported for visualisation in the Mission Control UI

  2. Optional Temporal Transformer branch (replaces or augments Bi-LSTM)
     • Multi-head self-attention captures non-local temporal dependencies
     • Positional encoding preserves frame ordering
     • Trains 2–3× faster than LSTM on GPU due to parallelism

  3. Focal Loss support in training utilities
     • Flare events are rare → severe class imbalance
     • Focal Loss down-weights easy negatives (quiet-sun majority)
     • γ=2 is the standard default; increase to 3 for very imbalanced sets

  4. Optical flow input fusion
     • Accepts an optional (B, T-1, 2, H, W) flow tensor
     • Fused with CNN features via a small projection MLP

Architecture:
  images_seq (B, T, 3, 224, 224)       flow_seq (B, T-1, 2, H/8, W/8)
         │                                       │
   EfficientNet-B0 per frame              FlowEncoder MLP
         │  (B, T, 1280)                         │ (B, T-1, 128)
         └──────────────┬────────────────────────┘
                        │ concat → (B, T', 1280+128)
              ┌─────────▼──────────────────┐
              │  Temporal Encoder          │
              │  [Bi-LSTM + Attention] OR  │
              │  [Transformer Encoder]     │
              └─────────┬──────────────────┘
                        │ (B, 1024+) attended summary
              ┌─────────▼──────────────────┐
              │  Tabular MLP  (B, 128)     │
              └─────────┬──────────────────┘
                        │ Fused (B, ~1280)
              ┌─────────▼──────────────────┐
              │  Classifier Head           │
              │  FC → LN → GELU → Dropout  │
              │  → logit (B, 1)            │
              └────────────────────────────┘
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import math
from typing import Optional, Dict, Tuple

# ---------------------------------------------------------------------------
# Dynamic Directory Setup
# ---------------------------------------------------------------------------
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "lstm_v2")
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  CNN Spatial Feature Extractor (unchanged — EfficientNet-B0)
# ---------------------------------------------------------------------------

class CNNFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.pool     = base.avgpool
        self.feat_dim = 1280
        if freeze_backbone:
            for layer in list(self.features.children())[:-2]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)  # (B, 1280)


# ---------------------------------------------------------------------------
# 2.  Optical Flow Encoder
#     Accepts a (B, 2, H', W') flow field → projects to 128-dim feature
# ---------------------------------------------------------------------------

class FlowEncoder(nn.Module):
    """
    Lightweight CNN that encodes a 2-channel optical flow frame (u, v)
    into a 128-dim feature vector, compatible with the CNN temporal stream.

    Input expected at 1/8 resolution (28×28) for efficiency.
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),   # (B, 32, 14, 14)
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (B, 64,  7,  7)
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                     # (B, 64,  1,  1)
            nn.Flatten(),                                # (B, 64)
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):  # x: (B, 2, H', W')
        return self.net(x)


# ---------------------------------------------------------------------------
# 3.  Positional Encoding for the Transformer branch
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE — adds temporal order awareness to the Transformer."""
    def __init__(self, d_model: int, max_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):   # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# 4a.  Bi-LSTM with Additive Attention Pooling
# ---------------------------------------------------------------------------

class BiLSTMWithAttention(nn.Module):
    """
    Two-layer bidirectional LSTM whose output is pooled by learned attention.

    KEY FIX OVER v1:
    ─────────────────
    v1 used:  lstm_out[:, -1, :]   ← discards most of the backward pass
    v2 uses:  attention-weighted sum over ALL T timesteps

    The attention weights (B, T, 1) are exposed so the Mission Control UI
    can highlight WHICH frame in the window triggered the anomaly.
    """
    def __init__(self, input_dim=1280, hidden_dim=512, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2            # 1024

        # Additive attention scorer (Bahdanau-style)
        self.attn_fc = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),                      # (B, T, 1) score
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            summary: (B, output_dim)    attention-pooled temporal feature
            weights: (B, T)             attention weights for visualisation
        """
        lstm_out, _ = self.lstm(x)                  # (B, T, output_dim)

        # Compute attention weights
        raw_scores  = self.attn_fc(lstm_out)        # (B, T, 1)
        weights     = torch.softmax(raw_scores, dim=1)  # (B, T, 1)

        # Weighted sum over time
        summary     = (lstm_out * weights).sum(dim=1)   # (B, output_dim)
        return summary, weights.squeeze(-1)         # (B,1024), (B,T)


# ---------------------------------------------------------------------------
# 4b.  Temporal Transformer Encoder
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """
    Transformer Encoder operating over the T-frame sequence.

    Advantages over LSTM:
      • Fully parallel — all timesteps processed simultaneously
      • Multi-head attention captures any-to-any frame interactions
        (e.g. pre-flare brightening at t=1 directly attends to eruption at t=5)
      • Trained 2-3× faster on GPU

    Uses a [CLS] token as the sequence summary — standard ViT practice.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=3,
                 dim_ff=1024, dropout=0.1, max_seq_len=16):
        super().__init__()
        self.d_model = d_model

        # Project CNN features to transformer d_model
        self.input_proj = nn.Linear(1280, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len + 1, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model      = d_model,
            nhead        = nhead,
            dim_feedforward = dim_ff,
            dropout      = dropout,
            activation   = "gelu",
            batch_first  = True,
            norm_first   = True,   # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, T, 1280)   CNN frame features
        Returns:
            cls_out: (B, d_model)   CLS token as sequence summary
            None                    (for API compatibility with BiLSTMWithAttention)
        """
        B = x.size(0)
        x = self.input_proj(x)                              # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                    # (B, T+1, d_model)
        x   = self.pos_enc(x)
        out = self.transformer(x)                           # (B, T+1, d_model)
        return out[:, 0], None                              # CLS token


# ---------------------------------------------------------------------------
# 5.  Tabular MLP Branch (physics header features)
# ---------------------------------------------------------------------------

class TabularMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


# ---------------------------------------------------------------------------
# 6.  Main Model: SolarFlareSequenceModelV2
# ---------------------------------------------------------------------------

class SolarFlareSequenceModelV2(nn.Module):
    """
    Upgraded CNN-Temporal Solar Flare Classifier.

    Key changes vs v1:
      • Attention-pooled LSTM output (not last-step slice)
      • Optional Transformer temporal encoder
      • Optional optical flow input fusion
      • Focal Loss utility baked in

    Input contract:
        images_seq  : (B, T, 3, 224, 224)
        tabular     : (B, 4)   — EXPTIME, SUN_CX, SUN_CY, R_SUN  (normalised)
        flow_seq    : (B, T-1, 2, H', W')  — optional optical flow (set None to skip)

    Output:
        dict with 'logits', 'probs', 'attn_weights'
    """

    def __init__(
        self,
        seq_len:          int   = 5,
        tabular_dim:      int   = 4,
        lstm_hidden:      int   = 512,
        lstm_layers:      int   = 2,
        dropout:          float = 0.4,
        freeze_backbone:  bool  = False,
        use_transformer:  bool  = False,    # Switch to Transformer temporal encoder
        use_flow:         bool  = True,     # Enable optical flow fusion
        transformer_dim:  int   = 512,
        transformer_heads:int   = 8,
        transformer_layers:int  = 3,
    ):
        super().__init__()
        self.seq_len       = seq_len
        self.use_transformer = use_transformer
        self.use_flow       = use_flow

        # -- Spatial feature extractor (per frame) --
        self.cnn = CNNFeatureExtractor(freeze_backbone)
        cnn_dim  = self.cnn.feat_dim  # 1280

        # -- Optional flow encoder --
        flow_dim = 0
        if use_flow:
            self.flow_enc = FlowEncoder(out_dim=128)
            flow_dim = 128
        else:
            self.flow_enc = None

        # Temporal input dim: CNN features (+ flow for T-1 frames, averaged)
        temporal_input_dim = cnn_dim + flow_dim  # up to 1408

        # -- Temporal encoder (LSTM w/ attention OR Transformer) --
        if use_transformer:
            self.temporal_enc = TemporalTransformer(
                d_model    = transformer_dim,
                nhead      = transformer_heads,
                num_layers = transformer_layers,
                dropout    = dropout / 2,
            )
            # Project CNN+flow → transformer_dim
            self.temporal_proj = nn.Linear(temporal_input_dim, transformer_dim)
            temporal_out_dim   = transformer_dim
        else:
            self.temporal_enc  = BiLSTMWithAttention(
                input_dim  = temporal_input_dim,
                hidden_dim = lstm_hidden,
                num_layers = lstm_layers,
                dropout    = dropout / 2,
            )
            self.temporal_proj = None
            temporal_out_dim   = self.temporal_enc.output_dim  # 1024

        # -- Tabular branch --
        self.tab_mlp    = TabularMLP(tabular_dim, 64, 128)

        # -- Fusion head --
        fused_dim = temporal_out_dim + 128
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),           nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def _encode_flow(
        self,
        flow_seq: torch.Tensor,     # (B, T-1, 2, H', W')
        B: int, T: int,
    ) -> torch.Tensor:              # returns (B, T, 128) — padded with zeros at t=0
        """Encodes flow frames and zero-pads t=0 (no flow before first frame)."""
        Tm1 = T - 1
        flat  = flow_seq.view(B * Tm1, 2, flow_seq.size(3), flow_seq.size(4))
        feats = self.flow_enc(flat).view(B, Tm1, -1)    # (B, T-1, 128)
        zeros = torch.zeros(B, 1, feats.size(-1), device=feats.device)
        return torch.cat([zeros, feats], dim=1)          # (B, T, 128)

    def forward(
        self,
        images_seq: torch.Tensor,
        tabular:    torch.Tensor,
        flow_seq:   Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B, T, C, H, W = images_seq.shape
        assert T == self.seq_len, f"Expected T={self.seq_len}, got {T}"

        # -- CNN features per frame --
        frames_flat = images_seq.view(B * T, C, H, W)
        cnn_feats   = self.cnn(frames_flat).view(B, T, -1)     # (B, T, 1280)

        # -- Fuse optical flow (if available) --
        if self.use_flow and flow_seq is not None:
            flow_feats = self._encode_flow(flow_seq, B, T)      # (B, T, 128)
            seq_input  = torch.cat([cnn_feats, flow_feats], -1) # (B, T, 1408)
        else:
            seq_input  = cnn_feats                               # (B, T, 1280)

        # -- Temporal projection for Transformer --
        if self.temporal_proj is not None:
            seq_input = self.temporal_proj(seq_input)            # (B, T, d_model)

        # -- Temporal encoding --
        temporal_feat, attn_weights = self.temporal_enc(seq_input)

        # -- Tabular branch --
        tab_feat = self.tab_mlp(tabular)                         # (B, 128)

        # -- Classify --
        fused  = torch.cat([temporal_feat, tab_feat], dim=1)
        logits = self.head(fused)
        probs  = torch.sigmoid(logits)

        return {
            "logits":       logits,
            "probs":        probs,
            "attn_weights": attn_weights,  # (B, T) or None for Transformer
        }


# ---------------------------------------------------------------------------
# 7.  Focal Loss (handles severe class imbalance — flares are rare)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss = -α·(1−p_t)^γ · log(p_t)

    • γ=2 (default): down-weights easy quiet-sun negatives by (1-0.99)^2 = 0.0001
    • α=0.25: balances gradient contribution of the rare positive class
    • Works as drop-in replacement for BCEWithLogitsLoss in train_lstm.py

    Reference: Lin et al. 2017 "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt   = torch.exp(-bce)                       # p_t = probability of correct class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl   = alpha_t * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean": return fl.mean()
        if self.reduction == "sum":  return fl.sum()
        return fl


# ---------------------------------------------------------------------------
# Weight initialisation helper (same as v1)
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:   nn.init.xavier_uniform_(param)
            elif "weight_hh" in name: nn.init.orthogonal_(param)
            elif "bias" in name:      nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_lstm_v2.py] Device: {device}")

    for use_transformer, use_flow in [(False, True), (True, True), (False, False)]:
        tag = f"transformer={use_transformer}, flow={use_flow}"
        model = SolarFlareSequenceModelV2(
            seq_len=5, use_transformer=use_transformer,
            use_flow=use_flow, freeze_backbone=True,
        ).to(device)
        model.apply(init_weights)

        imgs = torch.randn(2, 5, 3, 224, 224).to(device)
        tab  = torch.randn(2, 4).to(device)
        flow = torch.randn(2, 4, 2, 28, 28).to(device) if use_flow else None

        with torch.no_grad():
            out = model(imgs, tab, flow)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [{tag}] logits={out['logits'].shape}, "
              f"attn={out['attn_weights'].shape if out['attn_weights'] is not None else 'N/A'}, "
              f"params={trainable:,}")

    # Focal Loss test
    fl = FocalLoss()
    logits  = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8, 1)).float()
    loss = fl(logits, targets)
    print(f"  Focal loss: {loss.item():.5f}")
    print("[model_lstm_v2.py] PASSED ✓")
