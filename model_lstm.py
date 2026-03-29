"""
model_lstm.py
=============
CNN-LSTM Hybrid Architecture for Solar Flare Early Warning System
Mission: ISRO Aditya-L1
Author: Lead ML Engineer

Architecture Overview:
  ┌────────────────────┐
  │  Sequence of Frames │  (B, T, C, H, W)
  └────────┬───────────┘
           │  Per-frame spatial feature extraction
           ▼
  ┌────────────────────┐
  │  CNN Feature       │  EfficientNet-B0 backbone (pretrained)
  │  Extractor         │  Output: (B, T, 1280) feature vectors
  └────────┬───────────┘
           │  Temporal modelling
           ▼
  ┌────────────────────┐
  │  Bi-LSTM           │  2-layer bidirectional LSTM
  │  Temporal Encoder  │  Captures flare build-up dynamics
  └────────┬───────────┘
           │
  ┌────────────────────┐
  │  Tabular Branch    │  4 physics features (EXPTIME, SUN_CX,
  │  (MLP)             │  SUN_CY, R_SUN) via normalization + MLP
  └────────┬───────────┘
           │  Feature fusion (concat + projection)
           ▼
  ┌────────────────────┐
  │  Classifier Head   │  FC → Dropout → FC → Sigmoid
  └────────────────────┘
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# Sub-module 1: CNN Spatial Feature Extractor
# ---------------------------------------------------------------------------

class CNNFeatureExtractor(nn.Module):
    """
    Wraps a pretrained EfficientNet-B0 as a per-frame feature extractor.
    The classifier head is removed; we extract the 1280-dim pooled features.

    Why EfficientNet-B0?
      - Strong inductive bias for spatial patterns (active regions, flare ribbons)
      - Lightweight enough to run on every frame in the sequence
      - ImageNet pretraining gives robust low-level feature detectors
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Remove the final classifier → keep feature layers only
        self.features = base.features          # (B, 1280, 7, 7) at 224px input
        self.pool     = base.avgpool           # (B, 1280, 1, 1)
        self.feat_dim = 1280

        if freeze_backbone:
            # Freeze lower layers; fine-tune only the last 2 MBConv blocks
            layers_to_freeze = list(self.features.children())[:-2]
            for layer in layers_to_freeze:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) single-frame image tensor, normalised to [0,1]
        Returns:
            (B, 1280) spatial feature vector
        """
        x = self.features(x)           # (B, 1280, 7, 7)
        x = self.pool(x)               # (B, 1280, 1, 1)
        return x.flatten(1)            # (B, 1280)


# ---------------------------------------------------------------------------
# Sub-module 2: Tabular MLP Branch
# ---------------------------------------------------------------------------

class TabularMLP(nn.Module):
    """
    Small MLP that projects the 4 physics header features into a
    higher-dimensional space before fusion with the LSTM output.

    Physics features:
      - EXPTIME : Exposure time in seconds  (captures observation cadence)
      - SUN_CX  : Solar disk centre X pixel (tracks pointing accuracy)
      - SUN_CY  : Solar disk centre Y pixel (tracks pointing accuracy)
      - R_SUN   : Solar radius in pixels    (proxy for Sun-spacecraft distance)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4) normalised tabular feature tensor
        Returns:
            (B, output_dim) embedding
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Sub-module 3: Bi-LSTM Temporal Encoder
# ---------------------------------------------------------------------------

class BiLSTMEncoder(nn.Module):
    """
    Two-layer bidirectional LSTM that consumes the per-frame CNN features
    across the time axis and returns the final hidden state.

    Bidirectionality helps capture both the pre-flare magnetic build-up
    (forward pass) and any post-peak context (backward pass).
    """

    def __init__(
        self,
        input_dim:   int = 1280,
        hidden_dim:  int = 512,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,         # expects (B, T, input_dim)
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional doubles the output dim
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1280) sequence of CNN frame features
        Returns:
            (B, hidden_dim * 2) last-step hidden state (forward + backward)
        """
        # lstm_out: (B, T, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        # Use the representation at the final time-step as the sequence summary
        return lstm_out[:, -1, :]      # (B, hidden_dim * 2)


# ---------------------------------------------------------------------------
# Main Model: CNN-LSTM Solar Flare Classifier
# ---------------------------------------------------------------------------

class SolarFlareSequenceModel(nn.Module):
    """
    Full CNN-LSTM model for solar flare binary classification.

    Input contract:
      - images_seq : (B, T, C, H, W)  — sequence of T=5 solar images
      - tabular    : (B, 4)            — physics header features

    Output:
      - logits     : (B, 1)            — raw pre-sigmoid score
      - probs      : (B, 1)            — flare probability [0, 1]
    """

    def __init__(
        self,
        seq_len:          int   = 5,
        tabular_dim:      int   = 4,
        lstm_hidden:      int   = 512,
        lstm_layers:      int   = 2,
        dropout:          float = 0.4,
        freeze_backbone:  bool  = False,
    ):
        super().__init__()
        self.seq_len = seq_len

        # --- Spatial branch ---
        self.cnn      = CNNFeatureExtractor(freeze_backbone=freeze_backbone)
        self.lstm_enc = BiLSTMEncoder(
            input_dim  = self.cnn.feat_dim,
            hidden_dim = lstm_hidden,
            num_layers = lstm_layers,
        )

        # --- Tabular branch ---
        self.tab_mlp = TabularMLP(
            input_dim  = tabular_dim,
            output_dim = 128,
        )

        # --- Fusion & classifier head ---
        fused_dim = self.lstm_enc.output_dim + 128   # 1024 + 128 = 1152
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),   # Binary output — logit
        )

    def forward(
        self,
        images_seq: torch.Tensor,
        tabular:    torch.Tensor,
    ) -> dict:
        """
        Args:
            images_seq : (B, T, C, H, W)  — normalised to [0,1]
            tabular    : (B, 4)            — normalised physics features

        Returns:
            dict with keys:
              'logits' : (B, 1) — raw model output
              'probs'  : (B, 1) — sigmoid-activated probability
        """
        B, T, C, H, W = images_seq.shape
        assert T == self.seq_len, (
            f"Expected sequence length {self.seq_len}, got {T}."
        )

        # --- Encode each frame independently ---
        # Reshape to (B*T, C, H, W) for batched CNN pass
        frames_flat = images_seq.view(B * T, C, H, W)
        cnn_feats   = self.cnn(frames_flat)                   # (B*T, 1280)
        cnn_feats   = cnn_feats.view(B, T, -1)                # (B, T, 1280)

        # --- Temporal encoding ---
        temporal_feat = self.lstm_enc(cnn_feats)              # (B, 1024)

        # --- Tabular encoding ---
        tab_feat = self.tab_mlp(tabular)                      # (B, 128)

        # --- Fusion ---
        fused  = torch.cat([temporal_feat, tab_feat], dim=1)  # (B, 1152)
        logits = self.head(fused)                             # (B, 1)
        probs  = torch.sigmoid(logits)

        return {"logits": logits, "probs": probs}


# ---------------------------------------------------------------------------
# Weight Initialisation Helper
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module) -> None:
    """Apply sensible initialisations to linear and LSTM layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_lstm.py] Running sanity check on: {device}")

    model = SolarFlareSequenceModel(
        seq_len         = 5,
        tabular_dim     = 4,
        lstm_hidden     = 512,
        lstm_layers     = 2,
        dropout         = 0.4,
        freeze_backbone = True,
    ).to(device)

    # Apply custom weight init to non-pretrained layers
    model.lstm_enc.apply(init_weights)
    model.tab_mlp.apply(init_weights)
    model.head.apply(init_weights)

    # Dummy batch: B=4, T=5, C=3, H=224, W=224
    dummy_imgs = torch.randn(4, 5, 3, 224, 224).to(device)
    dummy_tab  = torch.randn(4, 4).to(device)

    with torch.no_grad():
        out = model(dummy_imgs, dummy_tab)

    print(f"  Logits shape : {out['logits'].shape}")   # (4, 1)
    print(f"  Probs  shape : {out['probs'].shape}")    # (4, 1)
    print(f"  Probs sample : {out['probs'].squeeze().cpu().numpy()}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
    print("[model_lstm.py] Sanity check PASSED ✓")
