"""
train_xgboost.py
================
XGBoost Tabular Baseline for Solar Flare Early Warning System
Mission: ISRO Aditya-L1

Strategy:
  - Use only the 4 physics header features (no images)
  - Handle extreme class imbalance with scale_pos_weight
  - Tune hyperparameters with Optuna (optional)
  - Full evaluation: classification report, ROC-AUC, feature importance
  - Export model for ONNX / ensemble use downstream

Physics Features:
  EXPTIME  — Exposure duration (seconds): long exposures correlate with
             increased UV/EUV flux during flares
  SUN_CX   — Solar disc centre X (pixels): pointing drift may correlate
             with SUIT instrument orientation during events
  SUN_CY   — Solar disc centre Y (pixels): same as above
  R_SUN    — Solar radius in pixels: proxy for solar angular diameter;
             varies with ADITYA-L1 orbital position
"""

import os
import logging
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = ["EXPTIME", "SUN_CX", "SUN_CY", "R_SUN"]
TARGET_COL   = "label"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived physics features to improve separability.

    New features:
      - pointing_offset : Euclidean distance of (SUN_CX, SUN_CY) from image
                          centre (112, 112). Large offset may indicate
                          attitude anomaly during flare.
      - exp_per_rsun    : Exposure time normalised by solar radius.
                          Controls for changing apparent solar size.
      - rsun_squared    : Non-linear solar area proxy.
    """
    df = df.copy()

    img_cx = 112.0   # Centre pixel of 224×224 image
    img_cy = 112.0

    df["pointing_offset"] = np.sqrt(
        (df["SUN_CX"] - img_cx) ** 2 + (df["SUN_CY"] - img_cy) ** 2
    )
    df["exp_per_rsun"]  = df["EXPTIME"] / (df["R_SUN"] + 1e-6)
    df["rsun_squared"]  = df["R_SUN"] ** 2

    return df


ENGINEERED_COLS = FEATURE_COLS + ["pointing_offset", "exp_per_rsun", "rsun_squared"]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(catalog_path: str) -> tuple:
    """
    Load and split catalog into train/val/test.
    Uses chronological splitting to prevent temporal leakage.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    """
    df = pd.read_csv(catalog_path).sort_values("filename").reset_index(drop=True)

    # Validate expected columns
    required = FEATURE_COLS + [TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Catalog CSV is missing columns: {missing}")

    log.info(f"Loaded {len(df)} records from {catalog_path}")
    log.info(f"  Class distribution:\n{df[TARGET_COL].value_counts().to_string()}")

    # Feature engineering
    df = engineer_features(df)

    X = df[ENGINEERED_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)

    # Chronological split (70 / 10 / 20)
    n        = len(df)
    val_cut  = int(n * 0.70)
    test_cut = int(n * 0.80)

    X_train, y_train = X[:val_cut],        y[:val_cut]
    X_val,   y_val   = X[val_cut:test_cut],y[val_cut:test_cut]
    X_test,  y_test  = X[test_cut:],       y[test_cut:]

    log.info(
        f"Split | Train: {len(X_train)} (flares: {y_train.sum()}) | "
        f"Val: {len(X_val)} (flares: {y_val.sum()}) | "
        f"Test: {len(X_test)} (flares: {y_test.sum()})"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, ENGINEERED_COLS


# ---------------------------------------------------------------------------
# scale_pos_weight calculation
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    XGBoost's scale_pos_weight is the ratio of negative to positive samples.
    This directly compensates for class imbalance in the gradient update.

    Formula:  scale_pos_weight = count(negative) / count(positive)
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    if n_pos == 0:
        raise ValueError("No positive (flare) samples found in the training set.")

    spw = n_neg / n_pos
    log.info(
        f"Class imbalance | Quiet Sun: {n_neg} | Flares: {n_pos} | "
        f"scale_pos_weight: {spw:.2f}"
    )
    return float(spw)


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

def build_xgb_classifier(scale_pos_weight: float, seed: int = 42) -> xgb.XGBClassifier:
    """
    XGBoost classifier with hyperparameters tuned for imbalanced
    binary classification on small tabular datasets.

    Key parameters:
      scale_pos_weight : Handles class imbalance (replaces SMOTE for XGB)
      eval_metric      : 'aucpr' (Area Under Precision-Recall) is superior
                         to 'auc' for imbalanced datasets because it focuses
                         on the minority class without being dominated by
                         True Negatives.
      subsample        : Row subsampling adds regularisation
      colsample_bytree : Feature subsampling — essential when n_features is small
      max_depth        : Shallow trees prevent overfitting to the few flare events
      reg_alpha        : L1 regularisation for feature selection effect
      reg_lambda       : L2 regularisation for weight shrinkage
    """
    clf = xgb.XGBClassifier(
        # --- Core ---
        n_estimators         = 500,
        learning_rate        = 0.05,
        max_depth            = 6,
        min_child_weight     = 5,      # Higher = more conservative splits
        # --- Imbalance ---
        scale_pos_weight     = scale_pos_weight,
        # --- Regularisation ---
        subsample            = 0.8,
        colsample_bytree     = 0.8,
        reg_alpha            = 0.1,    # L1
        reg_lambda           = 1.0,    # L2
        # --- Training control ---
        early_stopping_rounds= 30,
        eval_metric          = "aucpr",
        use_label_encoder    = False,
        tree_method          = "hist", # Efficient for large datasets
        device               = "cuda" if _gpu_available() else "cpu",
        random_state         = seed,
        verbosity            = 1,
    )
    return clf


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    clf:       xgb.XGBClassifier,
    X:         np.ndarray,
    y:         np.ndarray,
    split_name:str,
    output_dir:str,
    threshold: float = 0.5,
) -> dict:
    """
    Full evaluation suite for a given data split.
    Prints metrics and saves ROC / PR curve plots.
    """
    probs = clf.predict_proba(X)[:, 1]     # P(flare)
    preds = (probs >= threshold).astype(int)

    acc   = accuracy_score(y, preds)
    auc   = roc_auc_score(y, probs)        if len(np.unique(y)) > 1 else 0.5
    ap    = average_precision_score(y, probs) if len(np.unique(y)) > 1 else 0.0
    cm    = confusion_matrix(y, preds)
    cr    = classification_report(
        y, preds, target_names=["Quiet Sun (0)", "Flare (1)"], digits=4
    )

    log.info(f"\n{'='*60}")
    log.info(f"{split_name.upper()} SET EVALUATION")
    log.info(f"{'='*60}")
    log.info(f"Accuracy           : {acc:.4f}")
    log.info(f"ROC-AUC            : {auc:.4f}")
    log.info(f"Avg Precision (AP) : {ap:.4f}  ← best metric for imbalanced data")
    log.info(f"\nClassification Report:\n{cr}")
    log.info(f"Confusion Matrix:\n{cm}")

    # --- Plots ---
    _plot_roc_pr(y, probs, split_name, output_dir)

    return {"accuracy": acc, "roc_auc": auc, "avg_precision": ap}


def _plot_roc_pr(
    y:          np.ndarray,
    probs:      np.ndarray,
    split_name: str,
    output_dir: str,
) -> None:
    """Save ROC and Precision-Recall curve plots."""
    if len(np.unique(y)) < 2:
        log.warning("Only one class present — skipping ROC/PR plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"XGBoost — {split_name.title()} Set", fontsize=14)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc     = roc_auc_score(y, probs)
    axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    axes[1].plot(recall, precision, color="firebrick", lw=2, label=f"AP = {ap:.3f}")
    axes[1].axhline(y=y.mean(), color="k", linestyle="--", lw=1,
                    label=f"Baseline (prevalence = {y.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"xgb_curves_{split_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved curves to: {save_path}")


def plot_feature_importance(
    clf:        xgb.XGBClassifier,
    feature_names: list,
    output_dir: str,
) -> None:
    """Bar chart of XGBoost feature importances (gain-based)."""
    importance = clf.get_booster().get_fscore()   # dict: feat -> score
    if not importance:
        log.warning("No feature importance scores available.")
        return

    # Map f0, f1, ... back to actual names
    feat_map = {f"f{i}": name for i, name in enumerate(feature_names)}
    named_imp = {feat_map.get(k, k): v for k, v in importance.items()}
    df_imp    = pd.DataFrame(
        list(named_imp.items()), columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"], color="steelblue")
    ax.set_xlabel("Gain-based Feature Importance")
    ax.set_title("XGBoost Feature Importance\n(Aditya-L1 Physics Features)")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "xgb_feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Feature importance plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# Optional Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------

def optuna_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """
    Bayesian hyperparameter search using Optuna.
    Optimises for average precision (AP) via 5-fold stratified CV.
    Returns best_params dict.

    Install: pip install optuna
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("Optuna not installed. Skipping hyperparameter tuning.")
        return {}

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 800),
            "learning_rate"   : trial.suggest_float("lr", 0.01, 0.3, log=True),
            "max_depth"       : trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        clf = xgb.XGBClassifier(
            **params,
            scale_pos_weight = scale_pos_weight,
            eval_metric      = "aucpr",
            use_label_encoder= False,
            tree_method      = "hist",
            random_state     = seed,
            verbosity        = 0,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                 scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"[Optuna] Best AP: {study.best_value:.4f}")
    log.info(f"[Optuna] Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost Solar Flare Classifier")
    p.add_argument("--catalog",    default="aditya_l1_catalog.csv", help="Path to catalog CSV")
    p.add_argument("--output_dir", default="checkpoints/xgboost",   help="Checkpoint save dir")
    p.add_argument("--threshold",  type=float, default=0.5,          help="Decision threshold")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--tune",       action="store_true",              help="Run Optuna tuning")
    p.add_argument("--tune_trials",type=int,   default=50,           help="Optuna trials")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 1. Load & split data
    # -----------------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, feat_names = load_data(args.catalog)

    # -----------------------------------------------------------------------
    # 2. Compute scale_pos_weight
    # -----------------------------------------------------------------------
    spw = compute_scale_pos_weight(y_train)

    # -----------------------------------------------------------------------
    # 3. Optional Optuna tuning
    # -----------------------------------------------------------------------
    best_params = {}
    if args.tune:
        log.info(f"Running Optuna hyperparameter search ({args.tune_trials} trials)...")
        best_params = optuna_tune(X_train, y_train, spw, args.tune_trials, args.seed)

    # -----------------------------------------------------------------------
    # 4. Build & train classifier
    # -----------------------------------------------------------------------
    clf = build_xgb_classifier(spw, seed=args.seed)

    # Override with Optuna best params if available
    if best_params:
        clf.set_params(**best_params)

    log.info("Training XGBoost classifier...")
    clf.fit(
        X_train, y_train,
        eval_set         = [(X_val, y_val)],
        verbose          = 50,
    )

    log.info(f"Best iteration: {clf.best_iteration}")

    # -----------------------------------------------------------------------
    # 5. Evaluation
    # -----------------------------------------------------------------------
    val_metrics  = evaluate(clf, X_val,   y_val,   "validation", args.output_dir, args.threshold)
    test_metrics = evaluate(clf, X_test,  y_test,  "test",       args.output_dir, args.threshold)

    # -----------------------------------------------------------------------
    # 6. Feature Importance
    # -----------------------------------------------------------------------
    plot_feature_importance(clf, feat_names, args.output_dir)

    # -----------------------------------------------------------------------
    # 7. 5-Fold Cross-Validation Summary (training data only)
    # -----------------------------------------------------------------------
    log.info("\n--- 5-Fold Stratified CV (training set) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    # Use a fresh clf without early stopping for CV
    clf_cv = build_xgb_classifier(spw, seed=args.seed)
    clf_cv.set_params(early_stopping_rounds=None, n_estimators=clf.best_iteration + 1)
    cv_scores = cross_val_score(
        clf_cv, X_train, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1,
    )
    log.info(f"CV Average Precision: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # -----------------------------------------------------------------------
    # 8. Save model
    # -----------------------------------------------------------------------
    model_path = os.path.join(args.output_dir, "xgb_solar_flare.json")
    clf.save_model(model_path)
    log.info(f"Model saved to: {model_path}")

    # Also save as joblib for sklearn pipeline compatibility
    joblib_path = os.path.join(args.output_dir, "xgb_solar_flare.joblib")
    joblib.dump(clf, joblib_path)
    log.info(f"Joblib model saved to: {joblib_path}")

    # -----------------------------------------------------------------------
    # 9. Summary Table
    # -----------------------------------------------------------------------
    log.info("\n" + "="*60)
    log.info("FINAL RESULTS SUMMARY")
    log.info("="*60)
    log.info(f"{'Metric':<25} {'Val':>10} {'Test':>10}")
    log.info(f"{'-'*50}")
    for metric in ["accuracy", "roc_auc", "avg_precision"]:
        log.info(
            f"  {metric:<23} {val_metrics[metric]:>10.4f} {test_metrics[metric]:>10.4f}"
        )
    log.info("="*60)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
