"""Evaluation plots for ChurnShield. Saves PNGs to models/plots/."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("models/plots")


def _ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float
) -> Path:
    """Confusion matrix at the chosen threshold. Returns saved path."""
    _ensure_plots_dir()
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray) -> Path:
    """ROC curve with AUC annotation."""
    _ensure_plots_dir()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#2563eb")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "roc_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_pr_curve(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float | None = None
) -> Path:
    """Precision-Recall curve with AP annotation. Marks chosen threshold if given."""
    _ensure_plots_dir()
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})", color="#7c3aed")
    if threshold is not None:
        idx = (np.abs(thresholds - threshold)).argmin()
        ax.scatter(
            [recall[idx]],
            [precision[idx]],
            color="red",
            zorder=5,
            label=f"Threshold = {threshold:.3f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "pr_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_calibration_curve(
    y_true: pd.Series,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram comparing uncalibrated vs isotonic-calibrated probabilities."""
    _ensure_plots_dir()
    frac_u, mean_u = calibration_curve(
        y_true, y_proba_uncal, n_bins=n_bins, strategy="quantile"
    )
    frac_c, mean_c = calibration_curve(
        y_true, y_proba_cal, n_bins=n_bins, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mean_u, frac_u, marker="o", color="#dc2626", label="Uncalibrated")
    ax.plot(mean_c, frac_c, marker="s", color="#16a34a", label="Isotonic calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (before vs after)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_k: int = 20,
) -> Path | None:
    """Bar chart of top-k feature importances from the underlying tree model.

    Handles CalibratedClassifierCV by unwrapping. Returns None if the model
    has no feature_importances_ attribute (e.g. LogisticRegression).
    """
    _ensure_plots_dir()
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    if not hasattr(base, "feature_importances_"):
        logger.warning(
            "Model %s has no feature_importances_; skipping plot.", type(base).__name__
        )
        return None
    importances = pd.Series(base.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(top_k)
    fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.3)))
    ax.barh(top.index, top.values, color="#0891b2")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importances")
    fig.tight_layout()
    out = PLOTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
