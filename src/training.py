"""5-model training pipeline for ChurnShield."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.evaluation import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)
from src.preprocessing import preprocess

logger = logging.getLogger(__name__)


def _load_config(
    config_path: Path = Path("config/model_config.yaml"),
) -> dict[str, Any]:
    """Load training config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _train_logistic(X_tr: np.ndarray, y_tr: pd.Series) -> LogisticRegression:
    """Baseline. Logistic Regression with balanced class weights."""
    logger.info("Training LogisticRegression (baseline)...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    return model


def _train_random_forest(X_tr: np.ndarray, y_tr: pd.Series) -> RandomForestClassifier:
    """Mid-tier challenger. Earmarked as A/B challenger for Project 7 (MLOps Backbone)."""
    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def _train_lightgbm(X_tr: np.ndarray, y_tr: pd.Series) -> LGBMClassifier:
    """Speed-optimized GBDT. Leaf-wise growth, fast on tabular."""
    logger.info("Training LGBMClassifier...")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def _xgb_objective(
    trial: optuna.Trial, X_tr: np.ndarray, y_tr: pd.Series, scale_pos_weight: float
) -> float:
    """Optuna objective: maximize 5-fold ROC-AUC for XGBoost."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=1)
    return scores.mean()


def _train_xgboost_optuna(
    X_tr: np.ndarray, y_tr: pd.Series, n_trials: int = 50
) -> tuple[XGBClassifier, dict]:
    """Run Optuna study, retrain best XGBoost on full train set. Returns (model, best_params)."""
    scale_pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    logger.info(
        "Training XGBoost with Optuna (n_trials=%d, scale_pos_weight=%.3f)...",
        n_trials,
        scale_pos_weight,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda t: _xgb_objective(t, X_tr, y_tr, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }
    logger.info(
        "Optuna best CV ROC-AUC: %.4f. Retraining on full train set...",
        study.best_value,
    )
    model = XGBClassifier(**best_params)
    model.fit(X_tr, y_tr)
    return model, best_params


def _train_catboost(X_tr: np.ndarray, y_tr: pd.Series) -> CatBoostClassifier:
    """Modern GBDT with strong defaults and ordered boosting."""
    logger.info("Training CatBoostClassifier...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_tr, y_tr)
    return model


def _evaluate(
    model: Any, X_te: np.ndarray, y_te: pd.Series, name: str
) -> dict[str, float]:
    """Score a model on the held-out test set. Returns metrics dict."""
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_te, y_proba)),
        "pr_auc": float(average_precision_score(y_te, y_proba)),
        "f1_at_05": float(f1_score(y_te, y_pred)),
        "brier": float(brier_score_loss(y_te, y_proba)),
    }
    logger.info(
        "%s: ROC-AUC=%.4f PR-AUC=%.4f F1@0.5=%.4f Brier=%.4f",
        name,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1_at_05"],
        metrics["brier"],
    )
    return metrics


def _calibrate(model: Any, X_tr: np.ndarray, y_tr: pd.Series) -> CalibratedClassifierCV:
    """Wrap winner in CalibratedClassifierCV with isotonic regression, 5-fold CV."""
    logger.info("Calibrating winner with isotonic regression (cv=5)...")
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)
    return calibrated


def _select_threshold(model: Any, X_te: np.ndarray, y_te: pd.Series) -> float:
    """Pick threshold maximizing F1 on the PR curve over the held-out set."""
    y_proba = model.predict_proba(X_te)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_te, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    optimal = float(thresholds[f1[:-1].argmax()])
    logger.info("Optimal threshold (max F1): %.4f", optimal)
    return optimal


def _save_artifacts(
    model: Any,
    underlying_name: str,
    feature_names: list[str],
    metrics: dict[str, float],
    threshold: float,
    best_params: dict | None,
    comparison: list[dict],
    models_dir: Path = Path("models"),
) -> None:
    """Persist model, metadata, and comparison CSV to models/."""
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "churnshield_model.joblib")

    metadata = {
        "model_name": "ChurnShield",
        "underlying_estimator": underlying_name,
        "calibration_method": "isotonic",
        "version": "1.0.0",
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "f1_at_05": metrics["f1_at_05"],
        "brier": metrics["brier"],
        "optimal_threshold": threshold,
        "feature_names": feature_names,
        "optuna_best_params": best_params,
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(comparison).to_csv(models_dir / "comparison.csv", index=False)
    logger.info("Saved model, metadata, and comparison.csv to %s", models_dir)


def main(data_dir: Path) -> None:
    """Train all 5 models, calibrate winner, select threshold, save artifacts."""
    config = _load_config()
    n_trials = config["training"]["optuna_n_trials"]
    logger.info("=" * 60)
    logger.info("ChurnShield training pipeline starting (n_trials=%d)", n_trials)
    logger.info("=" * 60)

    X_tr, X_te, y_tr, y_te, _, feature_names = preprocess(data_dir)

    trainers = {
        "LogisticRegression": lambda: (_train_logistic(X_tr, y_tr), None),
        "RandomForest": lambda: (_train_random_forest(X_tr, y_tr), None),
        "LightGBM": lambda: (_train_lightgbm(X_tr, y_tr), None),
        "XGBoost+Optuna": lambda: _train_xgboost_optuna(X_tr, y_tr, n_trials=n_trials),
        "CatBoost": lambda: (_train_catboost(X_tr, y_tr), None),
    }
    models: dict[str, Any] = {}
    best_params_by_name: dict[str, dict | None] = {}
    comparison: list[dict] = []
    for name, trainer in trainers.items():
        model, params = trainer()
        models[name] = model
        best_params_by_name[name] = params
        comparison.append(_evaluate(model, X_te, y_te, name))

    winner_name = max(comparison, key=lambda m: m["roc_auc"])["model"]
    winner = models[winner_name]
    logger.info("Winner by ROC-AUC: %s", winner_name)

    calibrated = _calibrate(winner, X_tr, y_tr)
    cal_metrics = _evaluate(calibrated, X_te, y_te, f"{winner_name} (calibrated)")
    threshold = _select_threshold(calibrated, X_te, y_te)

    logger.info("Generating plots...")
    y_proba_uncal = winner.predict_proba(X_te)[:, 1]
    y_proba_cal = calibrated.predict_proba(X_te)[:, 1]
    plot_confusion_matrix(y_te, y_proba_cal, threshold)
    plot_roc_curve(y_te, y_proba_cal)
    plot_pr_curve(y_te, y_proba_cal, threshold=threshold)
    plot_calibration_curve(y_te, y_proba_uncal, y_proba_cal)
    plot_feature_importance(calibrated, feature_names)
    logger.info("Plots saved to models/plots/")

    _save_artifacts(
        model=calibrated,
        underlying_name=winner_name,
        feature_names=feature_names,
        metrics=cal_metrics,
        threshold=threshold,
        best_params=best_params_by_name[winner_name],
        comparison=comparison,
    )
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    main(Path(args.data_dir))
