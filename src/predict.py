"""Inference and SHAP explanation for ChurnShield."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


FIELD_MAP = {
    "tenure": "Tenure Months",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "contract": "Contract",
    "internet_service": "Internet Service",
    "payment_method": "Payment Method",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "paperless_billing": "Paperless Billing",
    "gender": "Gender",
}


def load_artifacts(
    model_path: Path,
    preprocessor_path: Path,
    metadata_path: Path,
) -> tuple[Any, Any, dict, Any]:
    """Load model, preprocessor, metadata, and a SHAP TreeExplainer."""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = json.loads(metadata_path.read_text())

    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV wraps the real estimator. SHAP needs the
        # underlying tree model, not the calibration wrapper. All folds share
        # the same architecture, so the first fold's estimator is fine.
        base = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base)
        logger.info("SHAP explainer initialized on calibrated base estimator")
    else:
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer initialized on uncalibrated model")

    logger.info(
        "Loaded %s: ROC-AUC=%.4f, threshold=%.4f, features=%d",
        metadata["model_name"],
        metadata["roc_auc"],
        metadata["optimal_threshold"],
        len(metadata["feature_names"]),
    )
    return model, preprocessor, metadata, explainer


def predict_single(
    customer: dict,
    model: Any,
    preprocessor: Any,
    metadata: dict,
    explainer: Any,
) -> dict:
    """Score a single customer. Skeleton: probability only, no SHAP/risk_band yet."""
    from src.features import engineer_features

    customer_id = customer.get("customer_id")
    row = {FIELD_MAP[k]: v for k, v in customer.items() if k in FIELD_MAP}
    df = pd.DataFrame([row])

    # Coerce binary columns to the 'Yes'/'No' string form the
    # OrdinalEncoder was fit on. Pydantic constrains 5 of these
    # already; Senior Citizen is the int 0/1 → 'No'/'Yes' shim.
    bin_cols = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Paperless Billing",
    ]
    for col in bin_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: "Yes"
                if v in (1, True, "Yes")
                else "No"
                if v in (0, False, "No")
                else v
            )

    df = engineer_features(df)
    X = preprocessor.transform(df)
    proba = float(model.predict_proba(X)[0, 1])

    threshold = float(metadata["optimal_threshold"])
    churn_prediction = bool(proba >= threshold)
    if proba < 0.30:
        risk_band = "low"
    elif proba < 0.60:
        risk_band = "medium"
    else:
        risk_band = "high"

    shap_out = explainer.shap_values(X)
    if isinstance(shap_out, list) and len(shap_out) == 2:
        shap_row = shap_out[1][0]
    else:
        arr = np.asarray(shap_out)
        if arr.ndim == 3:
            shap_row = arr[0, :, 1]
        elif arr.ndim == 2:
            shap_row = arr[0]
        else:
            shap_row = arr

    feature_names = metadata["feature_names"]
    if len(feature_names) != len(shap_row):
        raise ValueError(
            f"SHAP/feature_names length mismatch: "
            f"{len(shap_row)} vs {len(feature_names)}"
        )
    pairs = sorted(
        zip(feature_names, [float(v) for v in shap_row]),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    top_k = int(metadata.get("shap_top_k", 10))
    shap_dict = dict(pairs[:top_k])

    return {
        "customer_id": customer_id,
        "churn_probability": round(proba, 4),
        "churn_prediction": churn_prediction,
        "risk_band": risk_band,
        "threshold_used": threshold,
        "shap_values": shap_dict,
        "model_version": metadata.get("version", "1.0.0"),
        "calibration_method": metadata.get("calibration_method", "isotonic"),
    }


def predict_batch(
    df: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    metadata: dict,
    explainer: Any,
) -> list[dict]:
    """Score multiple customers. Returns a list of dicts matching PredictionResponse.

    Currently loops over predict_single. Vectorization is deferred —
    SHAP cost dominates at batch sizes <10k, and per-row SHAP is what
    Streamlit displays anyway.
    """
    results: list[dict] = []
    for record in df.to_dict(orient="records"):
        results.append(predict_single(record, model, preprocessor, metadata, explainer))
    return results
