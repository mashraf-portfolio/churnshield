"""Pydantic v2 request/response schemas for the ChurnShield API."""

from typing import Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Single customer record for churn prediction."""

    customer_id: str | None = Field(
        default=None,
        description="Optional customer identifier (echoed in response, not used by model).",
        examples=["CUST-0001"],
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=72,
        description="Months the customer has been with the company.",
        examples=[3],
    )
    monthly_charges: float = Field(
        ...,
        ge=0,
        description="Current monthly charge in USD.",
        examples=[89.95],
    )
    total_charges: float = Field(
        ...,
        ge=0,
        description="Lifetime total charges in USD.",
        examples=[269.85],
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Contract type. Month-to-month customers churn at the highest rate.",
        examples=["Month-to-month"],
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Internet service tier. Fiber subscribers churn above average.",
        examples=["Fiber optic"],
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="Payment method. Electronic check correlates with higher churn.",
        examples=["Electronic check"],
    )
    senior_citizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if 65+ years old, else 0.",
        examples=[0],
    )
    partner: Literal["Yes", "No"] = Field(
        ...,
        description="Has a partner.",
        examples=["No"],
    )
    dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Has dependents.",
        examples=["No"],
    )
    phone_service: Literal["Yes", "No"] = Field(
        ...,
        description="Has phone service.",
        examples=["Yes"],
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ...,
        description="Enrolled in paperless billing.",
        examples=["Yes"],
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Customer gender. Low predictive power; included for parity with training schema.",
        examples=["Female"],
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ...,
        description="Has multiple phone lines.",
        examples=["No"],
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Subscribed to online security add-on.",
        examples=["No"],
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Subscribed to online backup add-on.",
        examples=["No"],
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Subscribed to device protection add-on.",
        examples=["No"],
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Subscribed to tech support add-on.",
        examples=["No"],
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Subscribed to streaming TV.",
        examples=["Yes"],
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Subscribed to streaming movies.",
        examples=["Yes"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-0001",
                "tenure": 3,
                "monthly_charges": 89.95,
                "total_charges": 269.85,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "senior_citizen": 0,
                "partner": "No",
                "dependents": "No",
                "phone_service": "Yes",
                "paperless_billing": "Yes",
                "gender": "Female",
                "multiple_lines": "No",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction for a single customer with explainability + provenance."""

    customer_id: str | None = Field(
        ...,
        description="Customer identifier echoed from the request (null if not supplied).",
        examples=["CUST-0001"],
    )
    churn_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Calibrated probability of churn in [0, 1].",
        examples=[0.7821],
    )
    churn_prediction: bool = Field(
        ...,
        description="True if churn_probability >= threshold_used, else False.",
        examples=[True],
    )
    risk_band: Literal["low", "medium", "high"] = Field(
        ...,
        description="Discrete risk bucket derived from churn_probability.",
        examples=["high"],
    )
    threshold_used: float = Field(
        ...,
        ge=0,
        le=1,
        description="Decision threshold applied to obtain churn_prediction.",
        examples=[0.2156],
    )
    shap_values: dict[str, float] = Field(
        ...,
        description="Per-feature SHAP contributions (log-odds units). Positive = pushes toward churn.",
        examples=[
            {
                "num__Tenure Months": 0.42,
                "cat__Internet Service_Fiber optic": 0.21,
                "cat__Contract_Two year": 0.18,
                "cat__Online Security_Yes": 0.16,
            }
        ],
    )
    model_version: str = Field(
        ...,
        description="Semantic version of the deployed model artifact.",
        examples=["1.0.0"],
    )
    calibration_method: str = Field(
        ...,
        description="Probability calibration method applied to the underlying estimator.",
        examples=["isotonic"],
    )

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-0001",
                "churn_probability": 0.7821,
                "churn_prediction": True,
                "risk_band": "high",
                "threshold_used": 0.2156,
                "shap_values": {
                    "num__Tenure Months": 0.42,
                    "cat__Internet Service_Fiber optic": 0.21,
                    "cat__Contract_Two year": 0.18,
                    "cat__Contract_One year": 0.14,
                    "cat__Online Security_Yes": 0.16,
                    "cat__Tech Support_Yes": 0.13,
                    "num__Monthly Charges": 0.09,
                    "bin__Paperless Billing": 0.05,
                },
                "model_version": "1.0.0",
                "calibration_method": "isotonic",
            }
        },
    }


class BatchSummary(BaseModel):
    """Aggregate statistics for a batch prediction request."""

    total: int = Field(
        ...,
        description="Total number of customer rows received in the batch.",
        examples=[1000],
    )
    churners: int = Field(
        ...,
        description="Count of rows where churn_prediction is True.",
        examples=[287],
    )
    churn_rate: float = Field(
        ...,
        description="churners / rows_processed.",
        examples=[0.287],
    )
    high_risk: int = Field(
        ...,
        description="Count of rows with risk_band == 'high'.",
        examples=[214],
    )
    rows_processed: int = Field(
        ...,
        description="Rows that passed validation and were scored.",
        examples=[1000],
    )
    rows_rejected: int = Field(
        ...,
        description="Rows rejected due to validation errors.",
        examples=[0],
    )


class BatchResponse(BaseModel):
    """Response for a batch prediction request."""

    predictions: list[PredictionResponse] = Field(
        ...,
        description="One PredictionResponse per successfully scored row.",
        examples=[[]],
    )
    summary: BatchSummary = Field(
        ...,
        description="Aggregate statistics for the batch.",
        examples=[
            {
                "total": 1000,
                "churners": 287,
                "churn_rate": 0.287,
                "high_risk": 214,
                "rows_processed": 1000,
                "rows_rejected": 0,
            }
        ],
    )


class HealthResponse(BaseModel):
    """API liveness and model-readiness probe."""

    status: Literal["ok", "degraded"] = Field(
        ...,
        description="Overall service health. 'degraded' indicates the model failed to load.",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        ...,
        description="True if the joblib artifact loaded successfully at startup.",
        examples=[True],
    )
    model_version: str = Field(
        ...,
        description="Semantic version of the loaded model artifact.",
        examples=["1.0.0"],
    )
    uptime_seconds: float = Field(
        ...,
        description="Seconds since the API process started.",
        examples=[3742.18],
    )

    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Static metadata about the deployed model."""

    name: str = Field(
        ...,
        description="Product name of the model.",
        examples=["ChurnShield"],
    )
    underlying_estimator: str = Field(
        ...,
        description="Algorithm chosen as the calibration base estimator.",
        examples=["XGBoost+Optuna"],
    )
    calibration_method: str = Field(
        ...,
        description="Probability calibration method applied.",
        examples=["isotonic"],
    )
    version: str = Field(
        ...,
        description="Semantic version of the model artifact.",
        examples=["1.0.0"],
    )
    roc_auc: float = Field(
        ...,
        description="Held-out test ROC-AUC of the calibrated model.",
        examples=[0.857],
    )
    pr_auc: float = Field(
        ...,
        description="Held-out test PR-AUC (average precision) of the calibrated model.",
        examples=[0.671],
    )
    f1: float = Field(
        ...,
        description="F1 score at the optimal threshold on the held-out test set.",
        examples=[0.593],
    )
    brier_score: float = Field(
        ...,
        description="Brier score of the calibrated probabilities (lower is better).",
        examples=[0.132],
    )
    optimal_threshold: float = Field(
        ...,
        description="F1-maximizing decision threshold on the held-out test set.",
        examples=[0.2156],
    )
    n_train: int = Field(
        ...,
        description="Number of training rows after the train/test split.",
        examples=[5634],
    )
    feature_names: list[str] = Field(
        ...,
        description="Ordered list of feature names the model expects after preprocessing.",
        examples=[
            [
                "num__Tenure Months",
                "num__Monthly Charges",
                "num__Total Charges",
                "cat__Contract_Two year",
                "cat__Internet Service_Fiber optic",
            ]
        ],
    )
    trained_at: str = Field(
        ...,
        description="ISO 8601 timestamp of when the artifact was produced.",
        examples=["2026-04-27T11:06:08Z"],
    )

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "name": "ChurnShield",
                "underlying_estimator": "XGBoost+Optuna",
                "calibration_method": "isotonic",
                "version": "1.0.0",
                "roc_auc": 0.857,
                "pr_auc": 0.671,
                "f1": 0.593,
                "brier_score": 0.132,
                "optimal_threshold": 0.2156,
                "n_train": 5634,
                "feature_names": [
                    "num__Tenure Months",
                    "num__Monthly Charges",
                    "num__Total Charges",
                    "num__charges_per_month_ratio",
                    "num__service_bundle_count",
                    "num__contract_risk_score",
                    "bin__Senior Citizen",
                    "bin__Paperless Billing",
                    "cat__Contract_One year",
                    "cat__Contract_Two year",
                    "cat__Internet Service_Fiber optic",
                    "cat__Online Security_Yes",
                    "cat__Tech Support_Yes",
                    "cat__Payment Method_Electronic check",
                ],
                "trained_at": "2026-04-27T11:06:08Z",
            }
        },
    }


class MetricsSummaryResponse(BaseModel):
    """Rolling production-monitoring metrics derived from the prediction log."""

    total_predictions: int = Field(
        ...,
        description="Lifetime count of predictions logged by the API.",
        examples=[12_482],
    )
    churn_rate_last_30d: float = Field(
        ...,
        description="Fraction of predictions in the last 30 days flagged as churn.",
        examples=[0.274],
    )
    avg_probability: float = Field(
        ...,
        description="Mean churn_probability across logged predictions.",
        examples=[0.318],
    )
    p95_probability: float = Field(
        ...,
        description="95th percentile of churn_probability across logged predictions.",
        examples=[0.812],
    )
    last_prediction_at: str | None = Field(
        ...,
        description="ISO 8601 timestamp of the most recent logged prediction (null if log is empty).",
        examples=["2026-04-27T11:42:09Z"],
    )
