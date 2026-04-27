"""FastAPI application for ChurnShield churn prediction."""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.monitoring import append_prediction, read_log
from src.predict import load_artifacts, predict_single
from src.schemas import (
    BatchResponse,
    BatchSummary,
    CustomerInput,
    HealthResponse,
    MetricsSummaryResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
STARTUP_TIME: float = 0.0


def _load_batch_cap() -> int:
    """Read batch_row_cap from config/model_config.yaml. Default 10000."""
    config_path = Path(os.getenv("CONFIG_PATH", "config/model_config.yaml"))
    if not config_path.exists():
        return 10000
    cfg = yaml.safe_load(config_path.read_text())
    return int(cfg.get("batch_row_cap", 10000))


BATCH_ROW_CAP = _load_batch_cap()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, store on app.state."""
    global STARTUP_TIME
    STARTUP_TIME = time.time()

    model_path = Path(os.getenv("MODEL_PATH", "models/churnshield_model.joblib"))
    pre_path = Path(os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib"))
    meta_path = Path(os.getenv("METADATA_PATH", "models/metadata.json"))
    log_path = Path(os.getenv("PREDICTION_LOG_PATH", "data/prediction_log.csv"))

    model, preprocessor, metadata, explainer = load_artifacts(
        model_path, pre_path, meta_path
    )
    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.metadata = metadata
    app.state.explainer = explainer
    app.state.log_path = log_path

    logger.info("ChurnShield API ready")
    yield
    logger.info("ChurnShield API shutting down")


app = FastAPI(
    title="ChurnShield API",
    description="Customer churn prediction with calibrated probabilities and SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    """Liveness/readiness check. Returns 'ok' once the model is loaded."""
    meta = getattr(request.app.state, "metadata", None)
    return HealthResponse(
        status="ok" if meta else "degraded",
        model_loaded=meta is not None,
        model_version=(meta or {}).get("version", "unknown"),
        uptime_seconds=time.time() - STARTUP_TIME if STARTUP_TIME else 0.0,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
async def model_info(request: Request) -> ModelInfoResponse:
    """Static model metadata: name, metrics, threshold, feature names."""
    meta = request.app.state.metadata
    return ModelInfoResponse(
        name=meta.get("model_name", "ChurnShield"),
        underlying_estimator=meta.get("underlying_estimator", "XGBoost"),
        calibration_method=meta.get("calibration_method", "isotonic"),
        version=meta.get("version", "1.0.0"),
        roc_auc=meta["roc_auc"],
        pr_auc=meta.get("pr_auc", 0.0),
        f1=meta.get("f1", 0.0),
        brier_score=meta.get("brier_score", 0.0),
        optimal_threshold=meta["optimal_threshold"],
        n_train=meta.get("n_train", 0),
        feature_names=meta["feature_names"],
        trained_at=meta.get("trained_at", ""),
    )


@app.get("/metrics/summary", response_model=MetricsSummaryResponse, tags=["meta"])
async def metrics_summary(request: Request) -> MetricsSummaryResponse:
    """Aggregate stats over the prediction log (last 30 days)."""
    log_path = request.app.state.log_path
    df = read_log(log_path, days=30)
    if df.empty:
        return MetricsSummaryResponse(
            total_predictions=0,
            churn_rate_last_30d=0.0,
            avg_probability=0.0,
            p95_probability=0.0,
            last_prediction_at=None,
        )
    return MetricsSummaryResponse(
        total_predictions=len(df),
        churn_rate_last_30d=float(df["churn_prediction"].mean()),
        avg_probability=float(df["churn_probability"].mean()),
        p95_probability=float(df["churn_probability"].quantile(0.95)),
        last_prediction_at=df["timestamp"].max().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(customer: CustomerInput, request: Request) -> PredictionResponse:
    """Score a single customer and append the prediction to the log."""
    state = request.app.state
    result = predict_single(
        customer.model_dump(),
        state.model,
        state.preprocessor,
        state.metadata,
        state.explainer,
    )

    log_record = {
        **result,
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "contract": customer.contract,
        "internet_service": customer.internet_service,
    }
    append_prediction(state.log_path, log_record)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
async def predict_batch_endpoint(
    request: Request,
    file: UploadFile = File(..., description="CSV with CustomerInput columns"),
) -> BatchResponse:
    """Score a CSV of customers. Returns predictions + summary stats.
    Enforces batch_row_cap; rows beyond the cap cause a 413.
    """
    import pandas as pd

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if len(df) > BATCH_ROW_CAP:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(df)} exceeds cap of {BATCH_ROW_CAP}. "
            f"Split the file and resubmit.",
        )

    state = request.app.state
    rows_processed = 0
    rows_rejected = 0
    predictions: list = []

    for record in df.to_dict(orient="records"):
        try:
            customer = CustomerInput(**record)
            result = predict_single(
                customer.model_dump(),
                state.model,
                state.preprocessor,
                state.metadata,
                state.explainer,
            )
            log_record = {
                **result,
                "tenure": customer.tenure,
                "monthly_charges": customer.monthly_charges,
                "contract": customer.contract,
                "internet_service": customer.internet_service,
            }
            append_prediction(state.log_path, log_record)
            predictions.append(PredictionResponse(**result))
            rows_processed += 1
        except Exception as exc:
            logger.warning("Skipping row %s: %s", rows_processed + rows_rejected, exc)
            rows_rejected += 1

    if not predictions:
        raise HTTPException(
            status_code=400,
            detail="No valid rows in batch. Check column names and types.",
        )

    churners = sum(1 for p in predictions if p.churn_prediction)
    high_risk = sum(1 for p in predictions if p.risk_band == "high")
    summary = BatchSummary(
        total=len(df),
        churners=churners,
        churn_rate=churners / len(predictions) if predictions else 0.0,
        high_risk=high_risk,
        rows_processed=rows_processed,
        rows_rejected=rows_rejected,
    )
    return BatchResponse(predictions=predictions, summary=summary)
