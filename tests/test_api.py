import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import BATCH_ROW_CAP, app

MODEL_PATH = Path("models/churnshield_model.joblib")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifacts not present — CI skips, run training locally first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def high_risk_payload():
    return {
        "customer_id": "TEST-001",
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
        "multiple_lines": "No",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "paperless_billing": "Yes",
        "gender": "Male",
    }


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_has_optimal_threshold(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert "optimal_threshold" in body
    assert 0 < body["optimal_threshold"] < 1


def test_model_info_has_calibration_method(client):
    response = client.get("/model/info")
    assert response.json()["calibration_method"] in {"isotonic", "sigmoid"}


def test_predict_returns_valid_probability(client, high_risk_payload):
    response = client.post("/predict", json=high_risk_payload)
    assert response.status_code == 200
    prob = response.json()["churn_probability"]
    assert 0 <= prob <= 1


def test_predict_shap_values_are_labeled(client, high_risk_payload):
    response = client.post("/predict", json=high_risk_payload)
    shap_keys = response.json()["shap_values"].keys()
    feature_names = set(client.get("/model/info").json()["feature_names"])
    assert all(isinstance(k, str) for k in shap_keys)
    assert not any(re.fullmatch(r"f\d+", k) for k in shap_keys)
    assert all(k in feature_names for k in shap_keys)


def test_predict_high_risk_customer_is_flagged(client, high_risk_payload):
    body = client.post("/predict", json=high_risk_payload).json()
    assert body["risk_band"] in {"medium", "high"}
    assert body["churn_probability"] > 0.5


def test_predict_invalid_input_422(client):
    response = client.post("/predict", json={"tenure": "not a number"})
    assert response.status_code == 422


def test_batch_size_limit_413(client, high_risk_payload):
    df = pd.DataFrame([high_risk_payload] * (BATCH_ROW_CAP + 1))
    csv_bytes = df.to_csv(index=False).encode()
    response = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 413


def test_metrics_summary_endpoint(client):
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    body = response.json()
    required = {
        "total_predictions",
        "churn_rate_last_30d",
        "avg_probability",
        "p95_probability",
        "last_prediction_at",
    }
    assert required.issubset(body.keys())


def test_prediction_is_logged(client, high_risk_payload):
    log_path = app.state.log_path
    before = len(pd.read_csv(log_path)) if Path(log_path).exists() else 0
    client.post("/predict", json=high_risk_payload)
    after = len(pd.read_csv(log_path))
    assert after - before >= 1
