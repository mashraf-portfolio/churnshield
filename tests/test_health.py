from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

MODEL_PATH = Path("models/churnshield_model.joblib")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifacts not present — CI skips, run training locally first",
)


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
