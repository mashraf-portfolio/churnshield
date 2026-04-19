import pytest

try:
    from fastapi.testclient import TestClient

    from src.api import app

    client = TestClient(app)
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="API not implemented yet — Phase 3")
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
