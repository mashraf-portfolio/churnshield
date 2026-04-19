import json
from pathlib import Path

import pytest


def test_model_calibration_metadata():
    metadata_path = Path("models/metadata.json")
    if not metadata_path.exists():
        pytest.skip("No trained model yet")
    metadata = json.loads(metadata_path.read_text())
    assert metadata.get("calibration_method") in {"isotonic", "sigmoid"}
    assert 0.2 <= metadata["optimal_threshold"] <= 0.6
