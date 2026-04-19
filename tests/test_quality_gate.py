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


def test_production_model_meets_quality_bar():
    metadata_path = Path("models/metadata.json")
    if not metadata_path.exists():
        pytest.skip("No trained model yet — gate activates after Phase 2")

    import yaml

    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)

    metadata = json.loads(metadata_path.read_text())
    gate = config["quality_gate_roc_auc"]
    assert (
        metadata["roc_auc"] >= gate
    ), f"Model ROC-AUC {metadata['roc_auc']:.3f} below gate {gate}"
