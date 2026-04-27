import json
from pathlib import Path

import joblib
import pytest

DATA_DIR = Path("data")
MODEL_PATH = Path("models/churnshield_model.joblib")
METADATA_PATH = Path("models/metadata.json")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "Telco_customer_churn.xlsx").exists() or not MODEL_PATH.exists(),
    reason="Telco data or trained model missing — CI skips",
)


@pytest.fixture(scope="module")
def metadata():
    return json.loads(METADATA_PATH.read_text())


def test_metadata_has_required_v2_keys(metadata):
    required = {
        "roc_auc",
        "pr_auc",
        "f1",
        "brier_score",
        "optimal_threshold",
        "calibration_method",
        "underlying_estimator",
        "feature_names",
        "version",
        "trained_at",
        "n_train",
    }
    assert required.issubset(metadata.keys())


def test_calibration_method_is_isotonic_or_sigmoid(metadata):
    assert metadata["calibration_method"] in {"isotonic", "sigmoid"}


def test_optimal_threshold_in_reasonable_range(metadata):
    assert 0.10 < metadata["optimal_threshold"] < 0.60


def test_feature_names_match_preprocessor(metadata):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    pp_names = preprocessor.get_feature_names_out().tolist()
    assert len(pp_names) == len(metadata["feature_names"])


def test_brier_score_below_uncalibrated_threshold(metadata):
    assert metadata["brier_score"] < 0.20
