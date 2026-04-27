from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import preprocess

DATA_DIR = Path("data")
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "Telco_customer_churn.xlsx").exists(),
    reason="IBM Telco data not present — CI skips, run training locally first",
)


@pytest.fixture(scope="module")
def pipeline_output():
    return preprocess(DATA_DIR)


def test_join_produces_expected_row_count(pipeline_output):
    X_tr, X_te, _, _, _, _ = pipeline_output
    total = len(X_tr) + len(X_te)
    assert abs(total - 7043) <= 50


def test_population_file_not_joined(pipeline_output):
    X_tr, X_te, _, _, _, _ = pipeline_output
    pop = pd.read_excel(
        DATA_DIR / "Telco_customer_churn_population.xlsx", engine="openpyxl"
    )
    total = len(X_tr) + len(X_te)
    assert total != len(pop) and abs(total - 7043) <= 50


def test_target_is_binary(pipeline_output):
    _, _, y_tr, _, _, _ = pipeline_output
    assert set(y_tr.unique()) == {0, 1}
    assert y_tr.isna().sum() == 0


def test_train_test_stratified(pipeline_output):
    _, _, y_tr, y_te, _, _ = pipeline_output
    assert abs(y_tr.mean() - y_te.mean()) < 0.015


def test_leakage_columns_dropped(pipeline_output):
    _, _, _, _, _, names = pipeline_output
    leakage = [
        "Churn Reason",
        "Churn Score",
        "CLTV",
        "Customer Status",
        "Churn Category",
    ]
    joined = " ".join(names)
    assert not any(token in joined for token in leakage)


def test_no_nulls_in_processed_features(pipeline_output):
    X_tr, X_te, _, _, _, _ = pipeline_output
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_te).any()
