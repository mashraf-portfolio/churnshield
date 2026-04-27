import numpy as np
import pandas as pd

from src.features import (
    charges_per_month_ratio,
    contract_risk_score,
    service_bundle_count,
    tenure_bucket,
)


def test_tenure_bucket_new():
    df = pd.DataFrame({"Tenure Months": [5]})
    result = tenure_bucket(df)
    assert result["tenure_bucket"].iloc[0] == "new"


def test_tenure_bucket_loyal():
    df = pd.DataFrame({"Tenure Months": [60]})
    result = tenure_bucket(df)
    assert result["tenure_bucket"].iloc[0] == "loyal"


def test_tenure_bucket_zero_is_new():
    df = pd.DataFrame({"Tenure Months": [0]})
    result = tenure_bucket(df)
    assert result["tenure_bucket"].iloc[0] == "new"


def test_charges_ratio_no_division_error():
    df = pd.DataFrame({"Tenure Months": [0], "Total Charges": [50.0]})
    result = charges_per_month_ratio(df)
    assert np.isfinite(result["charges_per_month_ratio"].iloc[0])


def test_contract_risk_scores():
    df = pd.DataFrame({"Contract": ["Month-to-month", "One year", "Two year"]})
    result = contract_risk_score(df)
    assert result["contract_risk_score"].tolist() == [2, 1, 0]


def test_service_bundle_count_zero_when_all_no():
    services = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    df = pd.DataFrame({s: ["No"] for s in services})
    result = service_bundle_count(df)
    assert result["service_bundle_count"].iloc[0] == 0


def test_service_bundle_count_max_when_all_yes():
    services = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    df = pd.DataFrame({s: ["Yes"] for s in services})
    result = service_bundle_count(df)
    assert result["service_bundle_count"].iloc[0] == 6
