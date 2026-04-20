"""Feature engineering for ChurnShield. Called before ColumnTransformer."""

import pandas as pd


def tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bin Tenure Months into 4 cohorts. Bins start at -1 to include tenure=0."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["new", "growing", "established", "loyal"],
    )
    return df


def charges_per_month_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Total Charges divided by tenure+1. Measures spending consistency."""
    df = df.copy()
    df["charges_per_month_ratio"] = df["Total Charges"] / (df["Tenure Months"] + 1)
    return df


def contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal risk score: Month-to-month=2, One year=1, Two year=0."""
    df = df.copy()
    mapping = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk_score"] = df["Contract"].map(mapping)
    return df


def service_bundle_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count of active add-on services (0-6)."""
    df = df.copy()
    services = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    df["service_bundle_count"] = (df[services] == "Yes").sum(axis=1)
    return df


def high_value_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if Total Charges above median, else 0."""
    df = df.copy()
    df["high_value_flag"] = (df["Total Charges"] > df["Total Charges"].median()).astype(
        int
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 5 feature engineering functions in order. Returns a copy."""
    df = df.copy()
    df = tenure_bucket(df)
    df = charges_per_month_ratio(df)
    df = contract_risk_score(df)
    df = service_bundle_count(df)
    df = high_value_flag(df)
    return df
