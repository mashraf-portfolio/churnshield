"""Pure pandas feature-engineering functions for ChurnShield.

No I/O, no sklearn. Every function accepts a DataFrame and returns a new
DataFrame with one additional column appended.
"""

import pandas as pd

_ADD_ON_SERVICES = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

_CONTRACT_RISK = {
    "Month-to-month": 2,
    "One year": 1,
    "Two year": 0,
}

_TENURE_LABELS = ["0-12", "13-24", "25-36", "37-48", "49-60", "61+"]
_TENURE_BINS = [0, 12, 24, 36, 48, 60, float("inf")]


def tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bin 'Tenure Months' into six labelled cohorts → 'tenure_bucket'."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["Tenure Months"],
        bins=_TENURE_BINS,
        labels=_TENURE_LABELS,
        right=True,
        include_lowest=True,
    )
    return df


def contract_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'Contract' to an ordinal risk score (2=high, 1=mid, 0=low) → 'contract_risk'."""
    df = df.copy()
    df["contract_risk"] = df["Contract"].map(_CONTRACT_RISK).astype("Int8")
    return df


def service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count add-on services the customer subscribes to (Yes = 1) → 'service_count'."""
    df = df.copy()
    df["service_count"] = df[_ADD_ON_SERVICES].eq("Yes").sum(axis=1).astype("int8")
    return df


def monthly_charges_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Quartile-bin 'Monthly Charges' into four spend tiers → 'monthly_charges_bin'."""
    df = df.copy()
    df["monthly_charges_bin"] = pd.qcut(
        df["Monthly Charges"],
        q=4,
        labels=["low", "mid-low", "mid-high", "high"],
        duplicates="drop",
    )
    return df


def spend_per_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly spend rate normalised by tenure → 'spend_per_tenure'.

    Tenure is shifted by +1 to avoid division-by-zero for new customers.
    """
    df = df.copy()
    df["spend_per_tenure"] = df["Monthly Charges"] / (df["Tenure Months"] + 1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all five feature transforms in sequence and return enriched DataFrame."""
    df = tenure_bucket(df)
    df = contract_risk(df)
    df = service_count(df)
    df = monthly_charges_bin(df)
    df = spend_per_tenure(df)
    return df
