"""Data pipeline for ChurnShield. raw_df -> engineer_features -> preprocessor -> model."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features import engineer_features

logger = logging.getLogger(__name__)


def _load_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load 5 xlsx files. Population is excluded by design."""
    files = {
        "main": "Telco_customer_churn.xlsx",
        "demographics": "Telco_customer_churn_demographics.xlsx",
        "location": "Telco_customer_churn_location.xlsx",
        "services": "Telco_customer_churn_services.xlsx",
        "status": "Telco_customer_churn_status.xlsx",
    }
    frames = {}
    for key, name in files.items():
        frames[key] = pd.read_excel(data_dir / name, engine="openpyxl")
        logger.info("Loaded %s: %d rows", name, len(frames[key]))
    frames["main"] = frames["main"].rename(columns={"CustomerID": "Customer ID"})
    main_cols = set(frames["main"].columns) - {"Customer ID"}
    for key in ["demographics", "location", "services", "status"]:
        overlapping = [c for c in frames[key].columns if c in main_cols]
        if overlapping:
            frames[key] = frames[key].drop(columns=overlapping)
            logger.info("Dropped overlapping cols from %s: %s", key, overlapping)
    pop = data_dir / "Telco_customer_churn_population.xlsx"
    if pop.exists():
        logger.warning("Skipping %s by design", pop.name)
    return frames


def _join_on_customerid(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join the 4 auxiliary frames onto main on CustomerID."""
    df = frames["main"]
    logger.info("Join start: main has %d rows", len(df))
    for key in ["demographics", "location", "services", "status"]:
        before = len(df)
        df = df.merge(frames[key], on="Customer ID", how="left")
        if len(df) != before:
            logger.warning(
                "Row count changed joining %s: %d -> %d", key, before, len(df)
            )
    return df


def _extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract y from Churn Label, drop leakage columns from X."""
    y = df["Churn Label"].map({"Yes": 1, "No": 0}).astype(int)
    leakage = ["Churn Label", "Churn Reason", "Churn Score", "CLTV", "Churn Value"]
    existing = [c for c in leakage if c in df.columns]
    X = df.drop(columns=existing)
    logger.info("Churn rate: %.3f. Dropped leakage: %s", y.mean(), existing)
    return X, y


def _clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Fix Total Charges blanks, drop id/location cols, call engineer_features."""
    df = df.copy()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(df["Monthly Charges"])
    drop_cols = [
        "Customer ID",
        "Location ID",
        "Service ID",
        "Status ID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
    ]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    df = engineer_features(df)
    logger.info("After cleaning + engineering: %d columns", df.shape[1])
    return df


def _build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer with explicit column groups."""
    numeric_cols = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "charges_per_month_ratio",
        "service_bundle_count",
        "contract_risk_score",
        "high_value_flag",
    ]
    binary_cols = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Phone Service",
        "Paperless Billing",
    ]
    multiclass_cols = [
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Payment Method",
        "tenure_bucket",
    ]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "bin",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                binary_cols,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                multiclass_cols,
            ),
        ],
        remainder="drop",
    )


def preprocess(data_dir: Path):
    """Run full pipeline. Returns (X_train, X_test, y_train, y_test, preprocessor, feature_names)."""
    frames = _load_files(data_dir)
    df = _join_on_customerid(frames)
    X, y = _extract_target(df)
    X = _clean_and_engineer(X)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    preprocessor = _build_preprocessor()
    X_tr = preprocessor.fit_transform(X_tr_raw)
    X_te = preprocessor.transform(X_te_raw)
    names = preprocessor.get_feature_names_out().tolist()
    Path("models").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    logger.info("Train %d, test %d, features %d", len(X_tr), len(X_te), len(names))
    return X_tr, X_te, y_tr, y_te, preprocessor, names
