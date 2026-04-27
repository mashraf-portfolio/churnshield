"""Prediction logging and read helpers for live monitoring."""

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

_LOG_LOCK = threading.Lock()

LOG_COLUMNS = [
    "timestamp",
    "customer_id",
    "tenure",
    "monthly_charges",
    "contract",
    "internet_service",
    "churn_probability",
    "churn_prediction",
    "risk_band",
    "model_version",
]


def append_prediction(log_path: Path, record: dict) -> None:
    """Append one prediction to the CSV log. Thread-safe within a single
    process. Creates the file with header if it doesn't exist.

    `record` is the dict returned by predict_single, plus the input
    fields needed for monitoring (tenure, monthly_charges, contract,
    internet_service). Caller is responsible for assembling the merged
    dict — keeps this function dumb.
    """
    row = {col: record.get(col) for col in LOG_COLUMNS}
    if row["timestamp"] is None:
        row["timestamp"] = datetime.now(UTC).isoformat()
    df_row = pd.DataFrame([row], columns=LOG_COLUMNS)
    with _LOG_LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        df_row.to_csv(log_path, mode="a", header=write_header, index=False)


def read_log(log_path: Path, days: int = 30) -> pd.DataFrame:
    """Read recent log entries. Returns empty DataFrame with correct
    columns if the file doesn't exist yet (first-boot graceful path).
    """
    if not log_path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(log_path, parse_dates=["timestamp"])
    if df.empty:
        return df
    cutoff = datetime.now(UTC) - timedelta(days=days)
    # parse_dates may produce tz-naive timestamps; coerce to UTC for compare
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)
