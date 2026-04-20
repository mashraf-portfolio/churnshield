"""Verify the 6 IBM Telco churn xlsx files exist in data/ and report row counts."""

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPECTED_FILES = [
    "Telco_customer_churn.xlsx",
    "Telco_customer_churn_demographics.xlsx",
    "Telco_customer_churn_location.xlsx",
    "Telco_customer_churn_population.xlsx",
    "Telco_customer_churn_services.xlsx",
    "Telco_customer_churn_status.xlsx",
]

DATA_DIR = Path(__file__).parent


def verify() -> bool:
    print(f"Verifying IBM Telco churn files in: {DATA_DIR}\n")
    all_ok = True

    for filename in EXPECTED_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  MISSING  {filename}")
            all_ok = False
            continue
        try:
            df = pd.read_excel(path)
            print(
                f"  OK       {filename:55s}  rows={len(df):,}  cols={len(df.columns)}"
            )
        except Exception as exc:
            print(f"  ERROR    {filename}: {exc}")
            all_ok = False

    print()
    if all_ok:
        print("All 6 files present and readable.")
    else:
        print("One or more files are missing or unreadable.")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify IBM Telco churn xlsx files.")
    parser.add_argument(
        "--verify", action="store_true", help="Check files and print row counts."
    )
    args = parser.parse_args()

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
