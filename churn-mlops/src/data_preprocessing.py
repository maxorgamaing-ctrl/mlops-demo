"""
Data preprocessing pipeline for Telco Customer Churn dataset.
Reads raw CSV, cleans data, engineers features, and outputs train/test splits.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges is sometimes an empty string — convert and drop nulls
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Normalize column names to snake_case
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Drop customer ID — not a feature
    df.drop(columns=["customerid"], inplace=True, errors="ignore")

    # Encode binary target
    df["churn"] = (df["churn"] == "Yes").astype(int)

    # Encode SeniorCitizen if it's already 0/1 integer — leave as is
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Charge per month normalized by tenure (avoids div/0 for new customers)
    df["charge_per_tenure"] = df["monthlycharges"] / (df["tenure"] + 1)

    # Interaction: long-term customer on month-to-month — high churn risk signal
    df["long_term_mth_to_mth"] = (
        (df["tenure"] > 24) & (df["contract"] == "Month-to-month")
    ).astype(int)

    return df


def preprocess(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    raw_path = os.path.join("data", "raw", "telco_churn.csv")
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["preprocessing"]["test_size"],
        random_state=cfg["preprocessing"]["random_state"],
        stratify=df["churn"],
    )

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Saved {len(train_df)} train rows and {len(test_df)} test rows to {out_dir}/")


if __name__ == "__main__":
    preprocess()
