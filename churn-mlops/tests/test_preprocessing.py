"""
Tests for data_preprocessing.py — validates cleaning and feature engineering logic.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import clean_data, engineer_features


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 0, 36],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [70.0, 20.5, 89.0],
        "TotalCharges": ["840.0", "", "3204.0"],
        "Churn": ["No", "Yes", "No"],
    })


def test_clean_data_drops_empty_total_charges(raw_sample):
    cleaned = clean_data(raw_sample.copy())
    # Row with empty TotalCharges should be dropped
    assert len(cleaned) == 2


def test_clean_data_encodes_target(raw_sample):
    cleaned = clean_data(raw_sample.copy())
    assert set(cleaned["churn"].unique()).issubset({0, 1})


def test_clean_data_drops_customer_id(raw_sample):
    cleaned = clean_data(raw_sample.copy())
    assert "customerid" not in cleaned.columns


def test_engineer_features_charge_per_tenure(raw_sample):
    cleaned = clean_data(raw_sample.copy())
    engineered = engineer_features(cleaned)
    assert "charge_per_tenure" in engineered.columns
    # tenure=0 customer: charge_per_tenure = monthly_charges / 1 (no div/0)
    assert all(engineered["charge_per_tenure"].notna())


def test_engineer_features_long_term_mth_to_mth(raw_sample):
    cleaned = clean_data(raw_sample.copy())
    engineered = engineer_features(cleaned)
    assert "long_term_mth_to_mth" in engineered.columns
    # tenure=12, Month-to-month → not long term → 0
    row = engineered[engineered["tenure"] == 12]
    assert row["long_term_mth_to_mth"].values[0] == 0
