"""
Tests for the FastAPI prediction endpoint.
Uses TestClient — no real model needed (monkeypatches the model).
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


VALID_CUSTOMER = {
    "tenure": 12,
    "monthly_charges": 70.5,
    "total_charges": 846.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "paperless_billing": "Yes",
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
}


class MockModel:
    class metadata:
        run_id = "mock-run-id-123"

    def predict(self, df):
        return np.array([0.75])  # always returns high-risk churn probability


@pytest.fixture()
def client(monkeypatch):
    """Return a TestClient with the real MLflow model replaced by a mock."""
    import api.main as app_module
    monkeypatch.setattr(app_module, "model", MockModel())

    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_predict_churn(client):
    resp = client.post("/predict", json=VALID_CUSTOMER, params={"customer_id": "cust_001"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["customer_id"] == "cust_001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in ("churn", "stay")
    assert data["risk_tier"] in ("high", "medium", "low")


def test_predict_high_probability_is_churn(client):
    resp = client.post("/predict", json=VALID_CUSTOMER, params={"customer_id": "cust_002"})
    assert resp.status_code == 200
    data = resp.json()
    # MockModel returns 0.75 — above 0.5 threshold → should be "churn" + "high" risk
    assert data["prediction"] == "churn"
    assert data["risk_tier"] == "high"


def test_predict_missing_required_field(client):
    bad_payload = VALID_CUSTOMER.copy()
    del bad_payload["tenure"]
    resp = client.post("/predict", json=bad_payload, params={"customer_id": "cust_003"})
    assert resp.status_code == 422
