"""
FastAPI app for the Churn Prediction model.
Endpoints: GET /health, POST /predict
"""

import os
import pickle
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerFeatures, PredictionResponse, HealthResponse


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict):
    """Load model: try MLflow registry, fall back to local pickle."""
    # Try local pickle first (always available after training)
    pickle_path = os.path.join("models", "churn_model.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as fh:
            return pickle.load(fh), "local_pickle"
    # Fallback: MLflow registry
    import mlflow.pyfunc
    model_uri = f"models:/{cfg['serving']['model_name']}/{cfg['serving']['model_stage']}"
    m = mlflow.pyfunc.load_model(model_uri)
    return m, "mlflow_registry"


app = FastAPI(title="Churn Prediction API", version="1.0.0")

cfg = load_config()
model, _model_source = _load_model(cfg)


def _engineer(data: dict) -> dict:
    """Replicate the feature engineering from data_preprocessing.py.
    API schema uses snake_case; model was trained with lowercase-no-underscore
    column names for raw features, but keeps underscores in engineered features.
    """
    mc = data.get("monthly_charges", data.get("monthlycharges", 0))
    tenure = data.get("tenure", 0)
    contract = data.get("contract", "")
    # Rename raw snake_case → lowercase no-underscore to match training columns
    rename_map = {
        "monthly_charges": "monthlycharges",
        "total_charges": "totalcharges",
        "internet_service": "internetservice",
        "payment_method": "paymentmethod",
        "phone_service": "phoneservice",
        "multiple_lines": "multiplelines",
        "online_security": "onlinesecurity",
        "online_backup": "onlinebackup",
        "device_protection": "deviceprotection",
        "tech_support": "techsupport",
        "streaming_tv": "streamingtv",
        "streaming_movies": "streamingmovies",
        "paperless_billing": "paperlessbilling",
        "senior_citizen": "seniorcitizen",
    }
    out = {rename_map.get(k, k): v for k, v in data.items()}
    # Add engineered features (names with underscores — matches training CSV)
    out["charge_per_tenure"] = mc / (tenure + 1)
    out["long_term_mth_to_mth"] = int(tenure > 24 and contract == "Month-to-month")
    return out


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_version=_model_source,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, customer_id: str):
    features = customer.model_dump()
    features = _engineer(features)

    try:
        df = pd.DataFrame([features])
        proba_arr = model.predict_proba(df) if hasattr(model, "predict_proba") else model.predict(df)
        if hasattr(proba_arr, "__len__") and len(proba_arr.shape) == 2:
            probability = float(proba_arr[0, 1])
        else:
            probability = float(proba_arr[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    high = cfg["serving"]["high_risk_threshold"]
    low = cfg["serving"]["low_risk_threshold"]
    threshold = cfg["serving"]["churn_threshold"]

    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=probability,
        prediction="churn" if probability >= threshold else "stay",
        risk_tier="high" if probability >= high else ("medium" if probability >= low else "low"),
    )
