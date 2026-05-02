"""
Inference logic — load model from MLflow registry and run predictions.
Can be used standalone or imported by the FastAPI app.
"""

import yaml
import pandas as pd
import mlflow.pyfunc


_model = None


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(config_path: str = "config.yaml"):
    global _model
    if _model is None:
        cfg = load_config(config_path)
        model_uri = f"models:/{cfg['serving']['model_name']}/{cfg['serving']['model_stage']}"
        _model = mlflow.pyfunc.load_model(model_uri)
    return _model


def predict(features: dict, config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    model = get_model(config_path)

    df = pd.DataFrame([features])
    probability = float(model.predict(df)[0])

    high = cfg["serving"]["high_risk_threshold"]
    low = cfg["serving"]["low_risk_threshold"]
    threshold = cfg["serving"]["churn_threshold"]

    return {
        "churn_probability": probability,
        "prediction": "churn" if probability >= threshold else "stay",
        "risk_tier": "high" if probability >= high else ("medium" if probability >= low else "low"),
    }
