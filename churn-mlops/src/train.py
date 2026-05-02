"""
MLflow-tracked training run for the Churn Prediction model.
Trains an XGBoost pipeline and logs params, metrics, and the model to the registry.
"""

import argparse
import os
import json
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(cfg: dict, numeric_features: list, categorical_features: list) -> Pipeline:
    mc = cfg["model"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    classifier = XGBClassifier(
        n_estimators=mc["n_estimators"],
        max_depth=mc["max_depth"],
        learning_rate=mc["learning_rate"],
        subsample=mc["subsample"],
        colsample_bytree=mc["colsample_bytree"],
        scale_pos_weight=mc["scale_pos_weight"],
        random_state=mc["random_state"],
        eval_metric="logloss",
        use_label_encoder=False,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train(config_path: str = "config.yaml", fast_mode: bool = False):
    cfg = load_config(config_path)

    train_df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
    test_df = pd.read_csv(os.path.join("data", "processed", "test.csv"))

    numeric_features = cfg["preprocessing"]["numeric_features"]
    categorical_features = cfg["preprocessing"]["categorical_features"]

    # Add engineered features
    all_features = numeric_features + categorical_features + [
        "charge_per_tenure", "long_term_mth_to_mth"
    ]
    # Filter to only columns that exist (handles fast-mode or subset runs)
    all_features = [f for f in all_features if f in train_df.columns]

    X_train = train_df[all_features]
    y_train = train_df["churn"]
    X_test = test_df[all_features]
    y_test = test_df["churn"]

    # Reduce size for fast CI runs
    if fast_mode:
        X_train = X_train.head(500)
        y_train = y_train.head(500)

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        mc = cfg["model"]
        mlflow.log_params({
            "model_type": mc["type"],
            "n_estimators": mc["n_estimators"],
            "max_depth": mc["max_depth"],
            "learning_rate": mc["learning_rate"],
            "fast_mode": fast_mode,
        })

        num_feats = [f for f in numeric_features + ["charge_per_tenure"] if f in all_features]
        cat_feats = [f for f in categorical_features + ["long_term_mth_to_mth"] if f in all_features]

        pipeline = build_pipeline(cfg, num_feats, cat_feats)
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= cfg["serving"]["churn_threshold"]).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)

        os.makedirs("metrics", exist_ok=True)
        with open("metrics/scores.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        os.makedirs("models", exist_ok=True)

        # Save local pickle for UI / offline use
        import pickle
        with open(os.path.join("models", "churn_model.pkl"), "wb") as fh:
            pickle.dump(pipeline, fh)

        # Log to MLflow registry (optional — requires tracking server)
        try:
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                registered_model_name=cfg["serving"]["model_name"],
                input_example=X_test.head(5),
            )
        except Exception as e:
            print(f"MLflow registry logging skipped: {e}")

        print(f"Training complete - AUC-ROC: {metrics['auc_roc']:.4f}  F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", action="store_true", help="Use a small subset for CI runs")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(config_path=args.config, fast_mode=args.fast_mode)
