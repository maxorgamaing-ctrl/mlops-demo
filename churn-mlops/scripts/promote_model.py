"""
Script to promote the latest validated model version to a target stage
in the MLflow Model Registry. Called by the CD pipeline after CI passes.
"""

import argparse
import mlflow
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def promote_model(stage: str, config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    model_name = cfg["serving"]["model_name"]
    client = mlflow.tracking.MlflowClient()

    # Get the latest version in Staging (CI-validated)
    versions = client.get_latest_versions(model_name, stages=["Staging", "None"])
    if not versions:
        raise RuntimeError(f"No model versions found for '{model_name}' in Staging or None.")

    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    print(f"Promoting model '{model_name}' version {latest.version} → {stage}")

    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage=stage,
        archive_existing_versions=True,
    )
    print("Promotion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="Production", choices=["Staging", "Production", "Archived"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    promote_model(stage=args.stage, config_path=args.config)
