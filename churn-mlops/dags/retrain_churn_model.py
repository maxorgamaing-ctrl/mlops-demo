"""
Airflow DAG — weekly automated churn model retraining.
Flow: check drift → (if drift) retrain → evaluate vs production → promote if better.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import subprocess
import json
import os


def _evaluate_drift(**context) -> str:
    """
    Run full drift check (data drift + concept drift).
    Returns branch name for Airflow:
      - 'retrain'          if either data drift OR concept drift is detected
      - 'skip_retraining'  if model is healthy
    """
    result = subprocess.run(
        ["python", "monitoring/drift_check.py", "--output-json"],
        capture_output=True, text=True
    )
    # Last line of output is the JSON summary (preceded by human-readable prints)
    output_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    json_line = next((l for l in reversed(output_lines) if l.startswith("{")), "{}")
    summary = json.loads(json_line)

    data_drift    = summary.get("dataset_drift", False)
    concept_drift = summary.get("concept_drift", False)
    retrain       = summary.get("retrain_triggered", data_drift or concept_drift)

    print(f"Data drift: {data_drift} | Concept drift: {concept_drift} | Retrain: {retrain}")
    return "retrain" if retrain else "skip_retraining"


def _run_training_pipeline(**context):
    subprocess.run(["python", "src/data_preprocessing.py"], check=True)
    subprocess.run(["python", "src/train.py"], check=True)


def _compare_against_production(**context) -> bool:
    """Compare new model metrics vs current production model."""
    subprocess.run(["python", "src/evaluate.py"], check=True)
    with open("metrics/scores.json") as f:
        new_scores = json.load(f)
    print(f"New model AUC: {new_scores['auc_roc']:.4f}")
    context["task_instance"].xcom_push(key="new_auc", value=new_scores["auc_roc"])


def _promote_to_production_if_better(**context):
    new_auc = context["task_instance"].xcom_pull(key="new_auc", task_ids="evaluate_vs_baseline")
    import yaml, mlflow
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    threshold = cfg["quality_gate"]["min_auc_roc"]
    if new_auc >= threshold:
        subprocess.run(["python", "scripts/promote_model.py", "--stage", "Production"], check=True)
        print(f"Model promoted. AUC: {new_auc:.4f}")
    else:
        print(f"New model did not beat threshold ({new_auc:.4f} < {threshold}). Skipping promotion.")


default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
    "email": ["mlops-alerts@company.com"],
}

with DAG(
    dag_id="churn_model_retraining",
    default_args=default_args,
    description="Weekly churn model drift check and conditional retraining",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "churn"],
) as dag:

    check_drift = BranchPythonOperator(
        task_id="check_data_drift",
        python_callable=_evaluate_drift,
    )

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    retrain = PythonOperator(
        task_id="retrain",
        python_callable=_run_training_pipeline,
    )

    evaluate_new_model = PythonOperator(
        task_id="evaluate_vs_baseline",
        python_callable=_compare_against_production,
    )

    promote_if_better = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_to_production_if_better,
    )

    check_drift >> [retrain, skip_retraining]
    retrain >> evaluate_new_model >> promote_if_better
