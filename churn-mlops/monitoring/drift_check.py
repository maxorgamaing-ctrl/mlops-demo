"""
Drift detection job using scipy statistical tests.
Compares reference (training) data against recent production predictions.

Two types of drift are detected:
  1. DATA DRIFT    — input feature distributions have shifted (KS test / chi-squared)
  2. CONCEPT DRIFT — model performance has degraded on labeled production data
                     (AUC drop > threshold compared to training baseline)

Generates a JSON report and triggers a retrain alert when either drift type is detected.
"""

import argparse
import os
import json
import yaml
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def send_alert(message: str):
    """Log alert. Replace with PagerDuty / Slack webhook in production."""
    print(f"[ALERT] {datetime.now().isoformat()} -- {message}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA DRIFT: input feature distribution shift
# ─────────────────────────────────────────────────────────────────────────────

def check_data_drift(reference_data: pd.DataFrame, production_data: pd.DataFrame, cfg: dict) -> dict:
    """
    DATA DRIFT — detects if the input feature distributions have shifted.
    - Numeric features : two-sample KS test (p < 0.05 = drift)
    - Categorical features : chi-squared test  (p < 0.05 = drift)
    Triggers retrain when >30% of features are drifted.
    """
    numeric_features = cfg["preprocessing"]["numeric_features"]
    categorical_features = cfg["preprocessing"]["categorical_features"]

    drift_results = {}
    n_drifted = 0
    p_threshold = 0.05

    # --- Numeric features: KS test ---
    for feat in numeric_features:
        if feat in reference_data.columns and feat in production_data.columns:
            ref = reference_data[feat].dropna()
            prod = production_data[feat].dropna()
            if len(prod) > 0:
                ks_stat, p_value = stats.ks_2samp(ref, prod)
                drifted = bool(p_value < p_threshold)
                drift_results[feat] = {
                    "test": "ks_2samp",
                    "statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_value), 4),
                    "drifted": drifted,
                }
                if drifted:
                    n_drifted += 1

    # --- Categorical features: chi-squared test ---
    for feat in categorical_features:
        if feat in reference_data.columns and feat in production_data.columns:
            ref_counts = reference_data[feat].value_counts(normalize=True)
            prod_counts = production_data[feat].value_counts(normalize=True)
            all_cats = ref_counts.index.union(prod_counts.index)
            ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats])
            prod_freq = np.array([prod_counts.get(c, 0) for c in all_cats])
            # Scale to counts for chi2
            prod_obs = (prod_freq * len(production_data[feat].dropna())).clip(min=1e-6)
            ref_exp = (ref_freq * len(production_data[feat].dropna())).clip(min=1e-6)
            try:
                chi2, p_value = stats.chisquare(prod_obs, f_exp=ref_exp)
                drifted = bool(p_value < p_threshold)
                drift_results[feat] = {
                    "test": "chi_squared",
                    "statistic": round(float(chi2), 4),
                    "p_value": round(float(p_value), 4),
                    "drifted": drifted,
                }
                if drifted:
                    n_drifted += 1
            except Exception:
                pass

    dataset_drift = bool(n_drifted > (len(drift_results) * 0.3))
    return {
        "features_tested": len(drift_results),
        "features_drifted": n_drifted,
        "dataset_drift": dataset_drift,
        "feature_results": drift_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT DRIFT: model performance degradation on labeled production data
# ─────────────────────────────────────────────────────────────────────────────

def check_concept_drift(production_data: pd.DataFrame, cfg: dict,
                        model_path: str = "models/churn_model.pkl") -> dict:
    """
    CONCEPT DRIFT — detects if the model's predictive performance has degraded.
    Requires production data to have a 'churn' ground-truth label column.
    Compares current AUC against the training baseline from metrics/scores.json.
    Triggers retrain when AUC drops > concept_drift_auc_drop threshold.
    """
    concept_drift = False
    reason = "no_labeled_production_data"
    current_auc = None
    baseline_auc = None
    auc_drop = None

    if "churn" not in production_data.columns:
        return {
            "concept_drift": False,
            "reason": reason,
            "current_auc": None,
            "baseline_auc": None,
            "auc_drop": None,
        }

    # Load baseline metrics from last training run
    metrics_path = "metrics/scores.json"
    if not os.path.exists(metrics_path):
        return {"concept_drift": False, "reason": "no_baseline_metrics", "current_auc": None,
                "baseline_auc": None, "auc_drop": None}

    with open(metrics_path) as f:
        baseline = json.load(f)
    baseline_auc = baseline.get("auc_roc", 0.0)

    # Load model and score production data
    if not os.path.exists(model_path):
        return {"concept_drift": False, "reason": "model_not_found", "current_auc": None,
                "baseline_auc": baseline_auc, "auc_drop": None}

    from sklearn.metrics import roc_auc_score

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Build feature list matching training
    numeric_features = cfg["preprocessing"]["numeric_features"]
    categorical_features = cfg["preprocessing"]["categorical_features"]
    engineered = ["charge_per_tenure", "long_term_mth_to_mth"]
    all_features = [c for c in numeric_features + categorical_features + engineered
                    if c in production_data.columns]

    y_true = production_data["churn"]
    X = production_data[all_features]

    try:
        y_prob = model.predict_proba(X)[:, 1]
        current_auc = float(roc_auc_score(y_true, y_prob))
        auc_drop = round(baseline_auc - current_auc, 4)

        # Threshold from config (default 0.05 = 5% AUC drop triggers retrain)
        drop_threshold = cfg.get("monitoring", {}).get("concept_drift_auc_drop", 0.05)
        concept_drift = bool(auc_drop > drop_threshold)
        reason = (f"AUC dropped {auc_drop:.4f} (baseline={baseline_auc:.4f}, "
                  f"current={current_auc:.4f}, threshold={drop_threshold})")
    except Exception as e:
        reason = f"scoring_error: {e}"

    return {
        "concept_drift": concept_drift,
        "reason": reason,
        "current_auc": round(current_auc, 4) if current_auc is not None else None,
        "baseline_auc": round(baseline_auc, 4),
        "auc_drop": auc_drop,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED REPORT
# ─────────────────────────────────────────────────────────────────────────────

def run_drift_report(reference_data: pd.DataFrame, production_data: pd.DataFrame, cfg: dict):
    """Run both data drift and concept drift checks, save report, fire alerts."""
    print("\n--- DATA DRIFT CHECK ---")
    data_drift_result = check_data_drift(reference_data, production_data, cfg)

    print("\n--- CONCEPT DRIFT CHECK ---")
    concept_drift_result = check_concept_drift(production_data, cfg)

    retrain_triggered = data_drift_result["dataset_drift"] or concept_drift_result["concept_drift"]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "reference_rows": len(reference_data),
        "production_rows": len(production_data),
        # ── Data drift ──────────────────────────────────────────────────────
        "dataset_drift": data_drift_result["dataset_drift"],
        "features_tested": data_drift_result["features_tested"],
        "features_drifted": data_drift_result["features_drifted"],
        "feature_results": data_drift_result["feature_results"],
        # ── Concept drift ────────────────────────────────────────────────────
        "concept_drift": concept_drift_result["concept_drift"],
        "concept_drift_reason": concept_drift_result["reason"],
        "current_auc": concept_drift_result["current_auc"],
        "baseline_auc": concept_drift_result["baseline_auc"],
        "auc_drop": concept_drift_result["auc_drop"],
        # ── Overall retrain decision ─────────────────────────────────────────
        "retrain_triggered": retrain_triggered,
    }

    out_dir = cfg["monitoring"]["report_output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(
        out_dir, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDrift report saved to {report_path}")

    # ── Alerts ───────────────────────────────────────────────────────────────
    if data_drift_result["dataset_drift"]:
        send_alert(
            f"DATA DRIFT: {data_drift_result['features_drifted']}/"
            f"{data_drift_result['features_tested']} features shifted -- retrain triggered"
        )
    if concept_drift_result["concept_drift"]:
        send_alert(f"CONCEPT DRIFT: {concept_drift_result['reason']} -- retrain triggered")

    if not retrain_triggered:
        print("No drift detected -- model is healthy, no retrain needed")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", action="store_true",
                        help="Print summary JSON to stdout (used by Airflow DAG)")
    args = parser.parse_args()

    cfg = load_config()
    reference_df = pd.read_csv(os.path.join("data", "processed", "train.csv"))

    prod_path = os.path.join("data", "production", "recent_predictions.csv")
    if not os.path.exists(prod_path):
        print("No production data found -- running demo drift check with simulated data")
        rng = np.random.default_rng(42)
        production_df = reference_df.sample(300, random_state=42).copy()
        # Simulate distribution shift on numeric features
        for col in cfg["preprocessing"]["numeric_features"]:
            if col in production_df.columns:
                production_df[col] = production_df[col] * rng.uniform(0.9, 1.2, len(production_df))
    else:
        production_df = pd.read_csv(prod_path)

    summary = run_drift_report(reference_df, production_df, cfg)

    print(f"\n{'='*50}")
    print(f"DATA DRIFT    : {summary['dataset_drift']} "
          f"({summary['features_drifted']}/{summary['features_tested']} features)")
    print(f"CONCEPT DRIFT : {summary['concept_drift']} "
          f"-- {summary['concept_drift_reason']}")
    print(f"RETRAIN NOW   : {summary['retrain_triggered']}")
    print('='*50)

    # Airflow reads this JSON from stdout to decide whether to retrain
    if args.output_json:
        print(json.dumps(summary))

    return summary


if __name__ == "__main__":
    main()

