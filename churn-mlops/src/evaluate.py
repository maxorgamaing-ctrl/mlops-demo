"""
Evaluate the trained churn model against quality-gate thresholds.
Exits with code 1 if any threshold is not met (used by CI pipeline).
"""

import argparse
import json
import os
import sys
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(metric: str = None, config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    scores_path = os.path.join("metrics", "scores.json")

    if not os.path.exists(scores_path):
        print("ERROR: metrics/scores.json not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    with open(scores_path) as f:
        scores = json.load(f)

    # If called with --output-metric, just print that value (used in CI bash script)
    if metric:
        print(scores.get(metric, 0))
        return

    qg = cfg["quality_gate"]
    failed = []

    if scores["auc_roc"] < qg["min_auc_roc"]:
        failed.append(f"auc_roc {scores['auc_roc']:.4f} < threshold {qg['min_auc_roc']}")
    if scores["f1"] < qg["min_f1"]:
        failed.append(f"f1 {scores['f1']:.4f} < threshold {qg['min_f1']}")
    if scores["precision"] < qg["min_precision"]:
        failed.append(f"precision {scores['precision']:.4f} < threshold {qg['min_precision']}")
    if scores["recall"] < qg["min_recall"]:
        failed.append(f"recall {scores['recall']:.4f} < threshold {qg['min_recall']}")

    if failed:
        print("QUALITY GATE FAILED:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print("Quality gate PASSED:")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-metric", dest="metric", default=None,
                        help="Print a single metric value and exit (for CI scripts)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate(metric=args.metric, config_path=args.config)
