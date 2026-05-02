"""
Daily quality check using live production traces from LangSmith.
Alerts if average latency exceeds the SLA or error rate is too high.
"""

import os
import sys
import yaml
from datetime import datetime, timedelta


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def send_alert(message: str):
    """Log alert. Replace with PagerDuty / Slack webhook in production."""
    print(f"[ALERT] {datetime.now().isoformat()} — {message}")
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook_url:
        import json
        import urllib.request
        payload = json.dumps({"text": f":warning: {message}"}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)


def daily_quality_check(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    mon_cfg = cfg["monitoring"]

    try:
        from langsmith import Client
    except ImportError:
        print("langsmith not installed — skipping LangSmith quality check.")
        return

    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        print("LANGCHAIN_API_KEY not set — skipping LangSmith quality check.")
        return

    client = Client(api_key=api_key)
    project_name = mon_cfg["langsmith_project"]

    runs = list(client.list_runs(
        project_name=project_name,
        start_time=datetime.now() - timedelta(days=1),
        run_type="chain",
    ))

    if not runs:
        print("No runs found in the last 24h.")
        return

    latencies = [
        (r.end_time - r.start_time).total_seconds()
        for r in runs
        if r.end_time and r.start_time
    ]
    error_runs = [r for r in runs if r.error]

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    error_rate = len(error_runs) / len(runs) if runs else 0

    print(f"Runs (last 24h): {len(runs)}")
    print(f"Avg latency: {avg_latency:.2f}s  |  Error rate: {error_rate * 100:.1f}%")

    if avg_latency > mon_cfg["max_latency_seconds"]:
        send_alert(f"RAG latency degraded: avg {avg_latency:.1f}s > SLA {mon_cfg['max_latency_seconds']}s")

    if error_rate > mon_cfg["max_error_rate"]:
        send_alert(f"RAG error rate elevated: {error_rate * 100:.1f}% > threshold {mon_cfg['max_error_rate'] * 100:.0f}%")


if __name__ == "__main__":
    daily_quality_check()
