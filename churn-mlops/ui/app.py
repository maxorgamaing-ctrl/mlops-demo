"""
Streamlit UI for the Customer Churn Prediction demo.
Shows: model training status, live prediction form, drift report viewer.
Run: streamlit run ui/app.py  (from churn-mlops/ directory)
"""

import sys
import os
# Always run relative to churn-mlops root regardless of where streamlit was launched from
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import json
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import yaml

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction — MLOps Demo",
    page_icon="📉",
    layout="wide",
)

# ── Load config ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

cfg = load_config()

# ── Helpers ─────────────────────────────────────────────────────────────────────
def load_metrics():
    path = os.path.join("metrics", "scores.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_pipeline():
    steps = [
        ("Generating dataset...", ["python", "scripts/generate_dataset.py"]),
        ("Preprocessing data...", ["python", "src/data_preprocessing.py"]),
        ("Training model...", ["python", "src/train.py", "--fast-mode"]),
        ("Evaluating model...", ["python", "src/evaluate.py"]),
    ]
    results = []
    for label, cmd in steps:
        with st.spinner(label):
            res = subprocess.run(cmd, capture_output=True, text=True)
            results.append((label, res.returncode, res.stdout, res.stderr))
    return results


def engineer_features(d: dict) -> dict:
    d = d.copy()
    d["charge_per_tenure"] = d["monthlycharges"] / (d["tenure"] + 1)
    d["long_term_mth_to_mth"] = int(d["tenure"] > 24 and d["contract"] == "Month-to-month")
    return d


@st.cache_resource
def load_model():
    """Load trained sklearn pipeline from disk (bypasses MLflow for demo)."""
    import pickle
    model_path = os.path.join("models", "churn_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


# ── Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.title("📉 Churn MLOps Demo")
st.sidebar.markdown("**Use Case 1 — Classical ML**")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🚀 Train Pipeline", "🔮 Live Prediction", "📊 Model Metrics", "📡 Drift & Monitoring", "🔄 MLOps Pipeline"])

# ── Overview ─────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Customer Churn Prediction")
    st.markdown("""
    **Business Problem:** A telecom company loses ~15% of customers per year.  
    This model predicts which customers will churn **before** they do — enabling proactive retention.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size", "~7,000 rows")
    col2.metric("Features", "21")
    col3.metric("Model", "XGBoost")
    col4.metric("Quality Gate", "AUC ≥ 0.82")

    st.divider()
    st.subheader("MLOps Pillars Covered")

    pillars = {
        "Data Versioning": "DVC tracks every dataset change alongside code in Git",
        "Experiment Tracking": "MLflow logs params, metrics, and model artifacts for every run",
        "CI/CD Pipeline": "GitHub Actions trains + evaluates + blocks bad models automatically",
        "Model Serving": "FastAPI REST endpoint — same as any microservice",
        "Drift Monitoring": "Evidently AI detects feature distribution drift weekly",
        "Auto Retraining": "Airflow DAG re-trains when drift is detected",
    }
    for pillar, desc in pillars.items():
        st.markdown(f"- **{pillar}** — {desc}")

    st.divider()
    st.subheader("Repository Structure")
    st.code("""churn-mlops/
├── data/raw/          # DVC-tracked Telco CSV
├── src/               # train.py · evaluate.py · predict.py
├── api/               # FastAPI app (/health, /predict)
├── tests/             # pytest suite
├── monitoring/        # Evidently drift reports
├── dags/              # Airflow retraining DAG
└── .github/workflows/ # CI + CD pipelines""", language="text")

# ── Train Pipeline ───────────────────────────────────────────────────────────────
elif page == "🚀 Train Pipeline":
    st.title("Run the Training Pipeline")
    st.markdown("Runs the full MLOps loop: generate → preprocess → train → evaluate.")

    if st.button("▶ Run Full Pipeline", type="primary"):
        results = run_pipeline()
        for label, code, stdout, stderr in results:
            icon = "✅" if code == 0 else "❌"
            with st.expander(f"{icon} {label}", expanded=(code != 0)):
                if stdout:
                    st.code(stdout)
                if stderr and code != 0:
                    st.error(stderr)

        metrics = load_metrics()
        if metrics:
            st.success("Pipeline complete!")
            st.subheader("Model Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}", delta="≥0.82 gate")
            c2.metric("F1 Score", f"{metrics['f1']:.4f}")
            c3.metric("Precision", f"{metrics['precision']:.4f}")
            c4.metric("Recall", f"{metrics['recall']:.4f}")

            gate = cfg["quality_gate"]["min_auc_roc"]
            if metrics["auc_roc"] >= gate:
                st.success(f"✅ Quality gate PASSED — AUC {metrics['auc_roc']:.4f} ≥ {gate}")
            else:
                st.error(f"❌ Quality gate FAILED — AUC {metrics['auc_roc']:.4f} < {gate}")

# ── Live Prediction ──────────────────────────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.title("Live Churn Prediction")
    st.markdown("Enter customer features to get a real-time churn prediction.")

    model = load_model()
    if model is None:
        st.warning("No trained model found. Run the **Train Pipeline** tab first.")
        st.stop()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))
        with col2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col3:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        features = {
            "tenure": tenure,
            "monthlycharges": monthly_charges,
            "totalcharges": total_charges,
            "contract": contract,
            "internetservice": internet_service,
            "paymentmethod": payment_method,
            "phoneservice": phone_service,
            "multiplelines": "No",
            "onlinesecurity": online_security,
            "onlinebackup": "No",
            "deviceprotection": "No",
            "techsupport": tech_support,
            "streamingtv": "No",
            "streamingmovies": "No",
            "paperlessbilling": paperless_billing,
            "gender": "Male",
            "seniorcitizen": senior_citizen,
            "partner": "No",
            "dependents": "No",
        }
        features = engineer_features(features)
        df = pd.DataFrame([features])

        prob = model.predict_proba(df)[0][1]
        prediction = "CHURN" if prob >= 0.5 else "STAY"
        risk = "🔴 High" if prob >= 0.7 else ("🟡 Medium" if prob >= 0.4 else "🟢 Low")

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Churn Probability", f"{prob:.1%}")
        col_b.metric("Prediction", prediction)
        col_c.metric("Risk Tier", risk)

        st.progress(prob, text=f"Churn probability: {prob:.1%}")

# ── Model Metrics ────────────────────────────────────────────────────────────────
elif page == "📊 Model Metrics":
    st.title("Model Performance Metrics")

    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Run the **Train Pipeline** tab first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    c2.metric("F1 Score", f"{metrics['f1']:.4f}")
    c3.metric("Precision", f"{metrics['precision']:.4f}")
    c4.metric("Recall", f"{metrics['recall']:.4f}")

    st.divider()
    st.subheader("Quality Gate Thresholds")
    thresholds = {
        "AUC-ROC": (metrics["auc_roc"], cfg["quality_gate"]["min_auc_roc"]),
        "F1": (metrics["f1"], cfg["quality_gate"]["min_f1"]),
        "Precision": (metrics["precision"], cfg["quality_gate"]["min_precision"]),
        "Recall": (metrics["recall"], cfg["quality_gate"]["min_recall"]),
    }
    rows = []
    for metric, (val, threshold) in thresholds.items():
        rows.append({
            "Metric": metric,
            "Score": round(val, 4),
            "Threshold": threshold,
            "Status": "✅ PASS" if val >= threshold else "❌ FAIL",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Training Data Preview")
    train_path = os.path.join("data", "processed", "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        churn_rate = df["churn"].mean()
        st.metric("Churn Rate in Training Set", f"{churn_rate:.1%}")
        st.dataframe(df.head(10), use_container_width=True)

# ── Drift & Monitoring ────────────────────────────────────────────────────────────
elif page == "📡 Drift & Monitoring":
    import plotly.graph_objects as go

    st.title("Data Drift & Monitoring")
    st.markdown("""
    **What this shows:** Evidently AI compares training data (reference) vs incoming production data.
    If feature distributions shift significantly, the Airflow DAG triggers automatic retraining.
    """)

    train_path = os.path.join("data", "processed", "train.csv")
    if not os.path.exists(train_path):
        st.warning("Run the Train Pipeline first to generate data.")
        st.stop()

    ref_df = pd.read_csv(train_path)

    # Simulate drifted production data for demo
    np.random.seed(99)
    prod_df = ref_df.copy()
    prod_df["monthlycharges"] = prod_df["monthlycharges"] * np.random.uniform(1.1, 1.3, len(prod_df))
    prod_df["tenure"] = np.clip(prod_df["tenure"] - np.random.randint(0, 8, len(prod_df)), 0, 72)

    st.subheader("Drift Status Dashboard")
    features_to_check = ["monthlycharges", "totalcharges", "tenure", "charge_per_tenure"]
    drift_results = []
    for feat in features_to_check:
        if feat not in ref_df.columns:
            continue
        ref_mean = ref_df[feat].mean()
        prod_mean = prod_df[feat].mean()
        pct_change = abs(prod_mean - ref_mean) / (ref_mean + 1e-9) * 100
        drifted = pct_change > 10
        drift_results.append({
            "Feature": feat,
            "Reference Mean": round(ref_mean, 3),
            "Production Mean": round(prod_mean, 3),
            "Change %": round(pct_change, 1),
            "Status": "DRIFT" if drifted else "OK",
        })

    drift_df = pd.DataFrame(drift_results)
    n_drifted = (drift_df["Status"] == "DRIFT").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Features Monitored", len(drift_results))
    col2.metric("Features Drifted", int(n_drifted), delta=f"{n_drifted} alerts" if n_drifted else None, delta_color="inverse")
    col3.metric("Retraining Triggered", "YES" if n_drifted > 0 else "NO")

    st.dataframe(drift_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Feature Distribution: Reference vs Production")
    selected_feat = st.selectbox("Select feature to inspect:", [f for f in features_to_check if f in ref_df.columns])

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref_df[selected_feat], name="Reference (Training)",
                               opacity=0.65, marker_color="steelblue", nbinsx=40))
    fig.add_trace(go.Histogram(x=prod_df[selected_feat], name="Production (Simulated)",
                               opacity=0.65, marker_color="tomato", nbinsx=40))
    fig.update_layout(barmode="overlay", height=380,
                      xaxis_title=selected_feat, yaxis_title="Count",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Churn Rate Trend (Simulated Production Window)")
    weeks = pd.date_range("2025-01-01", periods=12, freq="W")
    churn_trend = 0.52 + np.cumsum(np.random.normal(0.008, 0.005, 12))
    retrain_week = 8

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=weeks, y=churn_trend, mode="lines+markers",
                              name="Churn Rate", line=dict(color="tomato", width=2)))
    fig2.add_hline(y=0.65, line_dash="dash", line_color="orange",
                   annotation_text="Drift Alert Threshold")
    fig2.add_vline(x=weeks[retrain_week].timestamp() * 1000, line_dash="dot",
                   line_color="green", annotation_text="Retrain Triggered")
    fig2.update_layout(height=320, xaxis_title="Week", yaxis_title="Churn Rate",
                       yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("In production: Evidently generates HTML reports weekly. The Airflow DAG reads the drift flag and calls `src/train.py` automatically.")

# ── MLOps Pipeline ────────────────────────────────────────────────────────────────
elif page == "🔄 MLOps Pipeline":
    st.title("MLOps Pipeline Overview")

    st.subheader("End-to-End Flow")
    st.code("""\
DATA & VERSIONING
  Raw CSV ──► DVC track ──► Git commit ──► S3 artifact store

CI/CD  (GitHub Actions — triggers on every git push)
  preprocess ──► train ──► quality gate ──► Docker build ──► push
                  │
            MLflow logs:
            · params: n_estimators, max_depth, learning_rate
            · metrics: AUC-ROC, F1, Precision, Recall
            · artifact: sklearn pipeline → model registry

SERVING
  MLflow registry (Staging → Production)
  FastAPI /predict ──► Kubernetes ──► Load Balancer

MONITORING  (Airflow DAG — runs every Monday 6am)
  check_drift ──► [drift detected?]
                      YES ──► preprocess ──► train ──► evaluate ──► promote
                      NO  ──► skip (model still valid)
""", language="text")

    st.divider()
    st.subheader("Airflow DAG Steps")
    dag_steps = [
        ("check_drift",           "Run Evidently drift report vs reference training data",        "Every Monday 6am"),
        ("branch_on_drift",       "Branch: retrain if drift > threshold, else skip",              "Conditional"),
        ("preprocess_data",       "Re-run data_preprocessing.py on fresh production data",        "On drift only"),
        ("train_model",           "src/train.py — XGBoost + MLflow tracking",                     "On drift only"),
        ("evaluate_vs_baseline",  "Compare new model AUC vs current production model",            "On drift only"),
        ("promote_if_better",     "promote_model.py — Staging → Production in MLflow registry",   "On improvement"),
    ]
    for name, desc, timing in dag_steps:
        st.markdown(f"**`{name}`** — {desc}  \n*Timing: {timing}*")
        st.markdown("↓")

    st.divider()
    st.subheader("MLflow Experiment Runs")
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = mlflow.tracking.MlflowClient()
        runs_data = []
        for exp in client.search_experiments():
            for run in client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=5):
                m = run.data.metrics
                runs_data.append({
                    "Run ID": run.info.run_id[:8],
                    "Status": run.info.status,
                    "AUC-ROC": round(m.get("auc_roc", 0), 4),
                    "F1": round(m.get("f1", 0), 4),
                    "Precision": round(m.get("precision", 0), 4),
                    "Recall": round(m.get("recall", 0), 4),
                })
        if runs_data:
            st.dataframe(pd.DataFrame(runs_data), use_container_width=True, hide_index=True)
            st.caption("Live from local MLflow tracking DB. Also run `mlflow ui` for the full experiment browser at http://localhost:5000")
        else:
            st.info("No runs yet — run the Train Pipeline tab to populate MLflow.")
    except Exception as e:
        st.warning(f"MLflow not available: {e}")

    st.divider()
    st.subheader("Model Registry Stages")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**None**\nNew model, not yet evaluated")
    c2.warning("**Staging**\nPassed quality gate, under shadow testing")
    c3.success("**Production**\nLive — serving real traffic")
    c4.markdown("**Archived**\nReplaced by newer version")
