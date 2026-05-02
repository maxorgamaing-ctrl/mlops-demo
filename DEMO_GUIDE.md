# MLOps Demo Guide — What to Show & What to Run
**Audience:** DevOps Team | **Duration:** 30–40 min | **Date:** May 2026

---

## Before You Start — Pre-Flight Checklist

Run these in order **before the session**. Each should return a clean result.

```powershell
# 1. Start the Churn API (port 8000)
python "c:\Users\omar_\OneDrive\Desktop\MLops_session\run_churn_api.py"

# 2. Start the RAG API (port 8001) — new terminal
$env:PYTHONPATH = "c:\Users\omar_\OneDrive\Desktop\MLops_session\rag-product-assistant"
python "c:\Users\omar_\OneDrive\Desktop\MLops_session\run_rag_api.py"

# 3. Start Churn Streamlit UI (port 8501) — new terminal
python -m streamlit run "c:\Users\omar_\OneDrive\Desktop\MLops_session\churn-mlops\ui\app.py" --server.port 8501

# 4. Start RAG Streamlit UI (port 8502) — new terminal
$env:PYTHONPATH = "c:\Users\omar_\OneDrive\Desktop\MLops_session\rag-product-assistant"
python -m streamlit run "c:\Users\omar_\OneDrive\Desktop\MLops_session\rag-product-assistant\ui\app.py" --server.port 8502

# 5. Start MLflow UI (port 5000) — new terminal
Set-Location "c:\Users\omar_\OneDrive\Desktop\MLops_session\churn-mlops"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Browser tabs to have open before starting:**
| Tab | URL |
|-----|-----|
| GitHub Actions | https://github.com/maxorgamaing-ctrl/mlops-demo/actions |
| Churn Streamlit UI | http://localhost:8501 |
| RAG Streamlit UI | http://localhost:8502 |
| MLflow UI | http://localhost:5000 |
| Churn API Swagger | http://localhost:8000/docs |
| RAG API Swagger | http://localhost:8001/docs |

---

## 0–5 min | The Problem (Slides)

**Show slides 1–6.** Key talking points:

- "87% of ML projects never reach production. MLOps is why."
- Draw the comparison: **Left column (without MLOps) = pain your team already feels with software. Right column = what CI/CD gave DevOps teams in 2010. MLOps does the same for ML.**
- Slide 6 is the bridge: "You already own Git, CI/CD, Docker, K8s. MLOps adds 3 things: data versioning, experiment tracking, drift detection."

---

## 5–20 min | USE CASE 1 — Customer Churn Prediction

### Part A — Show the pipeline ran (2 min)

Open the **Churn Streamlit UI** → `MLOps Pipeline` page.

Point out: generate → preprocess → train → evaluate → serve. Each step is logged.

Then open **MLflow UI** (http://localhost:5000):
- Click into the `churn-prediction` experiment
- Show multiple runs with AUC scores, hyperparameters
- Say: *"Every training run is reproducible. Any team member can re-run the exact same experiment."*

---

### Part B — Live prediction (2 min)

Open **Churn API Swagger** → http://localhost:8000/docs → `POST /predict`

Paste this and hit Execute:
```json
{
  "tenure": 2,
  "monthly_charges": 85.0,
  "total_charges": 170.0,
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
  "dependents": "No"
}
```

Show the response — churn probability + risk tier. Say: *"This is what every downstream system hits — CRM, email campaign, retention dashboard."*

---

### Part C — The CI/CD Quality Gate (5 min) ⭐ KEY MOMENT

**Go to GitHub Actions:** https://github.com/maxorgamaing-ctrl/mlops-demo/actions

Click the latest **CI -- Validate ML Code and Model** run. Walk through each step:

| Step | What to say |
|------|-------------|
| Run data validation tests | "Schema checks — if someone adds a bad column, CI fails immediately" |
| Train model (fast CI mode) | "Every PR trains a model. Not the full dataset, but enough to catch regressions" |
| Evaluate — AUC >= 0.82 | "This is the quality gate. Model must beat 0.82 or the PR is blocked" |
| Simulate data drift | "We inject drifted data and verify the drift detector catches it" |
| Simulate concept drift | "We flip labels to degrade model accuracy — system must detect it" |
| Build Docker image | "Same container runs in staging and prod — no 'works on my laptop'" |
| Smoke-test API container | "We curl /health against the actual Docker container before merging" |

**Key line:** *"This is identical to how you gate application deployments today — except the test is 'does the model still work?' not 'do the unit tests pass?'"*

**Then show CD run:** Click the `CD - Register and Deploy` run.
- Tests run again before deploy
- Docker image pushed to `ghcr.io/maxorgamaing-ctrl/churn-api`
- Say: *"On merge to main, the model is automatically promoted and the image is published. No manual steps."*

---

### Part D — Drift Detection & Auto-Retrain (5 min) ⭐ KEY MOMENT

Open a terminal. Run this to inject real data drift:

```powershell
Set-Location "c:\Users\omar_\OneDrive\Desktop\MLops_session\churn-mlops"

# Inject drift: shift monthly charges 2.5x, collapse tenure, lock all contracts
python -c "
import pandas as pd, numpy as np, os
df = pd.read_csv('data/processed/train.csv').sample(500, random_state=1).copy()
df['monthlycharges'] = df['monthlycharges'] * 2.5
df['totalcharges']   = df['totalcharges']   * 2.5
df['tenure']         = (df['tenure'] * 0.3).astype(int)
df['contract']       = 'Month-to-month'
os.makedirs('data/production', exist_ok=True)
df.to_csv('data/production/recent_predictions.csv', index=False)
print('Drift injected.')
"

# Run drift detection
python monitoring\drift_check.py
```

**Expected output:**
```
DATA DRIFT    : True  (6/19 features shifted)
CONCEPT DRIFT : False
RETRAIN NOW   : True
[ALERT] DATA DRIFT: 6/19 features shifted -- retrain triggered
```

Point to the output and say: *"This fires an alert. In production, this goes to PagerDuty — the same on-call rotation you use for app incidents. And it automatically triggers the Airflow DAG to retrain."*

**Then go to GitHub Actions → Drift Monitor → Run workflow (manually):**

```powershell
# Or trigger from CLI:
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
gh workflow run drift_monitor.yml --repo maxorgamaing-ctrl/mlops-demo --ref main -f simulate_drift=true
```

Show the workflow running. Point out:
- Runs daily at 02:00 UTC automatically
- If drift detected → triggers `retrain.yml` automatically
- New model is evaluated against the quality gate
- Only deployed if it's better than the current model

**Open Churn Streamlit UI → Drift & Monitoring page** to show the drift report visualized.

---

## 20–30 min | USE CASE 2 — Product Knowledge Assistant (RAG)

### Part A — Live Q&A Demo (3 min) ⭐ KEY MOMENT

Open **RAG Streamlit UI** → http://localhost:8502 → Chat page.

Ask these 3 questions in order:

1. `"Does the TP-Link AX3000 support WPA3?"`
2. `"What is the return policy for electronics?"`
3. `"Compare the storage options available in the laptop catalog"`

For each answer, point out:
- The **source document name** cited in the response
- The answer is **grounded** — it cannot hallucinate facts not in the catalog
- Say: *"This is not ChatGPT guessing. It retrieved the exact chunk from the product catalog and grounded the answer in it."*

---

### Part B — Show the API (2 min)

Open **RAG API Swagger** → http://localhost:8001/docs → `POST /answer`

Run:
```json
{
  "question": "Does the TP-Link AX3000 support WPA3?",
  "top_k": 5
}
```

Show the response includes `sources` array — the exact document chunks used. Say: *"Every production RAG system needs this. It's how you audit hallucinations."*

Also hit `GET /health` — show `documents_indexed: 63`.

---

### Part C — Show the RAG Pipeline (2 min)

Open **RAG Streamlit UI → RAG Evaluation page**.

Show the RAGAS metrics:
- **Faithfulness** — did the answer only use information from retrieved chunks?
- **Answer Relevancy** — did the answer actually address the question?
- **Context Precision** — were the right chunks retrieved?

Say: *"These are the RAG equivalents of AUC for a classifier. In CI, if faithfulness drops below 0.85, the PR is blocked — same as the AUC gate in Use Case 1."*

---

### Part D — Auto Re-index Pipeline (3 min)

Open a terminal. Add a new product to the catalog and trigger re-indexing:

```powershell
Set-Location "c:\Users\omar_\OneDrive\Desktop\MLops_session\rag-product-assistant"

# Re-run the indexer (simulates a catalog update triggering the pipeline)
python ingestion\indexer.py

# Verify new chunks are indexed
python -c "
import chromadb
client = chromadb.PersistentClient(path='vector_store/chroma')
col = client.get_collection('product_knowledge_base')
print(f'Documents indexed: {col.count()}')
"
```

Say: *"In production, pushing to the product catalog repo triggers the GitHub Actions `reindex.yml` workflow. No manual steps. New product is searchable in under 2 minutes."*

Show the `rag-product-assistant/.github/workflows/reindex.yml` file to prove it exists.

---

## 30–40 min | Q&A Anchor Points

Use slide 14 (the comparison table) as a visual. Common questions and answers:

**"How is this different from just calling the OpenAI API?"**
> "The RAG system grounds every answer in your own data. The LLM can't make up facts that aren't in your catalog. And every call is traced — you know exactly what was retrieved, at what cost, with what latency."

**"How long does this take to build for a real project?"**
> "The platform takes 4–6 weeks to build once. Every new model after that takes 1 week — the pipeline is already there. That's slide 15."

**"What happens when the model degrades?"**
> "Drift monitor fires. Airflow retriggers training. If the new model beats the quality gate, it's deployed automatically. If not, the old model stays in production and the team is alerted."

**"Can this run on our existing K8s cluster?"**
> "Yes. The Docker images are already being built in CD. Kubernetes manifests point at the ghcr.io image. The only addition is the MLflow tracking server — one more deployment."

**"What about cost?"**
> "Use Case 1 runs on a single CPU. Training takes under 2 minutes. The Airflow DAG runs weekly. RAG uses a local LLM (Ollama) — zero API cost per query. The only recurring cost is the vector DB storage."

---

## Emergency Commands (if something breaks mid-demo)

```powershell
# Kill anything on port 8000 or 8001
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue).OwningProcess -Force -ErrorAction SilentlyContinue

# Quick health check — both APIs running?
curl http://localhost:8000/health
curl http://localhost:8001/health

# Re-run the full churn pipeline (takes ~2 min)
Set-Location "c:\Users\omar_\OneDrive\Desktop\MLops_session\churn-mlops"
python scripts\generate_dataset.py; python src\data_preprocessing.py; python src\train.py; python src\evaluate.py

# Reset drift (remove injected data)
Remove-Item "data\production\recent_predictions.csv" -ErrorAction SilentlyContinue

# Check GitHub Actions live
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
gh run list --repo maxorgamaing-ctrl/mlops-demo --limit 5
```

---

## Key Numbers to Quote

| Metric | Value | Source |
|--------|-------|--------|
| Churn model AUC | 0.9893 | Last training run |
| Documents indexed | 63 chunks | ChromaDB |
| CI pipeline duration | ~4 min | GitHub Actions |
| CD pipeline duration | ~5 min | GitHub Actions |
| Drift threshold | 15% features shifted | `drift_check.py` |
| Quality gate | AUC >= 0.82 | `config.yaml` |
| Daily drift check | 02:00 UTC | `drift_monitor.yml` |
| GitHub repo | https://github.com/maxorgamaing-ctrl/mlops-demo | Live |
