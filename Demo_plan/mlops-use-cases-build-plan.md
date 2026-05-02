# MLOps Session — Build Plan for Two Production Use Cases
**Audience:** DevOps Team  
**Date:** May 2026  
**Format:** Hands-on demo plan + implementation guide

---

## Overview

This document is your complete build plan for two production-grade MLOps demos. Each use case follows real-world engineering patterns, not toy examples. Every component maps to a pillar of the MLOps lifecycle your DevOps team already understands from software delivery — extended for machine learning.

**Use Case 1 — Classical ML:** Customer Churn Prediction (Telecom)  
**Use Case 2 — LLM + RAG + Agent:** Product Knowledge Assistant (E-Commerce)

---

## Why These Two Use Cases?

**Churn Prediction** is the industry's consensus "best first MLOps project":
- Tabular data — no GPU needed, no exotic infrastructure
- Binary classification — metrics (accuracy, F1, AUC-ROC) are easy to explain to any audience
- Strong business story: every company loses customers; predicting churn = saving revenue
- Covers every single MLOps pillar: versioning, CI/CD, registry, serving, monitoring, drift

**Product Knowledge Assistant (RAG)** is the best way to show LLMOps:
- Relatable to any audience (ask a question, get an answer from a catalog)
- Demonstrates the full stack: ingestion → vector DB → retrieval → generation → evaluation
- Shows where MLOps extends into LLMOps: prompt versioning, RAG evaluation, observability
- No domain knowledge required to understand the demo

---

## Use Case 1 — Customer Churn Prediction

### Business Problem

A telecom company loses ~15% of customers per year (industry average). Reactive support (calling after a customer cancels) is too late. The model predicts which customers will churn **before** they do, enabling proactive retention campaigns. A 5% improvement in retention can increase profits by 25–95%.

### Dataset

Use the **Telco Customer Churn** dataset from Kaggle/IBM:
- ~7,000 rows, 21 features
- Target: `Churn` (Yes/No)
- Features: tenure, monthly charges, contract type, internet service, etc.
- Public domain — no data access issues for a demo

### ML Approach

| Decision | Choice | Why |
|---|---|---|
| Model | XGBoost + scikit-learn baseline | Simple, fast, interpretable |
| Preprocessing | StandardScaler + OneHotEncoder | Pipeline-safe, no leakage |
| Evaluation | AUC-ROC, F1, Precision/Recall | Handles class imbalance |
| Serving | FastAPI REST endpoint | DevOps-friendly, same as any microservice |

### Repository Structure

```
churn-mlops/
├── data/
│   ├── raw/                    # Original Kaggle CSV (tracked by DVC)
│   └── processed/              # Feature-engineered output
├── src/
│   ├── data_preprocessing.py   # Cleaning + feature engineering
│   ├── train.py                # MLflow-tracked training run
│   ├── evaluate.py             # Model evaluation vs. baseline
│   └── predict.py              # Inference logic
├── api/
│   ├── main.py                 # FastAPI app (/health, /predict)
│   └── schemas.py              # Pydantic request/response models
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── monitoring/
│   └── drift_check.py          # Evidently AI drift report
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint + test on every PR
│       └── cd.yml              # Build → register → deploy on merge to main
├── Dockerfile
├── dvc.yaml                    # Data pipeline definition
├── config.yaml                 # Hyperparameters + thresholds
└── requirements.txt
```

### MLOps Pillar Implementation

#### 1. Data Versioning (DVC + Git)

```yaml
# dvc.yaml — defines the data pipeline as code
stages:
  preprocess:
    cmd: python src/data_preprocessing.py
    deps:
      - data/raw/telco_churn.csv
      - src/data_preprocessing.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
  train:
    cmd: python src/train.py
    deps:
      - data/processed/train.csv
      - src/train.py
      - config.yaml
    outs:
      - models/churn_model.pkl
```

**What to show:** Run `dvc repro` and `dvc push` — the team sees data pipelines treated exactly like software pipelines. Run `git log` to show that data changes are tracked alongside code changes.

#### 2. Experiment Tracking (MLflow)

```python
# src/train.py — core training with full MLflow tracking
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score

mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    # Log everything: params, metrics, model, artifacts
    mlflow.log_params({
        "model_type": "XGBoost",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "dataset_version": "v1.2"
    })

    # ... train model ...

    mlflow.log_metrics({
        "auc_roc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision,
        "recall": recall
    })

    # Log model to registry with signature
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        registered_model_name="churn-model-prod",
        input_example=X_test[:5]
    )
```

**What to show:** Open the MLflow UI. Show experiment comparison — multiple runs with different hyperparameters. Show the model registry with `Staging` and `Production` stages.

#### 3. CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml — runs on every pull request
name: CI — Validate ML Code and Model

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run data validation
        run: python -m pytest tests/test_preprocessing.py -v

      - name: Train model (mini run)
        run: python src/train.py --fast-mode

      - name: Evaluate — must beat baseline AUC 0.82
        run: |
          AUC=$(python src/evaluate.py --output-metric auc_roc)
          if (( $(echo "$AUC < 0.82" | bc -l) )); then
            echo "Model failed quality gate: AUC=$AUC < 0.82"
            exit 1
          fi

      - name: Build and smoke-test API container
        run: |
          docker build -t churn-api:test .
          docker run -d -p 8000:8000 churn-api:test
          sleep 5
          curl -f http://localhost:8000/health

# .github/workflows/cd.yml — runs on merge to main
name: CD — Register and Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    steps:
      - name: Promote model in MLflow registry
        run: python scripts/promote_model.py --stage Production

      - name: Build and push Docker image
        run: |
          docker build -t churn-api:${{ github.sha }} .
          docker push registry.company.com/churn-api:${{ github.sha }}

      - name: Deploy to Kubernetes
        run: kubectl set image deployment/churn-api churn-api=registry.company.com/churn-api:${{ github.sha }}
```

**What to show:** Trigger a bad model (deliberately lower AUC) and watch the CI pipeline block the deployment. Then fix it and watch the full CD flow succeed. This is the DevOps muscle memory the team already has — now applied to ML.

#### 4. Model Serving (FastAPI + Docker)

```python
# api/main.py — production-grade serving API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Load latest Production model from registry at startup
model = mlflow.pyfunc.load_model("models:/churn-model-prod/Production")

class CustomerFeatures(BaseModel):
    tenure: int
    monthly_charges: float
    contract: str          # "Month-to-month", "One year", "Two year"
    internet_service: str
    total_charges: float
    # ... other features

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: str        # "churn" or "stay"
    risk_tier: str         # "high", "medium", "low"

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": model.metadata.run_id}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, customer_id: str):
    df = pd.DataFrame([customer.dict()])
    probability = model.predict(df)[0]
    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=float(probability),
        prediction="churn" if probability > 0.5 else "stay",
        risk_tier="high" if probability > 0.7 else "medium" if probability > 0.4 else "low"
    )
```

**Demo script:** Hit `/predict` with a high-risk customer profile. Show the JSON response. Then open the FastAPI Swagger UI at `/docs` — it's auto-generated from the Pydantic schemas.

#### 5. Monitoring and Drift Detection (Evidently AI)

```python
# monitoring/drift_check.py — weekly scheduled job
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def run_drift_report(reference_data, production_data):
    report = Report(metrics=[
        DataDriftPreset(),        # Feature distribution drift
        ClassificationPreset()    # Prediction drift + performance
    ])
    report.run(
        reference_data=reference_data,   # Training data distribution
        current_data=production_data     # Last 7 days of live predictions
    )
    report.save_html("reports/drift_report.html")

    # Alert if drift detected
    drift_summary = report.as_dict()
    if drift_summary["metrics"][0]["result"]["dataset_drift"]:
        send_alert("Data drift detected in churn model — review retraining")
```

**What to show:** Inject synthetic drifted data (e.g., suddenly all customers have month-to-month contracts). Open the Evidently HTML report. Show feature drift charts. Trigger the retraining pipeline.

#### 6. Automated Retraining Trigger (Airflow DAG)

```python
# dags/retrain_churn_model.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta

with DAG(
    "churn_model_retraining",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False
) as dag:

    check_drift = BranchPythonOperator(
        task_id="check_data_drift",
        python_callable=evaluate_drift,   # returns "retrain" or "skip"
    )

    retrain = PythonOperator(
        task_id="retrain",
        python_callable=run_training_pipeline
    )

    evaluate_new_model = PythonOperator(
        task_id="evaluate_vs_baseline",
        python_callable=compare_against_production
    )

    promote_if_better = PythonOperator(
        task_id="promote_model",
        python_callable=promote_to_production_if_better
    )

    check_drift >> [retrain, skip_retraining]
    retrain >> evaluate_new_model >> promote_if_better
```

**What to show:** Open the Airflow UI. Show the DAG graph. Manually trigger a run. Show logs for each step. Show the model registry in MLflow — a new version appears after the run, and the old one is automatically archived.

---

### Demo Flow (15 minutes)

1. **[2 min]** Show the problem: "87% of ML projects never reach production. Here's one that does."
2. **[3 min]** Walk the repo structure. Emphasize: this looks exactly like a software project.
3. **[3 min]** Open MLflow UI — compare 3 experiment runs. Show model registry (Staging → Production promotion).
4. **[3 min]** Trigger a CI pipeline with a bad model. Watch it fail the quality gate. Fix it. Watch it deploy.
5. **[2 min]** Hit the live `/predict` endpoint. Show the Swagger UI.
6. **[2 min]** Inject drifted data. Show the Evidently drift report. Show Airflow triggering retraining.

---

## Use Case 2 — Product Knowledge Assistant (RAG + Agent)

### Business Problem

An e-commerce company has 50,000+ products, each with specifications, FAQs, compatibility info, and user manuals. Customer support agents spend 60% of their time searching for product information. An AI assistant that answers product questions from the knowledge base — grounded in real documents, not hallucinating — reduces support time and improves answer accuracy.

### Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Gateway (auth, rate limiting, logging)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  LangChain Orchestration Layer                           │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Query       │    │  Retriever   │                   │
│  │  Rewriter    │───►│  (Hybrid     │                   │
│  │  (LLM)       │    │   Search)    │                   │
│  └──────────────┘    └──────┬───────┘                   │
│                             │                            │
│                    ┌────────▼────────┐                  │
│                    │  Reranker       │                  │
│                    │  (Cross-encoder)│                  │
│                    └────────┬────────┘                  │
│                             │                            │
│                    ┌────────▼────────┐                  │
│                    │  LLM Generator  │                  │
│                    │  (Claude/GPT)   │                  │
│                    └────────┬────────┘                  │
│                             │                            │
│                    ┌────────▼────────┐                  │
│                    │  Response +     │                  │
│                    │  Source Citatio │                  │
│                    └─────────────────┘                  │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Observability│
              │  (LangSmith) │
              └──────────────┘
```

### Tech Stack

| Component | Tool | Role |
|---|---|---|
| Orchestration | LangChain | Chain retrieval + generation |
| Vector Database | ChromaDB (dev) / Qdrant (prod) | Semantic search over product docs |
| Embeddings | OpenAI `text-embedding-3-small` | Document + query vectorization |
| LLM | Claude Sonnet or GPT-4o | Answer generation |
| Reranker | `cross-encoder/ms-marco-MiniLM` | Improve retrieval precision |
| Serving | FastAPI | REST API |
| Observability | LangSmith | Tracing, cost, latency |
| Evaluation | RAGAS | Context relevance, faithfulness, answer quality |
| Data Versioning | DVC | Track knowledge base document versions |
| Prompt Registry | MLflow (custom) | Version and track prompt changes |

### Repository Structure

```
rag-product-assistant/
├── data/
│   ├── raw/
│   │   ├── product_catalog.json    # Product specs (DVC tracked)
│   │   ├── faqs.csv
│   │   └── manuals/               # PDF user manuals
│   └── processed/
│       └── chunks/                # Pre-processed text chunks
├── ingestion/
│   ├── loader.py                  # Load + parse documents
│   ├── chunker.py                 # Semantic chunking strategy
│   └── indexer.py                 # Embed + store in vector DB
├── retrieval/
│   ├── retriever.py               # Hybrid search (dense + sparse)
│   └── reranker.py                # Cross-encoder reranking
├── generation/
│   ├── chain.py                   # LangChain RAG chain
│   └── prompts/
│       └── qa_prompt_v2.txt       # Versioned prompt template
├── api/
│   ├── main.py                    # FastAPI app
│   └── schemas.py
├── evaluation/
│   ├── test_questions.json        # Ground-truth Q&A pairs
│   └── evaluate_rag.py            # RAGAS evaluation suite
├── monitoring/
│   └── quality_check.py          # Daily quality run against live traces
├── .github/
│   └── workflows/
│       ├── ci_rag.yml             # Test retrieval quality gate
│       └── reindex.yml            # Trigger re-indexing on catalog update
├── Dockerfile
└── requirements.txt
```

### MLOps Pillar Implementation for RAG

#### 1. Data Ingestion Pipeline (Document Versioning)

```python
# ingestion/chunker.py — semantic chunking (critical for retrieval quality)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_product_document(doc: dict) -> list[str]:
    """
    Best practice: chunk by section, not by fixed token count.
    Each chunk includes product name as context header to prevent
    retrieval-serving skew (the RAG equivalent of training-serving skew).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,        # Overlap preserves context across boundaries
        separators=["\n\n", "\n", ".", " "]
    )
    # Prepend product name to every chunk — prevents decontextualized retrieval
    chunks = splitter.split_text(doc["description"])
    return [f"Product: {doc['name']}\n\n{chunk}" for chunk in chunks]
```

**Key MLOps point:** Every time the product catalog is updated, the ingestion pipeline re-runs automatically via a GitHub Actions trigger (`reindex.yml`). The vector database version is tagged in DVC, so you can roll back to any previous knowledge base state — just like rolling back a model version.

#### 2. Retrieval Chain (LangChain + Hybrid Search)

```python
# retrieval/retriever.py
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever

def build_retriever(vectorstore, documents):
    """
    Hybrid search = dense (semantic) + sparse (keyword/BM25).
    Dense finds conceptually similar docs; BM25 finds exact term matches.
    Neither alone is as good as both combined.
    """
    # Dense retriever — semantic similarity
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",         # Maximum Marginal Relevance — avoids redundant results
        search_kwargs={"k": 10, "fetch_k": 30}
    )

    # Sparse retriever — keyword matching
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = 10

    # Combine: 60% semantic, 40% keyword
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4]
    )
```

#### 3. RAG Chain with Versioned Prompts

```python
# generation/chain.py
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

# Prompt is versioned as a file — tracked in Git + logged to MLflow
PROMPT_TEMPLATE = open("generation/prompts/qa_prompt_v2.txt").read()

def build_rag_chain(retriever):
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True    # Always return sources for auditability
    )
```

```
# generation/prompts/qa_prompt_v2.txt
You are a helpful product assistant for an e-commerce store.
Answer the customer's question using ONLY the information provided in the context below.
If the answer is not in the context, say "I don't have that information — please contact support."
Do NOT make up product specifications, prices, or compatibility details.

Context:
{context}

Question: {question}

Answer (cite the product name when relevant):
```

**Key MLOps point:** Prompts are code. Every change to `qa_prompt_v2.txt` triggers CI. The old prompt is never deleted — it's versioned. You can roll back a prompt the same way you roll back a Docker image.

#### 4. Evaluation with RAGAS (CI Quality Gate)

```python
# evaluation/evaluate_rag.py — run as part of CI before deploying new prompt/retriever
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # Does the answer stay faithful to retrieved docs?
    answer_relevancy,      # Does the answer address the question?
    context_precision,     # Are the retrieved chunks actually relevant?
    context_recall         # Did retrieval find all necessary information?
)
from datasets import Dataset

def run_evaluation(rag_chain, test_questions_path: str) -> dict:
    test_data = load_test_questions(test_questions_path)

    # Generate answers for all test questions
    results = []
    for item in test_data:
        response = rag_chain({"query": item["question"]})
        results.append({
            "question": item["question"],
            "answer": response["result"],
            "contexts": [d.page_content for d in response["source_documents"]],
            "ground_truth": item["expected_answer"]
        })

    dataset = Dataset.from_list(results)
    scores = evaluate(dataset, metrics=[
        faithfulness, answer_relevancy,
        context_precision, context_recall
    ])

    # Quality gate — block deployment if scores drop
    assert scores["faithfulness"] > 0.85,     f"Faithfulness too low: {scores['faithfulness']}"
    assert scores["answer_relevancy"] > 0.80, f"Relevancy too low: {scores['answer_relevancy']}"
    assert scores["context_precision"] > 0.75, "Retrieval precision degraded"

    return scores
```

**What to show:** Change the chunking strategy (e.g., make chunks too small). Watch the `context_recall` score drop in CI. The deployment is blocked. This is the RAG equivalent of a model failing its AUC quality gate.

#### 5. Observability (LangSmith)

```python
# Every chain call is automatically traced when LANGCHAIN_TRACING_V2=true
# No code changes needed — set the env var and every request is logged

# monitoring/quality_check.py — daily job using live production traces
from langsmith import Client

def daily_quality_check():
    client = Client()
    # Pull last 24h of production traces
    runs = client.list_runs(
        project_name="product-assistant-prod",
        start_time=datetime.now() - timedelta(days=1)
    )

    # Compute metrics from live traffic
    latencies = [r.end_time - r.start_time for r in runs]
    error_runs = [r for r in runs if r.error]

    avg_latency = sum(latencies) / len(latencies)
    error_rate = len(error_runs) / len(runs)

    if avg_latency > 3.0:
        alert("RAG latency degraded: avg %.1fs" % avg_latency)
    if error_rate > 0.05:
        alert("RAG error rate elevated: %.1f%%" % (error_rate * 100))
```

**What to show:** Open the LangSmith UI. Show a trace for a single query — you can see every step: query rewrite → retrieval → documents fetched → LLM call → response. Click into any step to see input/output. This is the distributed tracing that DevOps teams know from Jaeger/Zipkin — now applied to LLM chains.

#### 6. Re-indexing CI/CD Trigger

```yaml
# .github/workflows/reindex.yml — triggered when product catalog changes
name: Re-index Knowledge Base

on:
  push:
    paths:
      - 'data/raw/product_catalog.json'
      - 'data/raw/faqs.csv'
      - 'data/raw/manuals/**'

jobs:
  reindex:
    runs-on: ubuntu-latest
    steps:
      - name: Run ingestion pipeline
        run: python ingestion/indexer.py --env production

      - name: Run retrieval quality smoke test
        run: python evaluation/evaluate_rag.py --quick-mode

      - name: Tag knowledge base version in DVC
        run: |
          dvc push
          git tag kb-$(date +%Y%m%d-%H%M%S)
          git push --tags
```

**Key MLOps point:** The knowledge base is a versioned artifact. When product specs change, the re-indexing pipeline runs automatically — same as how a code change triggers a model retraining. The DevOps team owns this pipeline.

---

### Demo Flow (15 minutes)

1. **[2 min]** Show the problem: "50,000 products. Customer asks: 'Does this router work with my modem?' Support agent spends 8 minutes searching. AI does it in 2 seconds."
2. **[3 min]** Show the architecture diagram. Walk through the 5 components: ingestion → vector DB → retrieval → LLM → response.
3. **[3 min]** Live demo: Ask 3 questions to the API. Show the `/answer` endpoint. Show source citations in the response.
4. **[3 min]** Open LangSmith. Show a trace — every step visible, latency per step, token count, cost per query.
5. **[2 min]** Show RAGAS evaluation running in CI — scores displayed as a pipeline step. Show what happens when a prompt change degrades faithfulness: the deployment is blocked.
6. **[2 min]** Push a catalog update. Watch the GitHub Actions re-indexing pipeline trigger automatically. Show the new version tag in DVC.

---

## Shared MLOps Infrastructure (Both Use Cases)

Both use cases share a common platform layer — build this once, it serves all future models and RAG systems.

| Component | Tool | Serves |
|---|---|---|
| Experiment & Model Registry | MLflow | Churn model + prompt versions |
| Data Versioning | DVC + S3/GCS | Training data + knowledge base |
| Container Registry | Docker + ECR/GCR | Both API images |
| CI/CD | GitHub Actions | Code, models, knowledge base |
| Orchestration | Apache Airflow | Retraining + re-indexing DAGs |
| Monitoring | Evidently AI (ML) + LangSmith (RAG) | Drift + LLM traces |
| Alerting | Prometheus + Grafana (or Datadog) | Latency, error rate, drift alerts |
| Secrets Management | Vault / AWS Secrets Manager | API keys, DB credentials |

---

## Production Readiness Checklist

Before calling either use case "production-grade," verify:

**For Churn Model:**
- [ ] Model version pinned in deployment config (not "latest")
- [ ] Rollback procedure documented and tested (can you go back to v1 in 5 minutes?)
- [ ] Drift alert fires before model degrades below business threshold
- [ ] CI quality gate blocks models below AUC 0.82
- [ ] API `/health` endpoint checks model is loaded and responsive
- [ ] Load test completed (can the API handle 100 req/sec?)
- [ ] Model card written (what data was used, what are known limitations?)

**For RAG Assistant:**
- [ ] Faithfulness score > 0.85 enforced in CI
- [ ] Prompt version tracked in Git — never modified in production directly
- [ ] Knowledge base versioned — can roll back to yesterday's index
- [ ] Source citations returned in every response (auditability)
- [ ] Hallucination guard: model says "I don't know" when context is missing
- [ ] LangSmith traces captured for every production request
- [ ] Re-indexing triggers automatically when source documents change
- [ ] Latency SLA defined (e.g., p95 < 3 seconds)

---

## Team Roles for the Demo Session

| Role | Owns |
|---|---|
| ML Engineer | Training scripts, model evaluation, MLflow |
| DevOps Engineer | Docker, CI/CD pipelines, Kubernetes deployment |
| Data Engineer | DVC, ingestion pipelines, feature store |
| Platform/MLOps | Airflow DAGs, monitoring, drift alerts |

**The point to make:** MLOps is not a new team. It's DevOps + Data + ML working from the same playbook, using the same tools they already know (Git, Docker, CI/CD, Kubernetes) extended for ML artifacts.

---

## Recommended Session Order

1. Start with the **churn model** — it's simpler and maps 1:1 to what DevOps already knows
2. Use the churn demo to establish: versioning, CI/CD, registries, monitoring — the full loop
3. Move to the **RAG assistant** — show that the same principles apply to LLM systems
4. End with the shared platform slide — emphasize: one platform, infinite models
5. Leave 10 minutes for Q&A on toolchain choices and how this maps to their current stack

---

*Built with research from: Google MLOps whitepaper, ZenML LLMOps database (1,182 case studies), Evidently AI, RAGAS docs, LangSmith docs, Kubeflow fraud detection blueprint, MLflow production patterns.*
