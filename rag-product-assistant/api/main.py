"""
FastAPI app for the Product Knowledge Assistant (RAG).
Endpoints: GET /health, POST /answer
LangSmith tracing is enabled via LANGCHAIN_TRACING_V2=true environment variable.
"""

import os
import yaml
from fastapi import FastAPI, HTTPException

from api.schemas import QuestionRequest, AnswerResponse, SourceDocument, HealthResponse
from ingestion.indexer import run_indexing, load_vectorstore
from ingestion.loader import load_all_documents
from generation.chain import RAGPipeline


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


cfg = load_config()
app = FastAPI(
    title=cfg["serving"]["title"],
    version=cfg["serving"]["version"],
)

# Build pipeline at startup
# In production, the vector store is pre-built by the indexer pipeline
_env = os.environ.get("APP_ENV", "dev")
_vectorstore = load_vectorstore(cfg, env=_env)
_documents = load_all_documents()
_pipeline = RAGPipeline(_vectorstore, _documents)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        doc_count = _vectorstore._collection.count() if hasattr(_vectorstore, "_collection") else -1
    except Exception:
        doc_count = -1
    return HealthResponse(
        status="healthy",
        vector_db_backend=cfg["vector_db"]["backend"],
        documents_indexed=doc_count,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: QuestionRequest):
    try:
        result = _pipeline.query(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    sources = [
        SourceDocument(
            content=doc.page_content[:300],
            product_name=doc.metadata.get("product_name"),
            source=doc.metadata.get("source"),
        )
        for doc in result.get("source_documents", [])
    ]

    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=sources,
        session_id=request.session_id,
    )
