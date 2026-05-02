"""
Indexer — embeds chunked documents and stores them in the vector database.
Run this whenever the product catalog changes (triggered by CI/CD reindex.yml).
"""

import argparse
import os
import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ingestion.loader import load_all_documents
from ingestion.chunker import chunk_documents


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_embeddings(cfg: dict) -> HuggingFaceEmbeddings:
    """Use local sentence-transformers model — no API key required."""
    model_name = cfg["embeddings"].get("local_model", "all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(chunks: list[Document], cfg: dict, env: str = "dev"):
    db_cfg = cfg["vector_db"]
    embeddings = get_embeddings(cfg)

    persist_dir = db_cfg["chroma_persist_dir"]
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=db_cfg["collection_name"],
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB at {persist_dir}")
    return vectorstore


def load_vectorstore(cfg: dict, env: str = "dev"):
    db_cfg = cfg["vector_db"]
    embeddings = get_embeddings(cfg)
    return Chroma(
        persist_directory=db_cfg["chroma_persist_dir"],
        embedding_function=embeddings,
        collection_name=db_cfg["collection_name"],
    )


def run_indexing(data_dir: str = "data/raw", env: str = "dev", config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    docs = load_all_documents(data_dir)
    chunks = chunk_documents(docs, config_path)
    vectorstore = build_vectorstore(chunks, cfg, env)
    return vectorstore, docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dev", choices=["dev", "production"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_indexing(data_dir=args.data_dir, env=args.env, config_path=args.config)
