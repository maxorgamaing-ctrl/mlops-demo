"""
Hybrid retriever: dense (MMR semantic search) + sparse (BM25 keyword).
Combines both for better recall than either approach alone.
"""

import yaml
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List


class EnsembleRetriever(BaseRetriever):
    """Simple weighted ensemble of dense + sparse retrievers (replaces removed langchain.retrievers.EnsembleRetriever)."""
    retrievers: list
    weights: list

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        seen_ids = set()
        merged: List[Document] = []
        for retriever in self.retrievers:
            for doc in retriever.invoke(query):
                doc_id = doc.page_content[:100]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(doc)
        return merged


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_retriever(vectorstore, documents: list[Document], config_path: str = "config.yaml") -> EnsembleRetriever:
    cfg = load_config(config_path)
    ret_cfg = cfg["retrieval"]
    weights = ret_cfg["ensemble_weights"]

    # Dense retriever — semantic similarity with MMR to avoid redundant results
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": ret_cfg["dense_k"],
            "fetch_k": ret_cfg["dense_fetch_k"],
        },
    )

    # Sparse retriever — keyword/BM25 matching
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = ret_cfg["sparse_k"]

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[weights["dense"], weights["sparse"]],
    )
