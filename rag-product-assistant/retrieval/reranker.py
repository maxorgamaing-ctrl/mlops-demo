"""
Cross-encoder reranker — improves retrieval precision by re-scoring
the top-k candidates from the ensemble retriever.
"""

from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    The cross-encoder scores each (query, document) pair jointly,
    which is more accurate than bi-encoder cosine similarity.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return documents

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = [doc for _, doc in scored_docs[: self.top_k]]
        for i, (score, doc) in enumerate(scored_docs[: self.top_k]):
            reranked[i].metadata["rerank_score"] = float(score)

        return reranked
