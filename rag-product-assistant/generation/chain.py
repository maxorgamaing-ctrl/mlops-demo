"""
LangChain RAG chain with versioned prompts — compatible with langchain >= 1.0.
Uses Ollama as the LLM backend (no external API keys needed).
Built with LCEL (LangChain Expression Language) instead of the removed RetrievalQA.
"""

import yaml
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from retrieval.reranker import CrossEncoderReranker


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def load_prompt(config_path: str = "config.yaml") -> PromptTemplate:
    cfg = load_config(config_path)
    prompt_path = cfg["generation"]["prompt_path"]
    template = open(prompt_path, encoding="utf-8").read()
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


def build_rag_chain(retriever, config_path: str = "config.yaml"):
    """LCEL chain: retriever | prompt | llm | output parser."""
    cfg = load_config(config_path)
    gen_cfg = cfg["generation"]

    llm = Ollama(
        model=gen_cfg["model"],
        base_url=gen_cfg.get("ollama_base_url", "http://localhost:11434"),
        temperature=gen_cfg["temperature"],
    )

    prompt = load_prompt(config_path)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


class RAGPipeline:
    """
    Full pipeline: hybrid retrieval → cross-encoder reranking → LLM generation.
    """

    def __init__(self, vectorstore, documents: list[Document], config_path: str = "config.yaml"):
        from retrieval.retriever import build_retriever
        self.cfg = load_config(config_path)
        self.retriever = build_retriever(vectorstore, documents, config_path)
        self.reranker = CrossEncoderReranker(top_k=5)
        self.chain = build_rag_chain(self.retriever, config_path)

    def query(self, question: str) -> dict:
        # Fetch docs for source citations separately
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        return {
            "answer": answer,
            "source_documents": docs,
        }
