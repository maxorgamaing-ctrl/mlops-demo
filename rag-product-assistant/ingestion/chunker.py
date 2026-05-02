"""
Semantic chunking strategy for product documents.
Each chunk retains the product name as a header to prevent
decontextualized retrieval (the RAG equivalent of training-serving skew).
"""

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def chunk_documents(docs: list[Document], config_path: str = "config.yaml") -> list[Document]:
    cfg = load_config(config_path)
    chunking_cfg = cfg["chunking"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_cfg["chunk_size"],
        chunk_overlap=chunking_cfg["chunk_overlap"],
        separators=["\n\n", "\n", ".", " "],
    )

    chunked_docs = []
    for doc in docs:
        product_name = doc.metadata.get("product_name", "")
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            # Prepend product name context to every chunk
            if product_name and not chunk.startswith(f"Product: {product_name}"):
                chunk_text = f"Product: {product_name}\n\n{chunk}"
            else:
                chunk_text = chunk

            chunked_docs.append(Document(
                page_content=chunk_text,
                metadata={**doc.metadata, "chunk_index": i},
            ))

    print(f"Split {len(docs)} documents into {len(chunked_docs)} chunks")
    return chunked_docs
