"""
Document loader — reads product catalog JSON, FAQs CSV, and PDF manuals.
Returns a list of LangChain Document objects ready for chunking.
"""

import json
import os
import csv
from langchain_core.documents import Document


def load_product_catalog(path: str) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        products = json.load(f)

    docs = []
    for product in products:
        # Main product description
        content = (
            f"Product: {product['name']}\n"
            f"Category: {product['category']}\n"
            f"Description: {product['description']}\n"
            f"Price: ${product['price_usd']}\n"
        )
        if product.get("specs"):
            specs_text = "\n".join(f"  {k}: {v}" for k, v in product["specs"].items())
            content += f"Specifications:\n{specs_text}\n"
        if product.get("compatibility"):
            content += f"Compatibility: {', '.join(product['compatibility'])}\n"

        docs.append(Document(
            page_content=content,
            metadata={"source": "product_catalog", "product_id": product["id"], "product_name": product["name"]},
        ))

        # FAQs as separate documents for better retrieval granularity
        for faq in product.get("faqs", []):
            faq_content = (
                f"Product: {product['name']}\n"
                f"Q: {faq['q']}\n"
                f"A: {faq['a']}"
            )
            docs.append(Document(
                page_content=faq_content,
                metadata={"source": "faq", "product_id": product["id"], "product_name": product["name"]},
            ))

    return docs


def load_faqs_csv(path: str) -> list[Document]:
    docs = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = f"Product ID: {row['product_id']}\nQ: {row['question']}\nA: {row['answer']}"
            docs.append(Document(
                page_content=content,
                metadata={"source": "faqs_csv", "product_id": row["product_id"]},
            ))
    return docs


def load_pdf_manuals(directory: str) -> list[Document]:
    """Load PDF user manuals. Requires pypdf."""
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    if not os.path.isdir(directory):
        return docs

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = "manual"
                page.metadata["filename"] = filename
            docs.extend(pages)

    return docs


def load_all_documents(data_dir: str = "data/raw") -> list[Document]:
    docs = []
    catalog_path = os.path.join(data_dir, "product_catalog.json")
    faqs_path = os.path.join(data_dir, "faqs.csv")
    manuals_dir = os.path.join(data_dir, "manuals")

    if os.path.exists(catalog_path):
        docs += load_product_catalog(catalog_path)
    if os.path.exists(faqs_path):
        docs += load_faqs_csv(faqs_path)
    docs += load_pdf_manuals(manuals_dir)

    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs
