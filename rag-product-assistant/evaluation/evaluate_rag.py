"""
RAGAS evaluation suite — runs as a CI quality gate before deploying
any changes to the prompt, retriever, or knowledge base.
Blocks deployment if faithfulness, relevancy, or context precision drop below thresholds.
"""

import argparse
import json
import os
import sys
import yaml
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_test_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    rag_pipeline,
    test_questions_path: str = "evaluation/test_questions.json",
    config_path: str = "config.yaml",
    quick_mode: bool = False,
) -> dict:
    cfg = load_config(config_path)
    qg = cfg["quality_gate"]
    test_data = load_test_questions(test_questions_path)

    if quick_mode:
        test_data = test_data[:3]

    results = []
    for item in test_data:
        response = rag_pipeline.query(item["question"])
        results.append({
            "question": item["question"],
            "answer": response["answer"],
            "contexts": [doc.page_content for doc in response.get("source_documents", [])],
            "ground_truth": item["expected_answer"],
        })

    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores_dict = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }

    print("RAGAS Evaluation Results:")
    for k, v in scores_dict.items():
        print(f"  {k}: {v:.4f}")

    # Quality gates — block deployment if any threshold is not met
    failed = []
    if scores_dict["faithfulness"] < qg["min_faithfulness"]:
        failed.append(f"faithfulness {scores_dict['faithfulness']:.4f} < {qg['min_faithfulness']}")
    if scores_dict["answer_relevancy"] < qg["min_answer_relevancy"]:
        failed.append(f"answer_relevancy {scores_dict['answer_relevancy']:.4f} < {qg['min_answer_relevancy']}")
    if scores_dict["context_precision"] < qg["min_context_precision"]:
        failed.append(f"context_precision {scores_dict['context_precision']:.4f} < {qg['min_context_precision']}")

    if failed:
        print("\nQUALITY GATE FAILED — deployment blocked:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)

    print("\nQuality gate PASSED — safe to deploy.")
    return scores_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-mode", action="store_true", help="Run on first 3 questions only")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Build pipeline for evaluation
    from ingestion.indexer import load_vectorstore
    from ingestion.loader import load_all_documents
    from generation.chain import RAGPipeline

    cfg = load_config(args.config)
    vectorstore = load_vectorstore(cfg)
    documents = load_all_documents()
    pipeline = RAGPipeline(vectorstore, documents, args.config)

    run_evaluation(pipeline, quick_mode=args.quick_mode, config_path=args.config)
