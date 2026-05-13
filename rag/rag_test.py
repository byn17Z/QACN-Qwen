"""
Description: RAG performance testing — retrieval quality and perplexity comparison.
Tests retrieval quality (Precision@k, Recall@k, MRR) and compares model perplexity
across three context modes: no context, labeled context, and RAG-retrieved context.
Usage: python rag/rag_test.py --config configs/config.yaml --mode validation --timestamp <ts>
Dependencies: sentence-transformers, chromadb, transformers, torch, tqdm
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.utils import load_config
from rag.rag_db import load_knowledge_db, load_embedder
from rag.rag_retrieve import load_reranker, retrieve


def test_retrieval_quality(
    dataset,
    collection,
    reranker,
    embedder,
    top_k: int = 20,
    top_n: int = 5,
) -> dict:
    """
    Test retrieval quality by checking if labeled raw_content appears in retrieved results.

    Args:
        dataset: The evaluation dataset with instruction and raw_content fields.
        collection: The ChromaDB collection.
        reranker: The cross-encoder reranker.
        embedder: The embedding model.
        top_k (int): Number of candidates from embedding search.
        top_n (int): Number of final results after reranking.

    Returns:
        dict: Retrieval metrics (Precision@k, Recall@k, MRR).
    """
    print(f"Testing retrieval quality on {len(dataset)} examples...")

    precisions = []
    recalls = []
    mrrs = []

    for example in tqdm(dataset, desc="Retrieval quality"):
        instruction = example.get("instruction", "")
        raw_content = example.get("raw_content", "")

        if not raw_content:
            continue

        # Retrieve using instruction as query
        retrieved_docs = retrieve(instruction, collection, reranker, embedder, top_k=top_k, top_n=top_n)

        # Check if labeled raw_content appears in retrieved results
        # Use substring matching for flexibility
        found = False
        rank = 0
        for i, doc in enumerate(retrieved_docs):
            if raw_content.strip() in doc.strip() or doc.strip() in raw_content.strip():
                found = True
                rank = i + 1
                break

        # Precision@k: 1 if found in top_n, else 0
        precisions.append(1.0 if found else 0.0)
        # Recall@k: same as precision for single relevant doc
        recalls.append(1.0 if found else 0.0)
        # MRR: 1/rank if found, else 0
        mrrs.append(1.0 / rank if found else 0.0)

    metrics = {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "total_examples": len(dataset),
        "evaluated_examples": len(precisions),
        "top_k": top_k,
        "top_n": top_n,
    }

    print(f"\nRetrieval Quality Metrics:")
    print(f"  Precision@{top_n}: {metrics['precision_at_k']:.4f}")
    print(f"  Recall@{top_n}: {metrics['recall_at_k']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")

    return metrics


def calculate_perplexity_with_context(
    model,
    tokenizer,
    dataset,
    max_length: int = 512,
    context_mode: str = "none",
    rag_components: dict = None,
) -> tuple:
    """
    Calculate perplexity with different context modes.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        dataset: The evaluation dataset.
        max_length (int): Maximum sequence length.
        context_mode (str): One of "none", "labeled", "rag".
            - "none": No context (baseline)
            - "labeled": Use labeled raw_content
            - "rag": Use RAG-retrieved context
        rag_components (dict): Dict with 'collection', 'reranker', 'embedder' for RAG mode.

    Returns:
        tuple: (perplexity, mean_nll)
    """
    model.eval()
    nlls = []

    print(f"Evaluating perplexity (context_mode={context_mode})...")
    for example in tqdm(dataset):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Build context based on mode
        context = ""
        if context_mode == "labeled":
            context = example.get("raw_content", "")
        elif context_mode == "rag" and rag_components:
            retrieved = retrieve(
                instruction,
                rag_components["collection"],
                rag_components["reranker"],
                rag_components["embedder"],
            )
            context = "\n".join(retrieved) if retrieved else ""

        # Build full text
        if context:
            full_text = f"Context: {context}\nInstruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
        else:
            full_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"

        encodings = tokenizer(full_text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = encodings.input_ids.to(model.device)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item(), torch.stack(nlls).mean().item()


def main(config_path: str, mode: str, timestamp: str):
    config = load_config(config_path)
    model_config = config["model"]
    training_config = config["training"]
    logging_config = config["logging"]
    rag_config = config.get("rag", {})

    # Load RAG components
    embedder_path = rag_config.get("embedder_path", "rag/models/bge-small-zh-v1.5")
    reranker_path = rag_config.get("reranker_path", "rag/models/bge-reranker-base")
    db_path = rag_config.get("db_path", "rag/db")
    top_k = int(rag_config.get("retrieve_top_k", 20))
    top_n = int(rag_config.get("retrieve_top_n", 5))

    print("Loading RAG components...")
    collection = load_knowledge_db(db_path, embedder_path)
    embedder = load_embedder(embedder_path)
    reranker = load_reranker(reranker_path)

    # Load evaluation dataset
    data_filename = "val.jsonl" if mode == "validation" else "test.jsonl"
    eval_data_path = os.path.join(config["dataset"]["processed_data_path"], data_filename)
    print(f"Loading {mode} set: {eval_data_path}")
    eval_dataset = load_dataset("json", data_files=eval_data_path, split="train")

    # Test 1: Retrieval Quality
    print("\n=== Test 1: Retrieval Quality ===")
    retrieval_metrics = test_retrieval_quality(eval_dataset, collection, reranker, embedder, top_k=top_k, top_n=top_n)

    # Test 2: Perplexity Comparison
    print("\n=== Test 2: Perplexity Comparison ===")

    # Load model
    model_path = model_config["current_model_path"]
    if not os.path.exists(model_path) or not os.listdir(model_path):
        model_path = model_config["base_model_path"]

    print(f"Loading model for evaluation: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    max_length = int(training_config["context_length"])
    rag_components = {"collection": collection, "reranker": reranker, "embedder": embedder}

    # Mode 1: No context (baseline)
    ppl_none, loss_none = calculate_perplexity_with_context(
        model, tokenizer, eval_dataset, max_length=max_length, context_mode="none"
    )

    # Mode 2: Labeled context
    ppl_labeled, loss_labeled = calculate_perplexity_with_context(
        model, tokenizer, eval_dataset, max_length=max_length, context_mode="labeled"
    )

    # Mode 3: RAG-retrieved context
    ppl_rag, loss_rag = calculate_perplexity_with_context(
        model, tokenizer, eval_dataset, max_length=max_length, context_mode="rag", rag_components=rag_components
    )

    perplexity_results = {
        "no_context": {"perplexity": ppl_none, "loss": loss_none},
        "labeled_context": {"perplexity": ppl_labeled, "loss": loss_labeled},
        "rag_context": {"perplexity": ppl_rag, "loss": loss_rag},
    }

    print(f"\nPerplexity Comparison:")
    print(f"  No context:      {ppl_none:.4f}")
    print(f"  Labeled context: {ppl_labeled:.4f}")
    print(f"  RAG context:     {ppl_rag:.4f}")

    # Save results
    results = {
        "mode": mode,
        "timestamp": timestamp,
        "retrieval_metrics": retrieval_metrics,
        "perplexity_comparison": perplexity_results,
    }

    output_dir = logging_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rag_test_stats.json")

    existing_results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except Exception:
                existing_results = []

    existing_results.append(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG performance testing")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["validation", "test"], required=True, help="Evaluation mode")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp string")
    args = parser.parse_args()

    main(args.config, args.mode, args.timestamp)
