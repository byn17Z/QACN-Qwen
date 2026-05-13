"""
Description: Retrieval and reranking logic for RAG.
Retrieves top-k candidates from ChromaDB with the embedder, then reranks with a cross-encoder.
Usage:
    from rag.rag_retrieve import load_reranker, retrieve
    reranker = load_reranker("rag/models/bge-reranker-base")
    results = retrieve("什么是光合作用?", collection, reranker, embedder, top_k=20, top_n=5)
Dependencies: sentence-transformers, chromadb, transformers
"""

from typing import List

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_reranker(model_path: str) -> CrossEncoder:
    """
    Load the cross-encoder reranker model.

    Args:
        model_path (str): Path to the reranker model directory.

    Returns:
        CrossEncoder: The loaded reranker model.
    """
    print(f"Loading reranker from: {model_path}")
    reranker = CrossEncoder(model_path)
    print("Reranker loaded")
    return reranker


def retrieve(
    query: str,
    collection: chromadb.Collection,
    reranker: CrossEncoder,
    embedder: SentenceTransformer,
    top_k: int = 20,
    top_n: int = 5,
) -> List[str]:
    """
    Retrieve and rerank context documents for a given query.

    Args:
        query (str): The query string (typically the instruction).
        collection (chromadb.Collection): The ChromaDB collection to search.
        reranker (CrossEncoder): The cross-encoder reranker model.
        embedder (SentenceTransformer): The embedding model for encoding the query.
        top_k (int): Number of candidates to retrieve from embedding search.
        top_n (int): Number of final results after reranking.

    Returns:
        List[str]: Top-n context strings, ordered by reranker score (descending).
    """
    # Encode query
    query_embedding = embedder.encode([query], normalize_embeddings=True)

    # Retrieve top-k candidates from ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "distances"],
    )

    candidates = results["documents"][0]
    if not candidates:
        return []

    # Rerank with cross-encoder
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)

    # Sort by score descending and return top-n
    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]
