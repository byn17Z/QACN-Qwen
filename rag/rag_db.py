"""
Description: Knowledge base construction and loading for RAG.
Builds or loads a ChromaDB vector store from raw_content fields in processed data.
Usage:
    from rag.rag_db import build_knowledge_db, load_knowledge_db
    collection = build_knowledge_db("data/processed/sft_data.jsonl", "rag/db", "rag/models/bge-small-zh-v1.5")
    collection = load_knowledge_db("rag/db", "rag/models/bge-small-zh-v1.5")
Dependencies: sentence-transformers, chromadb
"""

import json
import os
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


def load_embedder(model_path: str) -> SentenceTransformer:
    """
    Load the sentence-transformers embedder model.

    Args:
        model_path (str): Path to the embedder model directory.

    Returns:
        SentenceTransformer: The loaded embedder model.
    """
    print(f"Loading embedder from: {model_path}")
    embedder = SentenceTransformer(model_path)
    print(f"Embedder loaded (dim={embedder.get_sentence_embedding_dimension()})")
    return embedder


def build_knowledge_db(
    data_path: str,
    db_path: str,
    embedder_path: str,
    collection_name: str = "knowledge_base",
) -> chromadb.Collection:
    """
    Build a ChromaDB knowledge base from raw_content fields in the processed data.
    Overwrites any existing DB at db_path.

    Args:
        data_path (str): Path to sft_data.jsonl containing raw_content fields.
        db_path (str): Path to store the ChromaDB database.
        embedder_path (str): Path to the embedder model.
        collection_name (str): Name of the ChromaDB collection.

    Returns:
        chromadb.Collection: The populated collection.
    """
    print(f"Building knowledge DB from: {data_path}")

    # Load data
    entries = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            raw_content = entry.get("raw_content", "")
            if raw_content:
                entries.append({"raw_content": raw_content, "instruction": entry.get("instruction", "")})
    print(f"Loaded {len(entries)} entries with raw_content")

    # Load embedder
    embedder = load_embedder(embedder_path)

    # Encode all raw_content
    print("Encoding documents...")
    documents = [e["raw_content"] for e in entries]
    embeddings = embedder.encode(documents, show_progress_bar=True, normalize_embeddings=True)

    # Create ChromaDB (overwrite existing)
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print(f"Removed existing DB at {db_path}")

    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    # Delete collection if it exists (for rebuild)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    # Add documents in batches
    batch_size = 500
    for i in range(0, len(entries), batch_size):
        batch_end = min(i + batch_size, len(entries))
        ids = [f"doc_{j}" for j in range(i, batch_end)]
        collection.add(
            ids=ids,
            embeddings=embeddings[i:batch_end].tolist(),
            documents=documents[i:batch_end],
            metadatas=[{"instruction": entries[j]["instruction"]} for j in range(i, batch_end)],
        )
        print(f"  Added batch {i // batch_size + 1}: {batch_end}/{len(entries)}")

    print(f"Knowledge DB built: {collection.count()} documents at {db_path}")
    return collection


def load_knowledge_db(
    db_path: str,
    embedder_path: str,
    collection_name: str = "knowledge_base",
) -> chromadb.Collection:
    """
    Load an existing ChromaDB knowledge base.

    Args:
        db_path (str): Path to the ChromaDB database directory.
        embedder_path (str): Path to the embedder model (loaded for later use).
        collection_name (str): Name of the ChromaDB collection.

    Returns:
        chromadb.Collection: The loaded collection.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Knowledge DB not found at {db_path}. Run with build_rag_db=true first.")

    print(f"Loading knowledge DB from: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    print(f"Knowledge DB loaded: {collection.count()} documents")

    return collection
