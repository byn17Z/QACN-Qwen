"""
Description: RAG (Retrieval-Augmented Generation) package for QEdpediaCN-Qwen.
Provides knowledge base construction, retrieval, reranking, and testing utilities.
Dependencies: sentence-transformers, chromadb, transformers
"""

from rag.rag_db import build_knowledge_db, load_knowledge_db
from rag.rag_retrieve import load_reranker, retrieve

__all__ = [
    "build_knowledge_db",
    "load_knowledge_db",
    "load_reranker",
    "retrieve",
]
