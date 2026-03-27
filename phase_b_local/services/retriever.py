"""
Phase B - FAISS Retriever Service (CPU Only)
=============================================
Loads FAISS index + document store for semantic search.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VDB_DIR = os.path.join(PROJECT_ROOT, "vector_db")
INDEX_PATH = os.path.join(VDB_DIR, "college.index")
DOCS_PATH = os.path.join(VDB_DIR, "documents.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CONTEXT_WORDS = 800


def load_retriever():
    """Load FAISS index, documents, and embedding model."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}\n"
            "Run 07_build_vectordb.py in Colab first."
        )
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Documents file not found: {DOCS_PATH}\n"
            "Run 07_build_vectordb.py in Colab first."
        )

    print(f"  Loading FAISS index from {VDB_DIR}...")
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    print(f"  Loading embedding model: {EMBED_MODEL}...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    return index, documents, embed_model


def retrieve(query, index, documents, embed_model, top_k=6, max_distance=1.5):
    """Retrieve top-k relevant document chunks for a query.
    
    Filters out chunks with FAISS L2 distance > max_distance
    to prevent irrelevant results from reaching the LLM.
    """
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(documents) and dist < max_distance:
            results.append(documents[idx])
    return results


def format_context(chunks):
    """Join chunks with separator, truncating to max word count."""
    combined = "\n---\n".join(chunks)
    words = combined.split()
    if len(words) > MAX_CONTEXT_WORDS:
        combined = " ".join(words[:MAX_CONTEXT_WORDS])
    return combined
