"""
Phase A - Step 7: Build FAISS Vector Database
===============================================
Embeds all chunks + seed facts using sentence-transformers
and builds a FAISS index for retrieval.

Run on: Google Colab
Input : data/chunks/all_chunks.json + data/seed_facts.txt
Output: vector_db/college.index + vector_db/documents.pkl
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "chunks", "all_chunks.json")
SEED_PATH = os.path.join(PROJECT_ROOT, "data", "seed_facts.txt")
VDB_DIR = os.path.join(PROJECT_ROOT, "vector_db")
INDEX_PATH = os.path.join(VDB_DIR, "college.index")
DOCS_PATH = os.path.join(VDB_DIR, "documents.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def load_documents():
    """Load chunks + seed facts into a single list."""
    documents = []

    # Chunks
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            documents.append(c["text"])
        print(f"  Loaded {len(chunks)} chunks")
    else:
        print(f"  [WARN] {CHUNKS_PATH} not found")

    # Seed facts
    if os.path.exists(SEED_PATH):
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            facts = [l.strip() for l in f if l.strip()]
        documents.extend(facts)
        print(f"  Loaded {len(facts)} seed facts")
    else:
        print(f"  [WARN] {SEED_PATH} not found")

    return documents


def build_index(documents, model):
    """Embed documents and build FAISS index."""
    print(f"  Embedding {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True,
                              batch_size=32, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)
    return index


def verify(index, documents, model):
    """Run test queries to verify retrieval quality."""
    test_queries = [
        "What courses are offered at Gojan?",
        "What is the TNEA code?",
        "Where is the college located?",
        "Does the college have hostel?",
        "Who is the chairman?",
    ]

    print()
    print("  Verification Queries:")
    print("  " + "-" * 50)
    for query in test_queries:
        vec = model.encode([query]).astype("float32")
        D, I = index.search(vec, 3)
        print(f"  Q: {query}")
        for rank, idx in enumerate(I[0], 1):
            if 0 <= idx < len(documents):
                snippet = documents[idx][:80].replace("\n", " ")
                print(f"     [{rank}] {snippet}...")
        print()


def main():
    os.makedirs(VDB_DIR, exist_ok=True)

    print("=" * 55)
    print("  FAISS Vector Database Builder")
    print("=" * 55)
    print()

    documents = load_documents()
    if not documents:
        print("  ERROR: No documents to index!")
        return

    print(f"\n  Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    index = build_index(documents, model)

    # Save
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"\n  Saved: {INDEX_PATH}")
    print(f"  Saved: {DOCS_PATH}")
    print(f"  Total: {index.ntotal} documents indexed")

    verify(index, documents, model)

    print("=" * 55)
    print(f"  Vector DB built: {index.ntotal} documents indexed")
    print("=" * 55)


if __name__ == "__main__":
    main()
