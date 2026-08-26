"""
Central config for the RAG project.
Keeping every tunable knob in one place makes it easy to explain
your choices in an interview ("I set chunk_size=500 because...").
"""

import os

# --- Paths ---
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index_store")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_metadata.json")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25.pkl")

# --- Chunking ---
CHUNK_SIZE = 500       # characters per chunk (~100-120 tokens for English text)
CHUNK_OVERLAP = 50     # overlap between consecutive chunks, prevents cutting facts in half

# --- Embedding model ---
# Local, free, runs on CPU. 384-dim output. Good baseline for a portfolio project.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 3                    # how many chunks to retrieve per query
MIN_SIMILARITY_SCORE = 0.30  # below this, we treat retrieval as "no relevant context found"
                              # (cosine similarity, roughly calibrated for MiniLM embeddings)
USE_HYBRID_SEARCH = True     # combine semantic (FAISS) + keyword (BM25) search via RRF
SEMANTIC_CANDIDATE_POOL = 20 # how many semantic results to feed into RRF fusion (before truncating to TOP_K)
BM25_CANDIDATE_POOL = 20     # same, for BM25
