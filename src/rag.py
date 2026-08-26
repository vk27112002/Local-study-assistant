"""Validated hybrid retrieval and grounded answer generation."""

import json
import os
import pickle
from typing import List, Optional

import faiss
import numpy as np

import config
from hybrid_search import bm25_search, reciprocal_rank_fusion
from llm.groq_client import call_llm, has_api_key


_model = None
_index = None
_metadata = None
_bm25_index = None


def get_embedding_model():
    """Load only the shared model; uploaded PDFs do not require a disk index."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, local_files_only=True)
        except OSError:
            _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def _require_artifact(path: str, name: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{name} not found at {os.path.abspath(path)}. Run `python src/ingest.py` first."
        )


def _load_resources():
    global _index, _metadata, _bm25_index
    _require_artifact(config.FAISS_INDEX_PATH, "FAISS index")
    _require_artifact(config.METADATA_PATH, "Chunk metadata")

    if _index is None:
        try:
            _index = faiss.read_index(config.FAISS_INDEX_PATH)
        except Exception as exc:
            raise RuntimeError("FAISS index is unreadable; rebuild it with ingestion.") from exc
    if _metadata is None:
        try:
            with open(config.METADATA_PATH, "r", encoding="utf-8") as stream:
                _metadata = json.load(stream)
        except (OSError, ValueError) as exc:
            raise RuntimeError("Chunk metadata is unreadable; rebuild the index.") from exc
        if not isinstance(_metadata, list):
            raise RuntimeError("Chunk metadata must be a JSON array.")
    if _index.ntotal != len(_metadata):
        raise RuntimeError(
            f"Index/metadata mismatch: {_index.ntotal} vectors and {len(_metadata)} metadata rows. "
            "Rebuild all artifacts with `python src/ingest.py`."
        )
    if _bm25_index is None and os.path.isfile(config.BM25_INDEX_PATH):
        try:
            with open(config.BM25_INDEX_PATH, "rb") as stream:
                candidate = pickle.load(stream)
            if getattr(candidate, "corpus_size", len(_metadata)) == len(_metadata):
                _bm25_index = candidate
        except (OSError, pickle.UnpicklingError, EOFError):
            _bm25_index = None
    return get_embedding_model(), _index, _metadata, _bm25_index


def reset_resource_cache():
    global _model, _index, _metadata, _bm25_index
    _model = _index = _metadata = _bm25_index = None


def _semantic_search(query, index, model, pool_size):
    vector = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise RuntimeError(f"Embedding model returned invalid query shape {vector.shape}.")
    if hasattr(index, "d") and vector.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dimension {vector.shape[1]} does not match index dimension {index.d}."
        )
    scores, indices = index.search(vector, pool_size)
    ranked = [int(value) for value in indices[0] if int(value) >= 0]
    score_map = {
        int(index_id): float(score)
        for index_id, score in zip(indices[0], scores[0])
        if int(index_id) >= 0
    }
    return ranked, score_map


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    index=None,
    metadata=None,
    model=None,
    bm25_index=None,
    use_hybrid: Optional[bool] = None,
) -> List[dict]:
    """Return validated semantic or hybrid retrieval results."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    query = query.strip()
    top_k = config.TOP_K if top_k is None else top_k
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    if index is None or metadata is None or model is None:
        default_model, default_index, default_metadata, default_bm25 = _load_resources()
        model = default_model if model is None else model
        index = default_index if index is None else index
        metadata = default_metadata if metadata is None else metadata
        bm25_index = default_bm25 if bm25_index is None else bm25_index
    if not isinstance(metadata, list):
        raise TypeError("metadata must be a list")
    if index.ntotal != len(metadata):
        raise ValueError("The supplied index and metadata have different chunk counts.")
    if index.ntotal == 0:
        return []

    semantic_pool = min(max(top_k, config.SEMANTIC_CANDIDATE_POOL), index.ntotal)
    semantic_indices, semantic_scores = _semantic_search(query, index, model, semantic_pool)
    hybrid_enabled = config.USE_HYBRID_SEARCH if use_hybrid is None else bool(use_hybrid)

    if hybrid_enabled and bm25_index is not None:
        lexical = bm25_search(
            bm25_index, query, min(max(top_k, config.BM25_CANDIDATE_POOL), index.ntotal)
        )
        lexical_scores = dict(lexical)
        ranking = reciprocal_rank_fusion(semantic_indices, [item[0] for item in lexical])
        results = []
        for index_id, fused_score in ranking[:top_k]:
            if 0 <= index_id < len(metadata):
                chunk = metadata[index_id].copy()
                chunk.update(
                    score=float(fused_score),
                    semantic_score=semantic_scores.get(index_id),
                    bm25_score=lexical_scores.get(index_id),
                    fusion="hybrid",
                )
                results.append(chunk)
        return results

    results = []
    for index_id in semantic_indices[:top_k]:
        if 0 <= index_id < len(metadata):
            chunk = metadata[index_id].copy()
            chunk.update(
                score=semantic_scores[index_id],
                semantic_score=semantic_scores[index_id],
                fusion="semantic-only",
            )
            results.append(chunk)
    return results


def build_prompt(query: str, retrieved_chunks: list) -> str:
    evidence = []
    for number, chunk in enumerate(retrieved_chunks, 1):
        location = (
            f"page {chunk['page']}"
            if chunk.get("page") is not None
            else f"chunk {chunk.get('chunk_id', '?')}"
        )
        evidence.append(
            f"[Evidence {number}: {chunk.get('source', 'Unknown')}, {location}]\n"
            f"{chunk.get('text', '')}"
        )
    context = "\n\n".join(evidence)
    return f"""Answer the question using only the numbered evidence below.

EVIDENCE
{context}

QUESTION
{query}

INSTRUCTIONS
- Cite factual claims inline as [Evidence 1], [Evidence 2], and so on.
- Do not use outside knowledge.
- If evidence is insufficient, reply exactly: "I don't have enough information in the provided documents to answer this."

ANSWER"""


NO_ANSWER = "I don't have enough information in the provided documents to answer this."


def generate_answer(
    query: str,
    top_k: Optional[int] = None,
    index=None,
    metadata=None,
    model=None,
    bm25_index=None,
    use_hybrid: Optional[bool] = None,
) -> dict:
    retrieved = retrieve(
        query,
        top_k,
        index=index,
        metadata=metadata,
        model=model,
        bm25_index=bm25_index,
        use_hybrid=use_hybrid,
    )
    scores = [item["semantic_score"] for item in retrieved if item.get("semantic_score") is not None]
    lexical_scores = [item["bm25_score"] for item in retrieved if item.get("bm25_score") is not None]
    semantic_relevant = bool(scores and max(scores) >= config.MIN_SIMILARITY_SCORE)
    lexical_relevant = bool(lexical_scores and max(lexical_scores) > 0)
    if not retrieved or not (semantic_relevant or lexical_relevant):
        return {"answer": NO_ANSWER, "retrieved_chunks": retrieved, "grounded": False}

    prompt = build_prompt(query.strip(), retrieved)
    if not has_api_key():
        return {
            "answer": "LLM generation is disabled because GROQ_API_KEY is not configured.",
            "retrieved_chunks": retrieved,
            "grounded": False,
            "prompt_preview": prompt,
        }
    return {
        "answer": call_llm(prompt, max_tokens=800),
        "retrieved_chunks": retrieved,
        "grounded": True,
    }


if __name__ == "__main__":
    for result in retrieve("What is the difference between MAR and MCAR?", top_k=3):
        print(result["source"], result.get("semantic_score"))
