"""
Hybrid search: combine BM25 (keyword/lexical) search with semantic
(embedding) search using Reciprocal Rank Fusion (RRF).

Why hybrid instead of semantic-only?
Dense embeddings are great at matching *meaning* but can blur specific
tokens into a general "topic" vector. If someone searches for an exact
term -- an acronym, a person's name, a specific formula variable like
"alpha + beta" -- BM25's exact keyword matching often finds the right
chunk when semantic search retrieves something merely "related."
Combining both covers each other's blind spots.

Why Reciprocal Rank Fusion instead of averaging similarity scores?
BM25 scores and cosine similarity scores live on completely different,
incomparable scales (BM25 is an unbounded TF-IDF-weighted score; cosine
similarity is bounded [-1, 1]). Naively averaging them would let whichever
method happens to produce larger numbers dominate. RRF sidesteps this by
using each method's *rank* (1st place, 2nd place, ...) instead of its raw
score -- ranks are always comparable regardless of how the underlying
scores were computed.

RRF formula: score(doc) = sum over each retriever of 1 / (k + rank)
where k is a small constant (60 is the standard default from the original
RRF paper) that discounts the impact of very low ranks.
"""

import re
from typing import List

from rank_bm25 import BM25Okapi


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer. BM25 doesn't need anything fancier
    than this for a portfolio-scale corpus."""
    if not text:
        return []
    return [
        token for token in re.findall(r"\b\w+\b", text.lower())
        if token not in _STOPWORDS
    ]


def build_bm25_index(chunk_texts: List[str]) -> BM25Okapi:
    tokenized = [_tokenize(t) for t in chunk_texts]
    return BM25Okapi(tokenized)


def bm25_search(bm25_index: BM25Okapi, query: str, top_k: int) -> List[tuple]:
    """Returns list of (chunk_index, bm25_score) sorted by score descending."""
    if bm25_index is None or top_k <= 0:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25_index.get_scores(tokens)
    ranked = sorted(
        ((index, float(score)) for index, score in enumerate(scores) if float(score) > 0.0),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]


def reciprocal_rank_fusion(
    semantic_ranked_indices: List[int],
    bm25_ranked_indices: List[int],
    k: int = 60,
) -> List[tuple]:
    """
    Fuse two ranked lists of chunk indices into one combined ranking.
    Each list is ordered best-to-worst; only order matters, not the raw scores.
    Returns list of (chunk_index, fused_score) sorted by fused_score descending.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    fused_scores = {}

    for rank, idx in enumerate(semantic_ranked_indices):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, idx in enumerate(bm25_ranked_indices):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query: str,
    semantic_results: List[dict],
    semantic_all_indices: List[int],
    bm25_index: BM25Okapi,
    metadata: List[dict],
    top_k: int,
    bm25_candidate_pool: int = 20,
) -> List[dict]:
    """
    Combine semantic search results with BM25 results via RRF.

    `semantic_all_indices` should be the full ranked list of chunk indices
    from the semantic search (best-to-worst, before truncating to top_k) so
    RRF has a proper ranking to fuse -- not just the already-truncated top_k.
    """
    bm25_ranked = bm25_search(bm25_index, query, top_k=bm25_candidate_pool)
    bm25_ranked_indices = [idx for idx, _ in bm25_ranked]

    fused = reciprocal_rank_fusion(semantic_all_indices, bm25_ranked_indices)
    top_fused = fused[:top_k]

    results = []
    for idx, fused_score in top_fused:
        chunk = metadata[idx].copy()
        chunk["score"] = fused_score
        chunk["fusion"] = "hybrid (RRF)"
        results.append(chunk)
    return results
