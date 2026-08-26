"""
Tiny retrieval evaluation harness.

This is the single most interview-impressive file in this project.
Anyone can say "I built a RAG app." Very few can say "I measured
whether retrieval actually works." Even a 10-question eval set with
known correct sources is worth more than an unverified demo.

Metric used: hit-rate@k -- for each test question, did the correct
source document appear anywhere in the top-k retrieved chunks?

Run:
    python src/evaluate.py
"""

from rag import retrieve
import config

# Each entry: (question, expected_source_filename)
# Expand this as you add more documents.
EVAL_SET = [
    ("What does high variance in a model usually indicate?", "bias_variance_tradeoff.txt"),
    ("How do you decompose expected prediction error?", "bias_variance_tradeoff.txt"),
    ("What does bagging do to reduce error?", "bias_variance_tradeoff.txt"),
    ("What is the condition on alpha and beta for a GARCH(1,1) model to be stationary?", "garch_models.txt"),
    ("What test can be used to check for structural breaks before fitting a GARCH model?", "garch_models.txt"),
    ("What is the difference between ARCH(1) and GARCH(1,1)?", "garch_models.txt"),
    ("What is the difference between MAR and MCAR?", "handling_missing_data.txt"),
    ("What statistical test can check if missingness is associated with a categorical variable?", "handling_missing_data.txt"),
    ("What is MICE?", "handling_missing_data.txt"),
    ("What is a rule of thumb for when to be cautious about imputing a variable?", "handling_missing_data.txt"),
]


def run_eval(top_k: int = None, use_hybrid: bool = None):
    """
    Run the eval set once with a given retrieval mode.
    Pass use_hybrid=True/False to force a mode; None uses whatever config.USE_HYBRID_SEARCH is set to.
    """
    top_k = config.TOP_K if top_k is None else top_k
    if top_k < 1:
        raise ValueError("top_k must be positive")
    hybrid_enabled = config.USE_HYBRID_SEARCH if use_hybrid is None else use_hybrid
    mode_label = "hybrid (semantic + BM25)" if hybrid_enabled else "semantic-only"
    hits = 0
    print(f"Running retrieval eval (top_k={top_k}, mode={mode_label}) on {len(EVAL_SET)} questions...\n")

    for question, expected_source in EVAL_SET:
        results = retrieve(question, top_k=top_k, use_hybrid=hybrid_enabled)
        retrieved_sources = [r["source"] for r in results]
        hit = expected_source in retrieved_sources
        hits += hit

        status = "HIT " if hit else "MISS"
        top_score = results[0]["score"] if results else float("nan")
        print(f"[{status}] score={top_score:.4f}  Q: {question}")
        if not hit:
            print(f"        expected: {expected_source}, got: {retrieved_sources}")

    hit_rate = hits / len(EVAL_SET)
    print(f"\nHit-rate@{top_k} ({mode_label}): {hits}/{len(EVAL_SET)} = {hit_rate:.1%}")
    return hit_rate


def compare_hybrid_vs_semantic(top_k: int = None):
    """
    Run the eval set twice -- once with hybrid search, once semantic-only --
    and print both hit-rates side by side. This is the single most convincing
    piece of evidence you can put in a README: an actual measured before/after
    for the upgrade, not just a claim that hybrid search "should help."
    """
    print("=" * 60)
    print("SEMANTIC-ONLY")
    print("=" * 60)
    semantic_rate = run_eval(top_k=top_k, use_hybrid=False)

    print()
    print("=" * 60)
    print("HYBRID (semantic + BM25 via RRF)")
    print("=" * 60)
    hybrid_rate = run_eval(top_k=top_k, use_hybrid=True)

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Semantic-only hit-rate: {semantic_rate:.1%}")
    print(f"Hybrid hit-rate:        {hybrid_rate:.1%}")
    delta = hybrid_rate - semantic_rate
    print(f"Delta: {delta:+.1%}")
    return semantic_rate, hybrid_rate


if __name__ == "__main__":
    compare_hybrid_vs_semantic()
