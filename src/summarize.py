"""Bounded hierarchical map-reduce summarization."""

import time
from typing import Callable, List

from llm.groq_client import call_llm


MAX_GROUP_CHARS = 12_000
MAX_REDUCE_CHARS = 24_000


def _group_chunks(chunks: list, group_size: int = 20, max_chars: int = MAX_GROUP_CHARS):
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    groups, current, current_chars = [], [], 0
    for raw in chunks:
        chunk = str(raw).strip()
        if not chunk:
            continue
        for piece in (chunk[i:i + max_chars] for i in range(0, len(chunk), max_chars)):
            added = len(piece) + (2 if current else 0)
            if current and (len(current) >= group_size or current_chars + added > max_chars):
                groups.append(current)
                current, current_chars = [], 0
            current.append(piece)
            current_chars += len(piece) + (2 if len(current) > 1 else 0)
    if current:
        groups.append(current)
    return groups


def _call(llm: Callable, prompt: str, max_tokens: int):
    value = llm(prompt, max_tokens=max_tokens)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("The language model returned an empty summary.")
    return value.strip()


def summarize_document(
    chunk_texts: list,
    group_size: int = 20,
    *,
    llm: Callable = call_llm,
    pacing_delay: float = 0.25,
) -> dict:
    groups = _group_chunks(chunk_texts, group_size)
    if not groups:
        return {"summary": "No content to summarize.", "num_groups": 0, "partial_summaries": []}

    partials: List[str] = []
    for index, group in enumerate(groups):
        excerpt = "\n\n".join(group)
        partials.append(
            _call(
                llm,
                "Summarize this excerpt accurately. Preserve key facts, arguments, "
                f"qualifications, and conclusions; add no outside information.\n\n{excerpt}\n\nSUMMARY",
                450,
            )
        )
        if pacing_delay > 0 and index < len(groups) - 1:
            time.sleep(pacing_delay)

    level = partials[:]
    while len(level) > 1:
        reduced = []
        for group in _group_chunks(level, group_size=50, max_chars=MAX_REDUCE_CHARS):
            combined = "\n\n".join(f"- {item}" for item in group)
            reduced.append(
                _call(
                    llm,
                    "Combine these consecutive section summaries coherently. Remove repetition, "
                    f"retain qualifications, and add no facts.\n\n{combined}\n\nCOMBINED SUMMARY",
                    800,
                )
            )
        level = reduced
    return {"summary": level[0], "num_groups": len(groups), "partial_summaries": partials}
