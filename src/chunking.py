"""Small, dependency-free text chunker with bounded overlap."""

from typing import List


_SEPARATORS = ("\n\n", "\n", ". ", " ")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text at natural boundaries while enforcing a hard size limit."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            minimum_break = start + max(1, chunk_size // 2)
            for separator in _SEPARATORS:
                position = text.rfind(separator, minimum_break, hard_end)
                if position != -1:
                    end = position + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(0, end - chunk_overlap)
        start = end if next_start <= start else next_start
    return chunks
