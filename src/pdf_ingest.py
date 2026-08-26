"""Page-aware native-text extraction for uploaded PDFs."""

import io
import os
from typing import List, Tuple

import pdfplumber

import config
from chunking import split_text


def _clean_text(text: str) -> str:
    if not text:
        return ""
    return "\n".join(line.strip() for line in text.replace("\x00", " ").splitlines() if line.strip())


def extract_pages(pdf_file) -> List[Tuple[int, str]]:
    """Extract non-empty native text page by page from a path or file object."""
    source = pdf_file
    if not isinstance(pdf_file, (str, os.PathLike)):
        try:
            pdf_file.seek(0)
        except (AttributeError, OSError):
            pass
        data = pdf_file.read()
        if not data:
            raise ValueError("The uploaded PDF is empty.")
        source = io.BytesIO(data)
        try:
            pdf_file.seek(0)
        except (AttributeError, OSError):
            pass

    pages = []
    try:
        with pdfplumber.open(source) as pdf:
            for page_number, page in enumerate(pdf.pages, 1):
                text = _clean_text(page.extract_text() or "")
                if text:
                    pages.append((page_number, text))
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc
    return pages


def build_chunks_from_pdf(filename: str, pages: List[Tuple[int, str]]):
    if not filename or not str(filename).strip():
        raise ValueError("filename must be non-empty")
    chunks, metadata = [], []
    for page_number, page_text in pages:
        for chunk_id, piece in enumerate(
            split_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ):
            chunks.append(piece)
            metadata.append(
                {
                    "source": str(filename),
                    "page": int(page_number),
                    "chunk_id": chunk_id,
                    "text": piece,
                }
            )
    return chunks, metadata


def ingest_pdf(pdf_file, filename: str):
    pages = extract_pages(pdf_file)
    if not pages:
        raise ValueError(
            "No extractable text was found. This PDF may be scanned or image-only and requires OCR."
        )
    chunks, metadata = build_chunks_from_pdf(filename, pages)
    if not chunks:
        raise ValueError("The PDF did not produce any usable chunks.")
    return chunks, metadata
