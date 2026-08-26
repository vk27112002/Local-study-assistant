"""Build consistent FAISS, metadata, and BM25 artifacts from text files."""

import hashlib
import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np

import config
from chunking import split_text
from hybrid_search import build_bm25_index


def _text_paths(docs_dir: str):
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {root.resolve()}")
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.name.casefold(),
    )


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_documents(docs_dir: str):
    """Load case-insensitive .txt files and skip byte-identical copies."""
    docs, seen = [], {}
    for path in _text_paths(docs_dir):
        digest = _digest(path)
        if digest in seen:
            print(f"  Skipping duplicate {path.name} (same content as {seen[digest]}).")
            continue
        seen[digest] = path.name
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            docs.append((path.name, stream.read()))
    return docs


def build_chunks(docs):
    all_chunks, metadata = [], []
    for filename, text in docs:
        for chunk_id, piece in enumerate(
            split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ):
            all_chunks.append(piece)
            metadata.append({"source": filename, "chunk_id": chunk_id, "text": piece})
    return all_chunks, metadata


def embed_and_index(chunks, embedding_model):
    if not chunks:
        raise ValueError("No chunks were produced; add readable documents before indexing.")
    embeddings = np.asarray(
        embedding_model.encode(chunks, normalize_embeddings=True), dtype="float32"
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"Embedding model returned shape {embeddings.shape} for {len(chunks)} chunks."
        )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(config.EMBEDDING_MODEL_NAME, local_files_only=True)
    except OSError:
        return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def _write_artifacts(index, metadata, bm25_index):
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    temporary = {
        config.FAISS_INDEX_PATH: config.FAISS_INDEX_PATH + ".tmp",
        config.METADATA_PATH: config.METADATA_PATH + ".tmp",
        config.BM25_INDEX_PATH: config.BM25_INDEX_PATH + ".tmp",
    }
    try:
        faiss.write_index(index, temporary[config.FAISS_INDEX_PATH])
        with open(temporary[config.METADATA_PATH], "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
        with open(temporary[config.BM25_INDEX_PATH], "wb") as stream:
            pickle.dump(bm25_index, stream)
        for destination, source in temporary.items():
            os.replace(source, destination)
    finally:
        for source in temporary.values():
            if os.path.exists(source):
                os.remove(source)


def main():
    print(f"Loading documents from {os.path.abspath(config.DOCS_DIR)} ...")
    documents = load_documents(config.DOCS_DIR)
    chunks, metadata = build_chunks(documents)
    if not chunks:
        raise RuntimeError("No chunks were produced from sample_docs.")
    print(f"  Found {len(documents)} document(s), produced {len(chunks)} chunk(s).")

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' ...")
    index, _ = embed_and_index(chunks, _load_embedding_model())
    bm25_index = build_bm25_index(chunks)
    _write_artifacts(index, metadata, bm25_index)
    print(f"Done. Saved {index.ntotal} vectors to {os.path.abspath(config.INDEX_DIR)}")


if __name__ == "__main__":
    main()
