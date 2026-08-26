"""Streamlit interface for the generic RAG demo."""

import hashlib
import json
import os
import sys

import streamlit as st


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from hybrid_search import build_bm25_index
from ingest import embed_and_index
from llm.groq_client import has_api_key
from pdf_ingest import ingest_pdf
from rag import generate_answer, get_embedding_model
from summarize import summarize_document


st.set_page_config(page_title="RAG Demo", page_icon="🔎", layout="wide")
st.title("🔎 Retrieval-Augmented Generation Demo")
st.caption(
    "Documents → bounded chunks → local MiniLM embeddings → FAISS + BM25 "
    "→ answers grounded in cited evidence"
)


@st.cache_data(show_spinner=False)
def load_metadata(path: str, modified_ns: int):
    del modified_ns
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, list) else []
    except (OSError, ValueError):
        return []


def persisted_metadata():
    if not os.path.isfile(config.METADATA_PATH):
        return []
    return load_metadata(config.METADATA_PATH, os.stat(config.METADATA_PATH).st_mtime_ns)


def initialize_state():
    defaults = {
        "upload_index": None,
        "upload_metadata": [],
        "upload_bm25": None,
        "upload_name": None,
        "upload_digest": None,
        "summaries": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_pdf_index(uploaded_file):
    chunks, metadata = ingest_pdf(uploaded_file, uploaded_file.name)
    index, _ = embed_and_index(chunks, get_embedding_model())
    return index, metadata, build_bm25_index(chunks)


def evidence_label(item):
    location = (
        f"page {item['page']}"
        if item.get("page") is not None
        else f"chunk {item.get('chunk_id', '?')}"
    )
    return f"{item.get('source', 'Unknown')} — {location}"


def show_evidence(chunks):
    for item in chunks:
        semantic = item.get("semantic_score")
        score = f" · semantic {semantic:.3f}" if semantic is not None else ""
        with st.expander(evidence_label(item) + score):
            st.write(item.get("text", ""))


initialize_state()
disk_metadata = persisted_metadata()
disk_ready = bool(
    disk_metadata
    and os.path.isfile(config.FAISS_INDEX_PATH)
    and os.path.isfile(config.METADATA_PATH)
)


with st.sidebar:
    st.header("Document source")
    uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
    if uploaded is not None:
        content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        if digest != st.session_state.upload_digest:
            if len(content) > 50 * 1024 * 1024:
                st.error("Uploaded PDFs are limited to 50 MB.")
            else:
                try:
                    with st.spinner(f"Indexing {uploaded.name}..."):
                        index, metadata, bm25 = build_pdf_index(uploaded)
                    st.session_state.upload_index = index
                    st.session_state.upload_metadata = metadata
                    st.session_state.upload_bm25 = bm25
                    st.session_state.upload_name = uploaded.name
                    st.session_state.upload_digest = digest
                    st.success(f"Indexed {len(metadata)} chunks.")
                except Exception as exc:
                    st.error(f"PDF processing failed: {exc}")

    options = []
    if disk_ready:
        options.append("Sample documents")
    if st.session_state.upload_index is not None:
        options.append(f"Uploaded: {st.session_state.upload_name}")
    selected = st.radio("Use", options) if options else None
    if selected is None:
        st.warning("Run `python src/ingest.py` or upload a PDF.")

    st.divider()
    top_k = st.slider("Evidence chunks", 1, 10, config.TOP_K)
    use_hybrid = st.checkbox("Hybrid semantic + keyword search", config.USE_HYBRID_SEARCH)
    display_chunks = st.checkbox("Show retrieved evidence", True)
    if has_api_key():
        st.success("Groq generation is configured.")
    else:
        st.warning("Set GROQ_API_KEY in .env to enable generation.")


using_upload = bool(selected and selected.startswith("Uploaded:"))
active_metadata = st.session_state.upload_metadata if using_upload else disk_metadata


def retrieval_args():
    if not using_upload:
        return {"use_hybrid": use_hybrid}
    return {
        "index": st.session_state.upload_index,
        "metadata": st.session_state.upload_metadata,
        "model": get_embedding_model(),
        "bm25_index": st.session_state.upload_bm25,
        "use_hybrid": use_hybrid,
    }


tab_qa, tab_summary, tab_browse = st.tabs(["💬 Ask", "📝 Summarize", "📄 Browse"])


with tab_qa:
    query = st.text_input(
        "Ask a question about the selected documents",
        placeholder="Example: What is the bias-variance tradeoff?",
    )
    if st.button("Ask question", type="primary", disabled=selected is None):
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving and generating..."):
                    result = generate_answer(query, top_k=top_k, **retrieval_args())
                st.subheader("Answer")
                if result.get("grounded"):
                    st.write(result["answer"])
                else:
                    st.warning(result["answer"])
                if display_chunks and result.get("retrieved_chunks"):
                    st.subheader("Evidence")
                    show_evidence(result["retrieved_chunks"])
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")


with tab_summary:
    sources = sorted({item.get("source", "Unknown") for item in active_metadata})
    if using_upload:
        summary_source = st.session_state.upload_name
        summary_metadata = active_metadata
        cache_key = f"upload:{st.session_state.upload_digest}"
    elif sources:
        source_choice = st.selectbox("Summarize", ["All sample documents"] + sources)
        summary_source = source_choice
        summary_metadata = (
            active_metadata
            if source_choice == "All sample documents"
            else [item for item in active_metadata if item.get("source") == source_choice]
        )
        cache_key = f"disk:{source_choice}"
    else:
        summary_source, summary_metadata, cache_key = "No document", [], "none"

    summary_metadata = sorted(
        summary_metadata,
        key=lambda item: (item.get("source", ""), item.get("page", 0), item.get("chunk_id", 0)),
    )
    st.caption(f"{summary_source}: {len(summary_metadata)} indexed chunks")
    if st.button("Generate summary", type="primary", disabled=not summary_metadata):
        if not has_api_key():
            st.error("GROQ_API_KEY is required for summarization.")
        else:
            try:
                with st.spinner("Summarizing all selected chunks..."):
                    result = summarize_document([item["text"] for item in summary_metadata])
                st.session_state.summaries[cache_key] = result
            except Exception as exc:
                st.error(f"Summarization failed: {exc}")

    if cache_key in st.session_state.summaries:
        result = st.session_state.summaries[cache_key]
        st.subheader("Summary")
        st.write(result["summary"])
        if result["num_groups"] > 1:
            with st.expander(f"Show {result['num_groups']} section summaries"):
                for number, section in enumerate(result["partial_summaries"], 1):
                    st.markdown(f"**Section {number}**")
                    st.write(section)


with tab_browse:
    sources = sorted({item.get("source", "Unknown") for item in active_metadata})
    chosen_source = st.selectbox("Document", ["All documents"] + sources, key="browse_source") if sources else "All documents"
    filter_text = st.text_input("Text filter", key="browse_filter").strip().casefold()
    filtered = [
        item for item in active_metadata
        if (chosen_source == "All documents" or item.get("source") == chosen_source)
        and (not filter_text or filter_text in item.get("text", "").casefold())
    ]
    st.caption(f"Showing {len(filtered)} of {len(active_metadata)} chunks")
    for item in filtered[:500]:
        with st.expander(evidence_label(item)):
            st.write(item.get("text", ""))
    if len(filtered) > 500:
        st.info("Only the first 500 matches are shown; narrow the filter to see more.")
