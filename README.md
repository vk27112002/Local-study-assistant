# RAG from Scratch — A Beginner-Friendly, Interview-Defensible Project

A minimal Retrieval-Augmented Generation (RAG) pipeline built to be **explainable**,
not just runnable. Every component was chosen deliberately, and the reasoning
behind each choice is documented below — this is what interviewers actually probe.

## Architecture

```
Documents (.txt)
      │
      ▼
Chunking (boundary-aware splitter, max 500 chars, 50 overlap)
      │
      ▼
Embedding (sentence-transformers/all-MiniLM-L6-v2, local, free)
      │
      ▼
FAISS Index (IndexFlatIP — cosine similarity via normalized vectors)
      │
      ▼
Retrieval (top-k similarity search + minimum-score guard)
      │
      ▼
Prompt construction (context + grounding instructions)
      │
      ▼
LLM generation (configurable Groq model) → grounded answer + cited evidence
```

## Features

- **Hybrid search** — combines semantic (embedding) search with BM25 keyword search via Reciprocal Rank Fusion, toggleable in the sidebar. Catches exact-term queries that pure semantic search can miss.
- **Ask questions** against either the bundled `sample_docs/` or an uploaded PDF, with grounded, cited answers.
- **Upload a PDF** directly in the app — extracted page-by-page, chunked, and embedded into an in-memory index for that session (nothing is written to disk).
- **Summarize** — a map-reduce summary of the whole document, useful for long PDFs. Expand to see intermediate per-section summaries.
- **Browse chunks** — inspect every indexed chunk, with a text filter, to sanity-check retrieval quality.

## Project structure

```
rag_project/
├── .env.example               # copy to .env and add your Groq key
├── .gitignore                  # keeps .env and caches out of version control
├── sample_docs/                # source documents (swap these for your own)
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   └── groq_client.py       # ALL LLM API call logic + config lives here (single source of truth)
│   ├── config.py                # every tunable knob in one place
│   ├── chunking.py              # custom recursive text splitter
│   ├── ingest.py                 # build the FAISS + BM25 indexes from sample_docs/
│   ├── pdf_ingest.py             # page-aware PDF extraction + chunking
│   ├── hybrid_search.py          # BM25 + Reciprocal Rank Fusion
│   ├── rag.py                    # retrieve() + generate_answer()
│   ├── summarize.py              # map-reduce document summarization
│   └── evaluate.py               # retrieval hit-rate@k eval, incl. hybrid vs semantic-only comparison
├── app.py                     # Streamlit demo UI (Q&A / Summarize / Browse)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

On Windows, use `py -3.11` in place of `python` below if `python` is not on
your PATH. Uploaded PDFs must contain a native text layer; scanned/image-only
PDFs need OCR before upload.

**Set your Groq API key (one-time, works forever after):**
1. Copy `.env.example` to a new file named exactly `.env` (same folder)
2. Get a free key at console.groq.com (no credit card needed)
3. Open `.env` and paste your key: `GROQ_API_KEY=gsk_your_key_here`

This is better than `export`/`$env:` in your terminal, which only lasts for
that one session — `.env` is loaded automatically every time, in any
terminal, by every script in this project. It's also git-ignored by
default, so it never gets committed or accidentally shared.

```bash
# Build the index (run once, and again whenever sample_docs/ changes)
python src/ingest.py

# Check retrieval quality before trusting the generation layer
python src/evaluate.py

# Launch the demo
streamlit run app.py
```

## Design decisions (and why) — your interview cheat sheet

**Why hybrid search (semantic + BM25) instead of semantic-only?**
Dense embeddings are great at matching *meaning* but can blur specific terms
into a general "topic" vector. If someone asks about an exact acronym, name,
or formula variable (e.g. "EGARCH" or "MICE"), semantic search sometimes
retrieves something merely topically related rather than the chunk containing
that literal term. BM25 (classic TF-IDF-style keyword search) catches exactly
those cases. Combining both via Reciprocal Rank Fusion (RRF) covers each
method's blind spot without needing to compare their raw scores directly —
RRF works purely on *rank position* (1st place, 2nd place...) from each
method, which sidesteps the problem that BM25 scores and cosine similarities
live on completely different, incomparable scales. Run
`python src/evaluate.py` to see a measured hit-rate comparison between hybrid
and semantic-only on this project's eval set — the actual before/after number
is far more convincing in an interview than just describing the technique.

**Why chunk at all, instead of embedding whole documents?**
Embedding models compress text into a fixed-size vector. A whole document
crammed into one vector loses fine-grained detail — a single embedding can't
represent ten different facts precisely. Chunking keeps each vector focused
on one localized piece of information, so retrieval can pull the *specific*
passage relevant to a question instead of "the whole document, vaguely."

**Why chunk_size=500 with 50-char overlap?**
Too small and a chunk loses surrounding context (e.g. a formula split from
its explanation). Too large and irrelevant text dilutes the embedding,
hurting retrieval precision. Overlap prevents a sentence or fact from being
severed exactly at a chunk boundary. These are starting values — in a real
project you'd sweep a few sizes and check the eval hit-rate for each.

**Why a local embedding model (MiniLM) instead of an API-based one?**
No per-query cost, no external dependency for the retrieval step, runs on
CPU, and it's fast enough to embed a corpus in seconds. The tradeoff: a
larger API-hosted embedding model would likely retrieve slightly better —
this is a genuine accuracy-vs-cost decision, not a "no reason" default.

**Why FAISS instead of a managed vector DB (Pinecone, Weaviate, etc.)?**
For a portfolio-scale corpus, FAISS running in-memory is simpler, free, and
transparent — you can explain exactly what `IndexFlatIP` is doing (brute-force
cosine similarity search) instead of treating a hosted service as a black box.
At production scale with millions of vectors you'd reach for an approximate
nearest-neighbor index (HNSW, IVF) or a managed DB — worth mentioning that
tradeoff explicitly if asked "would this scale?"

**Why the minimum similarity score guard before calling the LLM?**
If nothing in the corpus is actually relevant to the question, retrieval
still returns *something* (the "top 3" by definition, even if none of them
are good matches). Without a score threshold, the LLM would try to answer
from irrelevant context and likely hallucinate. This guard is a cheap,
explicit hallucination defense — and it saves an API call.

**Why explicitly instruct the LLM to say "I don't know"?**
This is the actual point of RAG. Without grounding instructions, an LLM will
often answer confidently from its own training knowledge even when told to
use "the context" — instruction alone isn't a hard guarantee, but it's the
first and cheapest layer of defense, paired with the retrieval score guard above.

**Why is there an eval script?**
Because "it works when I tried it" is not evidence. `evaluate.py` measures
hit-rate@k against a small set of known question→source pairs. This is the
single detail that separates "I followed a tutorial" from "I understand how
to validate an ML system" in an interview.

**Why chunk PDF pages individually instead of concatenating the whole PDF first?**
Chunking per-page means every retrieved chunk carries an exact, citable page
number. The tradeoff: a fact split exactly across a page boundary could end
up split across two chunks. For a document Q&A tool where "which page was
that on?" matters, this tradeoff is worth it.

**Why map-reduce for summarization instead of one big prompt?**
A long PDF's full text can exceed the model's context window, and even within
the window, models tend to skim unevenly over very long inputs. Map-reduce
summarizes small groups of chunks first (the "map" step), then combines those
partial summaries into one coherent final summary (the "reduce" step) — the
same pattern LangChain's summarization chain uses, implemented directly here
so it's easy to explain and tune (e.g. the `group_size` parameter).

**Why keep the uploaded-PDF index in memory instead of writing it to disk?**
It's session-scoped and disposable by design — the person uploading shouldn't
have to worry about their document persisting anywhere. It also means the
sample-docs index used for the eval script never gets polluted by whatever
gets uploaded during a demo.

## Extending this project (good next steps for your portfolio)

- Swap `sample_docs/` for a real corpus (your GATE ST notes, papers, etc.) — the pdf skill in this workspace can help you extract text from PDFs first
- Add a re-ranker (e.g. cross-encoder) after initial retrieval to improve precision
- Log retrieval scores over many real queries to tune `MIN_SIMILARITY_SCORE` empirically
- Add citation-level highlighting: show *which sentence* in the retrieved chunk supports the answer
- Compare MiniLM against a larger embedding model and quantify the hit-rate difference — a genuinely interesting mini-experiment to put in your README

## Known limitation (be upfront about this if asked)

This is a "naive RAG" — no re-ranking, no query rewriting, no handling of
multi-hop questions that need info from multiple documents synthesized
together. That's fine for a portfolio piece as long as you can name the
limitation yourself rather than have an interviewer catch you off guard.
cd "C:\Users\vaibh\Desktop\Placement prep\project\RAG\resume_optimizer"
.\venv\Scripts\Activate.ps1
streamlit run app.py