import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunking import split_text
from hybrid_search import bm25_search, build_bm25_index, reciprocal_rank_fusion
from ingest import build_chunks, embed_and_index, load_documents
from pdf_ingest import build_chunks_from_pdf
from summarize import _group_chunks, summarize_document
from llm import groq_client
import rag


class ChunkingTests(unittest.TestCase):
    def test_chunk_bound_and_overlap(self):
        chunks = split_text("A sentence with words. " * 30, 100, 15)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 100 for chunk in chunks))
        self.assertTrue(any(chunks[i - 1][-10:].strip() in chunks[i] for i in range(1, len(chunks))))

    def test_invalid_configuration(self):
        for size, overlap in [(0, 0), (10, -1), (10, 10)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                split_text("text", size, overlap)


class IngestionTests(unittest.TestCase):
    def test_loading_is_case_insensitive_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.TXT").write_text("same", encoding="utf-8")
            (root / "b.txt").write_text("same", encoding="utf-8")
            documents = load_documents(directory)
        self.assertEqual(len(documents), 1)

    def test_empty_embedding_input_is_rejected(self):
        with self.assertRaises(ValueError):
            embed_and_index([], Mock())

    def test_build_chunks_has_consistent_metadata(self):
        chunks, metadata = build_chunks([("doc.txt", "word " * 200)])
        self.assertEqual(len(chunks), len(metadata))
        self.assertTrue(all(item["source"] == "doc.txt" for item in metadata))
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))


class HybridTests(unittest.TestCase):
    def test_zero_score_results_are_removed(self):
        index = build_bm25_index(["alpha beta", "gamma delta", "epsilon zeta"])
        self.assertEqual(bm25_search(index, "missing", 3), [])
        self.assertEqual(bm25_search(index, "what is the missing", 3), [])
        self.assertEqual([item[0] for item in bm25_search(index, "alpha", 3)], [0])

    def test_rrf_validation(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([0], [1], k=-1)


class FakeModel:
    def encode(self, values, normalize_embeddings=True):
        return np.asarray([[1.0, 0.0] for _ in values], dtype="float32")


class FakeIndex:
    ntotal = 2
    d = 2

    def search(self, vector, count):
        return (
            np.asarray([[0.9, 0.6]], dtype="float32")[:, :count],
            np.asarray([[0, 1]], dtype="int64")[:, :count],
        )


class RagTests(unittest.TestCase):
    metadata = [
        {"source": "a.txt", "chunk_id": 0, "text": "first"},
        {"source": "b.txt", "chunk_id": 0, "text": "second"},
    ]

    def test_retrieval_validation_and_semantic_results(self):
        with self.assertRaises(ValueError):
            rag.retrieve(" ", index=FakeIndex(), metadata=self.metadata, model=FakeModel())
        results = rag.retrieve(
            "question",
            top_k=2,
            index=FakeIndex(),
            metadata=self.metadata,
            model=FakeModel(),
            use_hybrid=False,
        )
        self.assertEqual([item["source"] for item in results], ["a.txt", "b.txt"])
        self.assertEqual(results[0]["fusion"], "semantic-only")

    def test_prompt_numbers_evidence(self):
        prompt = rag.build_prompt("question", self.metadata)
        self.assertIn("[Evidence 1: a.txt, chunk 0]", prompt)
        self.assertIn("[Evidence 2: b.txt, chunk 0]", prompt)

    def test_keyword_evidence_can_pass_low_semantic_guard(self):
        evidence = [{
            "source": "a.txt",
            "chunk_id": 0,
            "text": "MICE means multiple imputation by chained equations.",
            "semantic_score": 0.2,
            "bm25_score": 2.0,
        }]
        with patch.object(rag, "retrieve", return_value=evidence), patch.object(
            rag, "has_api_key", return_value=False
        ):
            result = rag.generate_answer("What is MICE?")
        self.assertIn("GROQ_API_KEY", result["answer"])
        self.assertNotEqual(result["answer"], rag.NO_ANSWER)


class SummaryTests(unittest.TestCase):
    def test_groups_respect_character_limit(self):
        groups = _group_chunks(["a" * 60, "b" * 60], max_chars=100)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len("\n\n".join(group)) <= 100 for group in groups))

    def test_all_groups_are_reduced(self):
        calls = []

        def fake_llm(prompt, max_tokens):
            calls.append((prompt, max_tokens))
            return f"summary-{len(calls)}"

        result = summarize_document(
            ["a" * 80, "b" * 80], group_size=1, llm=fake_llm, pacing_delay=0
        )
        self.assertEqual(result["num_groups"], 2)
        self.assertEqual(len(calls), 3)


class PdfTests(unittest.TestCase):
    def test_page_metadata_is_preserved(self):
        chunks, metadata = build_chunks_from_pdf("sample.pdf", [(4, "word " * 150)])
        self.assertEqual(len(chunks), len(metadata))
        self.assertTrue(all(item["page"] == 4 for item in metadata))
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))


class GroqTests(unittest.TestCase):
    def test_success_response_and_arguments(self):
        response = Mock(status_code=200, headers={})
        response.json.return_value = {"choices": [{"message": {"content": " answer "}}]}
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}), patch.object(
            groq_client.requests, "post", return_value=response
        ) as post:
            self.assertEqual(groq_client.call_llm("prompt", max_tokens=25), "answer")
            self.assertEqual(post.call_args.kwargs["json"]["max_completion_tokens"], 25)
        with self.assertRaises(ValueError):
            groq_client.call_llm("prompt", max_tokens=0)


if __name__ == "__main__":
    unittest.main()
