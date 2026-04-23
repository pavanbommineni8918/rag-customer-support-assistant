#!/usr/bin/env python3
"""
tests/test_system.py — Unit and integration tests for the RAG system
Run: python tests/test_system.py
"""

import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────
#  TEST: Document Processor
# ─────────────────────────────────────────────────────

class TestDocumentProcessor(unittest.TestCase):

    def test_clean_text_removes_extra_whitespace(self):
        from src.document_processor import _clean_text
        result = _clean_text("Hello   World\n\n\n\nTest")
        self.assertNotIn("   ", result)
        self.assertNotIn("\n\n\n", result)

    def test_clean_text_preserves_content(self):
        from src.document_processor import _clean_text
        text = "Return Policy: Items must be returned within 30 days."
        result = _clean_text(text)
        self.assertIn("Return Policy", result)
        self.assertIn("30 days", result)

    def test_chunk_documents_produces_chunks(self):
        from src.document_processor import chunk_documents
        from langchain_core.documents import Document
        docs = [Document(
            page_content="A" * 1500,  # Enough to produce multiple chunks
            metadata={"source": "test.pdf", "page": 1}
        )]
        chunks = chunk_documents(docs)
        self.assertGreater(len(chunks), 1, "Should produce multiple chunks")

    def test_chunk_metadata_preserved(self):
        from src.document_processor import chunk_documents
        from langchain_core.documents import Document
        docs = [Document(
            page_content="Test content " * 100,
            metadata={"source": "test.pdf", "page": 2}
        )]
        chunks = chunk_documents(docs)
        for chunk in chunks:
            self.assertIn("source", chunk.metadata)
            self.assertIn("chunk_id", chunk.metadata)
            self.assertEqual(chunk.metadata["source"], "test.pdf")

    def test_load_pdf_raises_on_missing_file(self):
        from src.document_processor import load_pdf
        with self.assertRaises(FileNotFoundError):
            load_pdf("/nonexistent/path/file.pdf")


# ─────────────────────────────────────────────────────
#  TEST: LLM Handler
# ─────────────────────────────────────────────────────

class TestLLMHandler(unittest.TestCase):

    def test_parse_response_high_confidence(self):
        from src.llm_handler import parse_response
        raw = "The return policy is 30 days.\n\nCONFIDENCE: HIGH"
        answer, score, label = parse_response(raw)
        self.assertEqual(label, "HIGH")
        self.assertGreaterEqual(score, 0.8)
        self.assertNotIn("CONFIDENCE", answer)

    def test_parse_response_medium_confidence(self):
        from src.llm_handler import parse_response
        raw = "It may be possible to return the item.\n\nCONFIDENCE: MEDIUM"
        answer, score, label = parse_response(raw)
        self.assertEqual(label, "MEDIUM")
        self.assertGreater(score, 0.5)

    def test_parse_response_low_confidence(self):
        from src.llm_handler import parse_response
        raw = "Some response without confidence tag"
        answer, score, label = parse_response(raw)
        self.assertEqual(label, "LOW")
        self.assertLess(score, 0.5)

    def test_parse_response_insufficient_info(self):
        from src.llm_handler import parse_response
        raw = "I don't have sufficient information to answer this question.\n\nCONFIDENCE: MEDIUM"
        answer, score, label = parse_response(raw)
        # Should downgrade confidence because of explicit uncertainty
        self.assertLessEqual(score, 0.4)

    def test_build_prompt_includes_query(self):
        from src.llm_handler import build_prompt
        from langchain_core.documents import Document
        chunks = [Document(page_content="Test context", metadata={"source": "test.pdf", "page": 1})]
        prompt = build_prompt("What is the return policy?", chunks)
        self.assertIn("What is the return policy?", prompt)
        self.assertIn("Test context", prompt)

    def test_build_prompt_empty_chunks(self):
        from src.llm_handler import build_prompt
        prompt = build_prompt("A question?", [])
        self.assertIn("No relevant information", prompt)


# ─────────────────────────────────────────────────────
#  TEST: HITL Handler
# ─────────────────────────────────────────────────────

class TestHITLHandler(unittest.TestCase):

    def test_classify_intent_sensitive(self):
        from src.hitl_handler import classify_intent
        self.assertEqual(classify_intent("I want to sue your company"), "sensitive")
        self.assertEqual(classify_intent("I need a lawyer"), "sensitive")
        self.assertEqual(classify_intent("This is a complaint"), "sensitive")

    def test_classify_intent_standard(self):
        from src.hitl_handler import classify_intent
        self.assertEqual(classify_intent("What is your return policy?"), "standard")
        self.assertEqual(classify_intent("How do I track my order?"), "standard")

    def test_classify_intent_complex(self):
        from src.hitl_handler import classify_intent
        long_query = "What is your return policy? " * 10  # Very long
        self.assertEqual(classify_intent(long_query), "complex")

    def test_should_escalate_low_confidence(self):
        from src.hitl_handler import should_escalate
        escalate, reason = should_escalate(0.3, 2, "Some answer", "standard")
        self.assertTrue(escalate)
        self.assertIn("confidence", reason)

    def test_should_escalate_no_chunks(self):
        from src.hitl_handler import should_escalate
        escalate, reason = should_escalate(0.9, 0, "Some answer", "standard")
        self.assertTrue(escalate)
        self.assertIn("chunk", reason)

    def test_should_not_escalate_high_confidence(self):
        from src.hitl_handler import should_escalate
        escalate, reason = should_escalate(0.9, 3, "Clear answer about policy", "standard")
        self.assertFalse(escalate)
        self.assertEqual(reason, "")

    def test_should_escalate_sensitive_intent(self):
        from src.hitl_handler import should_escalate
        escalate, reason = should_escalate(0.95, 4, "Good answer", "sensitive")
        self.assertTrue(escalate)

    def test_log_escalation_creates_file(self):
        from src.hitl_handler import log_escalation
        from langchain_core.documents import Document

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            log_path = f.name

        # Patch config to use temp file
        with patch("src.hitl_handler.config") as mock_cfg:
            mock_cfg.ESCALATION_LOG = log_path
            mock_cfg.HITL_TIMEOUT_SECONDS = 30
            mock_cfg.CONFIDENCE_THRESHOLD = 0.6

            chunks = [Document(page_content="test", metadata={"source": "t.pdf", "page": 1})]
            log_escalation(
                query="Test query",
                chunks=chunks,
                llm_answer="Test answer",
                confidence=0.3,
                reason="low_confidence",
                escalation_id="TEST-001",
            )

        with open(log_path) as f:
            records = json.load(f)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "TEST-001")
        self.assertEqual(records[0]["query"], "Test query")
        os.unlink(log_path)


# ─────────────────────────────────────────────────────
#  TEST: Graph Engine (Routing Logic)
# ─────────────────────────────────────────────────────

class TestGraphRouting(unittest.TestCase):

    def _make_state(self, escalation_flag=False):
        return {
            "query": "Test query",
            "cleaned_query": "Test query",
            "intent": "standard",
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "augmented_prompt": "",
            "raw_llm_response": "",
            "answer": "",
            "confidence": 0.8,
            "confidence_label": "HIGH",
            "escalation_flag": escalation_flag,
            "escalation_reason": "",
            "sources": [],
            "human_response": None,
            "final_response": "",
            "total_time_ms": None,
        }

    def test_routing_output_when_not_escalating(self):
        from src.graph_engine import routing_logic
        state = self._make_state(escalation_flag=False)
        self.assertEqual(routing_logic(state), "output")

    def test_routing_hitl_when_escalating(self):
        from src.graph_engine import routing_logic
        state = self._make_state(escalation_flag=True)
        self.assertEqual(routing_logic(state), "hitl")

    def test_input_node_cleans_query(self):
        from src.graph_engine import input_node
        state = self._make_state()
        state["query"] = "  What   is   your   policy?  "
        result = input_node(state)
        self.assertEqual(result["cleaned_query"].strip(), result["cleaned_query"])
        self.assertNotIn("   ", result["cleaned_query"])

    def test_input_node_caps_query_length(self):
        from src.graph_engine import input_node
        state = self._make_state()
        state["query"] = "Q" * 1000  # Very long query
        result = input_node(state)
        self.assertLessEqual(len(result["cleaned_query"]), 500)

    def test_output_node_formats_response(self):
        from src.graph_engine import output_node
        state = self._make_state()
        state["answer"] = "The return period is 30 days."
        state["confidence_label"] = "HIGH"
        state["sources"] = ["policy.pdf p.5"]
        result = output_node(state)
        self.assertIn("30 days", result["final_response"])
        self.assertIn("HIGH", result["final_response"])


# ─────────────────────────────────────────────────────
#  TEST: Config
# ─────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):

    def test_default_values_are_set(self):
        from src.config import config
        self.assertEqual(config.TOP_K_CHUNKS, 4)
        self.assertEqual(config.CHUNK_SIZE, 500)
        self.assertEqual(config.CHUNK_OVERLAP, 50)
        self.assertEqual(config.CONFIDENCE_THRESHOLD, 0.6)

    def test_sensitive_keywords_list(self):
        from src.config import config
        self.assertIn("sue", config.SENSITIVE_KEYWORDS)
        self.assertIn("lawyer", config.SENSITIVE_KEYWORDS)
        self.assertIsInstance(config.SENSITIVE_KEYWORDS, list)


# ─────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  RAG System Test Suite")
    print("═" * 50 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for cls in [
        TestDocumentProcessor,
        TestLLMHandler,
        TestHITLHandler,
        TestGraphRouting,
        TestConfig,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "═" * 50)
    print(f"  Tests: {result.testsRun} | "
          f"Passed: {result.testsRun - len(result.failures) - len(result.errors)} | "
          f"Failed: {len(result.failures)} | "
          f"Errors: {len(result.errors)}")
    print("═" * 50 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
