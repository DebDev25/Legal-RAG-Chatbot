"""
test_pipeline.py — end-to-end smoke test for the Legal RAG pipeline.

This is an INTEGRATION test: it exercises the real pipeline, so it requires
Ollama to be running with the qwen2.5:7b model pulled and a populated
storage/chroma_db (run src/ingest.py first). It is not a pure unit test.

Run directly:
    python tests/test_pipeline.py
or with pytest:
    pytest tests/test_pipeline.py -s
"""

import sys
from pathlib import Path

# Allow importing from src/ when run from the project root or from tests/.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from retrieval import query_rag, LegalResponse

load_dotenv()


def test_query_returns_structured_response():
    """A normal in-domain query returns a populated LegalResponse + source docs."""
    response, docs = query_rag(
        "What did the court rule regarding the admissibility of the evidence?"
    )

    # Contract: query_rag always returns (LegalResponse, list[Document]).
    assert isinstance(response, LegalResponse)
    assert isinstance(docs, list)

    # All five schema fields must be present and correctly typed.
    assert isinstance(response.answer, str) and response.answer.strip()
    assert response.confidence in {"high", "medium", "low", "none"}
    assert isinstance(response.found_in_context, bool)


def test_out_of_domain_query_is_gated():
    """A query with no relevant law should be gated to the 'no context' response."""
    response, docs = query_rag("What is the best recipe for chocolate chip cookies?")

    # The relevance gate should fire: no documents and a 'none' confidence.
    assert docs == []
    assert response.confidence == "none"
    assert response.found_in_context is False


if __name__ == "__main__":
    print("Running Legal RAG pipeline smoke tests (requires Ollama + ChromaDB)...\n")

    print("[1/2] In-domain query returns structured response...")
    test_query_returns_structured_response()
    print("      PASSED\n")

    print("[2/2] Out-of-domain query is gated...")
    test_out_of_domain_query_is_gated()
    print("      PASSED\n")

    print("All smoke tests passed.")
