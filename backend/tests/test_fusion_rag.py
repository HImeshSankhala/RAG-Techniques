"""Fusion RAG against the real corpus, with the LLM call stubbed.

Requires the index: run `make index` first.
"""

import pytest

from core import keyword, llm, vectorstore
from core.llm import LLMResponse
from implementations.fusion_rag import FusionRAG

QUERY = "How does Dynamo handle conflicting concurrent writes?"


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []

    def fake_generate(system: str, user: str, model=None, **kwargs) -> LLMResponse:
        captured.append(user)
        return LLMResponse(
            text="Stubbed answer citing dynamo.md.",
            input_tokens=100,
            output_tokens=20,
            model=model or "stub-model",
            backend="ollama",
        )

    monkeypatch.setattr(llm, "generate", fake_generate)
    return captured


def test_run_returns_a_populated_result(stub_llm: list[str]) -> None:
    result = FusionRAG().run(QUERY)

    assert result.answer
    assert result.retrieved_chunks
    assert result.steps


def test_steps_show_both_retrievers_and_the_merge(stub_llm: list[str]) -> None:
    """The trace is the teaching surface — it must show that two retrievers ran."""
    steps = FusionRAG().run(QUERY).steps
    names = [s.name for s in steps]

    assert names == [
        "Retrieve (dense + BM25 in parallel)",
        "Fuse by reciprocal rank",
        "Generate answer",
    ]
    assert "dense" in steps[0].detail and "BM25" in steps[0].detail
    assert "k=60" in steps[1].detail


def test_still_one_retrieval_pass(stub_llm: list[str]) -> None:
    """Two retrievers, one pass. This is what separates Fusion from Multi-Pass,
    and the compare view diffs on exactly this field."""
    metadata = FusionRAG().run(QUERY).metadata

    assert metadata.retrieval_passes == 1
    assert metadata.llm_calls == 1


def test_bm25_finds_exact_terms_dense_misses() -> None:
    """The reason this technique exists, asserted against the real corpus.

    `hinted handoff` is a Dynamo concept. Dense retrieval puts raft.md on top —
    the right topic, the wrong document. BM25 gets it right.
    """
    from core import embeddings

    dense = vectorstore.query(embeddings.embed_query("hinted handoff"), 4)
    sparse = keyword.query("hinted handoff", 4)

    assert dense[0].source != "dynamo.md", "dense unexpectedly got this right"
    assert sparse[0].source == "dynamo.md", "BM25 should nail an exact rare term"


def test_fusion_recovers_the_correct_document(stub_llm: list[str]) -> None:
    """End to end: the merged result leads with the right document even though
    dense retrieval alone did not."""
    result = FusionRAG().run("hinted handoff")
    assert result.retrieved_chunks[0].source == "dynamo.md"


def test_registered_and_runnable() -> None:
    from implementations.registry import get_pipeline

    assert get_pipeline("fusion-rag") is not None
    assert get_pipeline("fusion-rag").name == FusionRAG.name
