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

    `hinted handoff` is a rare literal term. Dense retrieval leads with raft.md —
    right topic, wrong document, and the phrase appears nowhere in it. BM25 leads
    with a passage that actually contains the words.

    Asserted on the text rather than on a filename. The first version of this test
    pinned `dynamo.md`, and adding cassandra.md broke it — Cassandra inherits
    hinted handoff from Dynamo, so BM25 now ranks that passage first. Both answers
    are correct; the claim being tested is about which retriever finds the literal
    term, not about which file happens to win.
    """
    from core import embeddings

    dense = vectorstore.query(embeddings.embed_query("hinted handoff"), 4)
    sparse = keyword.query("hinted handoff", 4)

    assert "hinted handoff" not in dense[0].text.lower(), "dense unexpectedly got this right"
    assert "hinted handoff" in sparse[0].text.lower(), "BM25 should nail an exact rare term"


def test_fusion_recovers_the_correct_document(stub_llm: list[str]) -> None:
    """End to end: the merged result leads with a passage that actually contains
    the term, even though dense retrieval alone did not."""
    result = FusionRAG().run("hinted handoff")
    assert "hinted handoff" in result.retrieved_chunks[0].text.lower()


def test_empty_index_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both retrievers must return empty on an unbuilt index, not one of each.

    BM25 used to raise here while dense returned []. Fusion never reached its own
    empty-index branch, and the RuntimeError escaped /api/run as a 500 — the one
    setup mistake most likely to be a new user's first request.
    """
    monkeypatch.setattr(vectorstore, "count", lambda: 0)
    monkeypatch.setattr(vectorstore, "query", lambda embedding, top_k: [])

    assert keyword.query("hinted handoff", 4) == []

    result = FusionRAG().run(QUERY)

    assert result.retrieved_chunks == []
    assert result.metadata.termination_reason == "empty_index"
    assert result.metadata.backend == "ollama"
    assert "make index" in result.answer


def test_registered_and_runnable() -> None:
    from implementations.registry import get_pipeline

    assert get_pipeline("fusion-rag") is not None
    assert get_pipeline("fusion-rag").name == FusionRAG.name
