"""Smoke test for the Standard RAG pipeline against the real sample corpus.

The LLM call is stubbed. That is deliberate rather than a shortcut: it keeps the
suite runnable without an API key or network, makes it deterministic, and keeps
the assertions on the part this phase actually owns — retrieval, step recording,
and the shape of RAGResult. Whether the model writes a good sentence is not
something a unit test can assert anyway.

Requires the index to exist: run `make index` first.
"""

import pytest

from core import llm, vectorstore
from core.llm import LLMResponse
from implementations.standard_rag import StandardRAG

QUERY = "How does Dynamo handle conflicting concurrent writes?"


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the LLM call, capturing the prompt so we can assert on grounding."""
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
    result = StandardRAG().run(QUERY)

    assert result.answer == "Stubbed answer citing dynamo.md."
    assert result.retrieved_chunks, "retrieved nothing"
    assert result.steps, "recorded no steps"


def test_retrieves_from_the_relevant_document(stub_llm: list[str]) -> None:
    """Vector search should route a Dynamo question to the Dynamo document."""
    result = StandardRAG().run(QUERY)
    assert any(c.source == "dynamo.md" for c in result.retrieved_chunks)


def test_chunks_are_ordered_best_first(stub_llm: list[str]) -> None:
    """The UI shows these in order and calls the first one the best match."""
    scores = [c.score for c in StandardRAG().run(QUERY).retrieved_chunks]
    assert scores == sorted(scores, reverse=True)


def test_prompt_contains_the_retrieved_text(stub_llm: list[str]) -> None:
    """The grounding guarantee: what was retrieved is what the model was shown."""
    result = StandardRAG().run(QUERY)
    prompt = stub_llm[0]

    assert QUERY in prompt
    for chunk in result.retrieved_chunks:
        assert chunk.text in prompt


def test_steps_trace_covers_the_three_stages(stub_llm: list[str]) -> None:
    """The compare view is built on these names, so they are part of the contract."""
    steps = StandardRAG().run(QUERY).steps

    assert [s.name for s in steps] == ["Embed query", "Retrieve chunks", "Generate answer"]
    for step in steps:
        assert step.detail, f"step '{step.name}' recorded no detail"
        assert step.duration_ms >= 0


def test_metadata_reports_cost_and_latency(stub_llm: list[str]) -> None:
    metadata = StandardRAG().run(QUERY).metadata

    assert metadata.llm_calls == 1
    assert metadata.retrieval_passes == 1
    assert metadata.latency_ms > 0
    assert metadata.tokens_in == 100
    assert metadata.tokens_out == 20
    assert metadata.termination_reason == "single_pass"
    # Local backend is free; the paid path is what carries a cost.
    assert metadata.cost_estimate_usd == 0.0


def test_registered_under_its_slug() -> None:
    """The registry key must match the class attribute, or /api/run 404s."""
    from implementations.registry import get_pipeline

    assert get_pipeline("standard-rag") is not None
    assert get_pipeline("standard-rag").name == StandardRAG.name
