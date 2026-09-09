"""Auto RAG against the real corpus, with the router's verdict scripted.

The router is the one part of this technique that cannot be tested by asking a
model — its reply is nondeterministic, and the property under test is what the
pipeline DOES with a verdict, not which verdict a small model happens to produce.
So the LLM is stubbed and the route is dictated; retrieval stays real, because
"keyword routed to BM25" is only meaningful if BM25 actually ran.

Requires the index: run `make index` first.
"""

import pytest

from core import keyword, llm, retrieval, vectorstore
from core.llm import LLMResponse
from implementations.auto_rag import FALLBACK_ROUTE, AutoRAG, parse_route

# Conceptual phrasing, so the corpus term and the query term differ.
QUERY = "How does Dynamo handle conflicting concurrent writes?"


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch):
    """Script the router's reply and record every call's flags.

    Returns a helper: `stub_llm(route_reply)` installs the stub and hands back the
    list of captured calls, so a test can assert both on the route taken and on
    how the router was invoked.
    """
    calls: list[dict] = []

    def install(router_reply: str) -> list[dict]:
        def fake_generate(system: str, user: str, model=None, **kwargs) -> LLMResponse:
            calls.append({"system": system, "user": user, **kwargs})
            # First call is the router, second is the answer.
            text = router_reply if len(calls) == 1 else "Stubbed answer citing dynamo.md."
            return LLMResponse(
                text=text,
                input_tokens=10 if len(calls) == 1 else 100,
                output_tokens=1 if len(calls) == 1 else 20,
                model=model or "stub-model",
                backend="ollama",
            )

        monkeypatch.setattr(llm, "generate", fake_generate)
        return calls

    return install


# --- The router call itself -------------------------------------------------


def test_router_is_a_helper_call_that_does_not_think(stub_llm) -> None:
    """The load-bearing property of this whole phase.

    Measured on this machine, the same helper call costs ~10s with thinking on and
    ~0.5s off. A thinking router costs more than the retrieval it is choosing
    between, which inverts the pattern: the cheap classifier stops being cheap and
    Auto RAG becomes a strictly slower Fusion RAG. See tests/test_llm_backends.py
    for the same assertion one layer down, on the wire payload.
    """
    calls = stub_llm("vector")
    AutoRAG().run(QUERY)

    router_call = calls[0]
    assert router_call["helper"] is True
    assert router_call["reason"] is False


def test_the_answer_call_is_not_a_helper_call(stub_llm) -> None:
    """Only the router is cheap. The answer gets the full budget and no thinking."""
    calls = stub_llm("vector")
    AutoRAG().run(QUERY)

    assert len(calls) == 2
    assert calls[1].get("helper", False) is False
    assert calls[1].get("reason", False) is False


# --- Each route dispatches to the right retriever ---------------------------


def test_vector_route_uses_dense_only(stub_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_llm("vector")
    monkeypatch.setattr(keyword, "query", _must_not_run("BM25"))

    result = AutoRAG().run(QUERY)

    assert result.metadata.termination_reason == "routed_vector"
    assert result.retrieved_chunks == retrieval.dense(QUERY, 4)


def test_keyword_route_uses_bm25_only(stub_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_llm("keyword")
    monkeypatch.setattr(retrieval, "dense", _must_not_run("dense retrieval"))

    result = AutoRAG().run("hinted handoff")

    assert result.metadata.termination_reason == "routed_keyword"
    assert result.retrieved_chunks == keyword.query("hinted handoff", 4)


def test_hybrid_route_fuses_both(stub_llm) -> None:
    stub_llm("hybrid")

    result = AutoRAG().run(QUERY)

    assert result.metadata.termination_reason == "routed_hybrid"
    assert result.retrieved_chunks == retrieval.hybrid(QUERY, 4).fused
    # RRF scores are tiny by construction (ceiling 2/61 for two lists), which is
    # the cheapest way to prove fusion ran rather than a single retriever.
    assert result.retrieved_chunks[0].score < 0.05


# --- Robustness: the router is a small model emitting free text -------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("keyword", "keyword"),
        ("  VECTOR\n", "vector"),
        ("**hybrid**", "hybrid"),
        ("Route: keyword.", "keyword"),
        ("vector, not keyword", "vector"),  # earliest wins
    ],
)
def test_decorated_replies_are_still_understood(reply: str, expected: str) -> None:
    assert parse_route(reply) == (expected, True)


@pytest.mark.parametrize("reply", ["", "I'm not sure.", "banana", "42"])
def test_unusable_replies_fall_back(reply: str) -> None:
    assert parse_route(reply) == (FALLBACK_ROUTE, False)


def test_garbage_router_output_falls_back_to_hybrid_and_says_so(stub_llm) -> None:
    """The fallback must be loud. A silent one is indistinguishable in the trace
    from a confident correct decision, which is the case worth telling apart."""
    stub_llm("I cannot decide, sorry!")

    result = AutoRAG().run(QUERY)

    assert result.metadata.termination_reason == "routed_hybrid_fallback"
    assert result.retrieved_chunks == retrieval.hybrid(QUERY, 4).fused
    assert "FALLBACK" in result.steps[0].detail
    assert "I cannot decide" in result.steps[0].detail


# --- The trace, which is the teaching surface -------------------------------


def test_steps_name_the_chosen_route_and_quote_the_router(stub_llm) -> None:
    stub_llm("keyword")

    steps = AutoRAG().run("hinted handoff").steps
    names = [s.name for s in steps]

    assert names == ["Route query", "Retrieve (BM25 only)", "Generate answer"]
    assert "route=keyword" in steps[0].detail
    # The raw reply, not just the parsed outcome — see the docstring on the step.
    assert "'keyword'" in steps[0].detail


def test_router_is_counted_as_an_llm_call(stub_llm) -> None:
    """The router is cheap, not free. Reporting one call would flatter the
    technique in exactly the view built to compare costs."""
    stub_llm("vector")

    metadata = AutoRAG().run(QUERY).metadata

    assert metadata.llm_calls == 2
    assert metadata.retrieval_passes == 1
    assert metadata.tokens_in == 110  # router 10 + answer 100
    assert metadata.tokens_out == 21


def test_empty_index_is_reported_not_raised(
    stub_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_llm("hybrid")
    monkeypatch.setattr(vectorstore, "count", lambda: 0)
    monkeypatch.setattr(vectorstore, "query", lambda embedding, top_k: [])

    result = AutoRAG().run(QUERY)

    assert result.retrieved_chunks == []
    assert result.metadata.termination_reason == "empty_index"
    assert result.metadata.backend == "ollama"
    assert "make index" in result.answer


def test_registered_and_runnable() -> None:
    from implementations.registry import get_pipeline, list_techniques

    assert get_pipeline("auto-rag") is not None
    assert get_pipeline("auto-rag").name == AutoRAG.name
    assert ("auto-rag", True) in [(t.name, ok) for t, ok in list_techniques()]


def _must_not_run(what: str):
    def fail(*args, **kwargs):
        raise AssertionError(f"{what} ran on a route that should not use it")

    return fail
