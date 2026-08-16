"""Multi-Pass RAG against the real corpus, with the LLM scripted.

The LLM is not merely stubbed here, it is *scripted*: each test supplies the exact
sequence of replies the model would give, because what this phase adds is a loop
whose shape is decided by those replies. Retrieval stays real — the gap queries
have to hit the actual index for "no new evidence" to mean anything.

Requires the index: run `make index` first.
"""

import pytest

from core import llm, vectorstore
from core.config import settings
from core.llm import LLMResponse
from implementations.multi_pass_rag import MultiPassRAG, _merge, _parse_gaps

MAX_PASSES = settings.multi_pass_max_passes

QUERY = "How does Dynamo handle conflicting concurrent writes?"


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch):
    """Queue replies the fake LLM returns in order, and record the prompts it saw."""

    def install(*replies: str) -> list[tuple[str, bool]]:
        remaining = list(replies)
        seen: list[tuple[str, bool]] = []

        def fake_generate(system, user, model=None, *, helper=False, **kwargs) -> LLMResponse:
            seen.append((user, helper))
            assert remaining, "pipeline made more LLM calls than the script supplied"
            return LLMResponse(
                text=remaining.pop(0),
                input_tokens=100,
                output_tokens=20,
                model=model or "stub-model",
                backend="ollama",
            )

        monkeypatch.setattr(llm, "generate", fake_generate)
        return seen

    return install


# --- Termination: the part that actually needed designing --------------------


def test_an_empty_critique_is_named_in_the_trace(script) -> None:
    """Silence and agreement produce the same control flow, so they must not
    produce the same trace. An empty reply meant the reasoning ate the output
    budget — the loop looked healthy while doing nothing."""
    script("Draft citing dynamo.md.", "")

    steps = MultiPassRAG().run(QUERY).steps

    assert "returned nothing" in steps[-1].detail


def test_stops_after_one_pass_when_the_critique_finds_nothing(script) -> None:
    script("Draft citing dynamo.md.", "COMPLETE")

    result = MultiPassRAG().run(QUERY)

    assert result.metadata.termination_reason == "single_pass"
    assert result.metadata.retrieval_passes == 1
    assert result.metadata.llm_calls == 2  # draft + critique


def test_closes_a_gap_and_stops(script) -> None:
    """The happy path the technique exists for: one gap, filled, then done."""
    script(
        "Draft citing dynamo.md.",
        "hinted handoff replica failure",
        "Better draft citing dynamo.md.",
        "COMPLETE",
    )

    result = MultiPassRAG().run(QUERY)

    assert result.metadata.termination_reason == "gaps_closed"
    assert result.metadata.retrieval_passes == 2
    assert result.metadata.llm_calls == 4  # draft, critique, redraft, critique
    assert result.answer == "Better draft citing dynamo.md."


def test_hard_cap_stops_a_critique_that_never_says_complete(script) -> None:
    """The guarantee. A model that always finds gaps must still terminate."""
    script(
        "Draft one.",
        "vector clocks reconciliation",
        "Draft two.",
        "merkle trees anti-entropy",
        "Draft three.",
    )

    result = MultiPassRAG().run(QUERY)

    assert result.metadata.termination_reason == "max_iterations"
    assert result.metadata.retrieval_passes == MAX_PASSES
    assert result.answer == "Draft three."


def test_stops_when_re_retrieval_surfaces_nothing_new(script) -> None:
    """A gap the corpus cannot fill must not consume the remaining passes.

    The gap query repeats the original question, so retrieval returns the chunks
    pass 1 already had.
    """
    script("Draft citing dynamo.md.", QUERY)

    result = MultiPassRAG().run(QUERY)

    assert result.metadata.termination_reason == "no_new_evidence"
    # 2, not 1: the second retrieval ran and cost time. Counting only retrievals
    # that paid off would hide the technique's wasted work.
    assert result.metadata.retrieval_passes == 2


def test_never_exceeds_the_pass_cap_regardless_of_the_critique(script) -> None:
    """Belt and braces on the cost guarantee, stated as an inequality."""
    script("d1", "gap a", "d2", "gap b", "d3", "gap c", "d4")

    result = MultiPassRAG().run(QUERY)

    assert result.metadata.retrieval_passes <= MAX_PASSES
    assert result.metadata.llm_calls <= 2 * MAX_PASSES - 1


# --- Evidence accumulates ----------------------------------------------------


def test_later_passes_add_evidence_rather_than_replacing_it(script) -> None:
    script("Draft.", "hinted handoff replica failure", "Redraft.", "COMPLETE")

    result = MultiPassRAG().run(QUERY)
    ids = [c.chunk_id for c in result.retrieved_chunks]

    assert len(ids) > 4, "re-retrieval added nothing"
    assert len(ids) == len(set(ids)), "the same chunk was included twice"


def test_the_critique_is_shown_the_passages_and_the_draft(script) -> None:
    """Otherwise it reviews the draft against its own pretrained knowledge and
    reports every fact the corpus happens not to contain."""
    seen = script("Draft citing dynamo.md.", "COMPLETE")

    result = MultiPassRAG().run(QUERY)
    critique_prompt, is_helper = seen[1]

    assert is_helper, "the critique must take the tighter helper token cap"
    assert "Draft citing dynamo.md." in critique_prompt
    for chunk in result.retrieved_chunks:
        assert chunk.text in critique_prompt


def test_steps_trace_narrates_the_loop(script) -> None:
    """In compare mode the trace next to Standard RAG's three steps IS the lesson."""
    script("Draft.", "hinted handoff replica failure", "Redraft.", "COMPLETE")

    result = MultiPassRAG().run(QUERY)
    names = [s.name for s in result.steps]

    # The redraft step names its context size, which depends on how much the gap
    # query overlapped pass 1 — asserted against the result rather than hardcoded,
    # so a retrieval change fails the assertion that cares instead of this one.
    assert names == [
        "Retrieve (pass 1)",
        "Draft answer",
        "Critique draft (pass 1)",
        "Retrieve for gaps (pass 2)",
        f"Redraft with {len(result.retrieved_chunks)} chunks (pass 2)",
        "Critique draft (pass 2)",
    ]
    for step in result.steps:
        assert step.detail, f"step '{step.name}' recorded no detail"


def test_more_steps_than_standard_rag(script) -> None:
    """The compare view's steps_delta only carries information once a technique
    with a genuinely different shape exists. This is that technique."""
    script("Draft.", "hinted handoff replica failure", "Redraft.", "COMPLETE")

    assert len(MultiPassRAG().run(QUERY).steps) > 3


# --- Parsing free text from a small local model ------------------------------


@pytest.mark.parametrize(
    "critique",
    ["COMPLETE", "complete", "COMPLETE.", "Complete - the draft answers it", "", "   \n  "],
)
def test_complete_in_any_form_means_no_gaps(critique: str) -> None:
    """Misreading a completion signal as a gap spends a whole pass on nothing."""
    assert _parse_gaps(critique) == []


def test_gaps_are_stripped_of_list_markers_and_capped() -> None:
    parsed = _parse_gaps("- vector clocks\n* merkle trees\n• read repair\n- gossip")

    assert parsed == ["vector clocks", "merkle trees"]


def test_reasoning_preamble_is_not_mistaken_for_a_search_query() -> None:
    """With thinking enabled the critique narrates before it searches.

    Retrieving for the narration embeds the critique's own prose instead of the
    missing topic — a wasted pass that also pollutes the context. This is the exact
    shape qwen3:8b produced live.
    """
    parsed = _parse_gaps(
        "The provided context passages do not mention Spanner or TrueTime. "
        "Therefore, the answer is incomplete.\nspanner TrueTime\nexternal consistency"
    )

    assert parsed == ["spanner TrueTime", "external consistency"]


def test_merge_keeps_order_and_drops_duplicates() -> None:
    a, b, c = (
        vectorstore.query(embedding, 3)[0]
        for embedding in [[0.0] * 384, [0.0] * 384, [0.0] * 384]
    )
    assert _merge([a], [b, c]) == [a]  # identical retrieval → nothing new


# --- Registration ------------------------------------------------------------


def test_registered_and_runnable() -> None:
    from implementations.registry import get_pipeline

    pipeline = get_pipeline("multi-pass-rag")

    assert pipeline is not None
    assert pipeline.name == MultiPassRAG.name


def test_empty_index_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vectorstore, "query", lambda embedding, top_k: [])

    metadata = MultiPassRAG().run(QUERY).metadata

    assert metadata.termination_reason == "empty_index"
    assert metadata.backend == "ollama"
