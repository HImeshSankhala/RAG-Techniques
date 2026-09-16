"""Agentic RAG against the real corpus, with the planner scripted.

Same approach as test_multi_pass_rag.py, and for the same reason: what this phase
adds is a loop whose shape is decided entirely by the model's replies, so the
replies are supplied verbatim rather than stubbed. Retrieval stays real — an
agent's repeat detection and its "nothing new came back" guard only mean
something against the actual index.

Requires the index: run `make index` first.
"""

import pytest

from core import llm, vectorstore
from core.config import settings
from core.llm import LLMResponse
from implementations.agentic_rag import Action, AgenticRAG, parse_plan

MAX_ITERATIONS = settings.agentic_max_iterations

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


# --- Termination: the phase's actual subject ---------------------------------


def test_the_agent_can_stop_itself_after_one_search(script) -> None:
    script("SEARCH vector dynamo vector clocks", "ANSWER", "Final answer citing dynamo.md.")

    result = AgenticRAG().run(QUERY)

    # Not Multi-Pass's `no_gaps_found`: that reports a critique's verdict on a
    # draft. This is the agent predicting sufficiency before anything was written.
    assert result.metadata.termination_reason == "agent_stopped"
    assert result.metadata.retrieval_passes == 1
    assert result.metadata.llm_calls == 3  # two plans + the answer
    assert result.answer == "Final answer citing dynamo.md."


def test_the_hard_cap_stops_a_planner_that_never_says_answer(script) -> None:
    """The guarantee. Every other guard is best-effort; only the counter cannot be
    talked out of stopping by a confused model."""
    script(
        "SEARCH vector dynamo vector clocks",
        "SEARCH keyword merkle anti-entropy",
        "SEARCH hybrid gossip membership protocol",
        "Final answer.",
    )

    result = AgenticRAG().run(QUERY)

    assert result.metadata.termination_reason == "max_iterations"
    assert result.metadata.retrieval_passes == MAX_ITERATIONS
    assert result.metadata.llm_calls == MAX_ITERATIONS + 1


def test_repeating_a_planned_action_ends_the_loop(script) -> None:
    """An agent that chooses its own next action can choose the same one forever.

    This is the guard Multi-Pass does not have — it can only notice a repeat after
    paying for the retrieval it produced.
    """
    script(
        "SEARCH vector dynamo vector clocks",
        "SEARCH vector  Dynamo Vector Clocks ",  # same action, different casing
        "Final answer.",
    )

    result = AgenticRAG().run(QUERY)

    assert result.metadata.termination_reason == "repeated_action"
    # 1, not 2: the repeat was caught before the retrieval ran, so no second
    # search happened and the metadata must not claim one.
    assert result.metadata.retrieval_passes == 1


def test_the_same_words_through_a_different_retriever_are_not_a_repeat(script) -> None:
    """Switching strategy after a miss is the one repeat worth allowing — BM25 and
    embeddings genuinely answer the same words differently."""
    script(
        "SEARCH vector KRaft metadata quorum",
        "SEARCH keyword KRaft metadata quorum",
        "ANSWER",
        "Final answer.",
    )

    result = AgenticRAG().run(QUERY)

    assert result.metadata.retrieval_passes == 2
    assert result.metadata.termination_reason == "agent_stopped"


def test_a_new_search_that_surfaces_nothing_new_ends_the_loop(script) -> None:
    """A different query returning only chunks the agent already had means the
    corpus is out of material, not that the planner is circling."""
    script(
        "SEARCH vector dynamo conflicting concurrent writes",
        # Different action (so not `repeated_action`), same neighbourhood of the
        # index, so dense retrieval returns the identical chunks.
        "SEARCH vector how does dynamo handle conflicting concurrent writes",
        "Final answer.",
    )

    result = AgenticRAG().run(QUERY)

    assert result.metadata.termination_reason == "no_new_evidence"
    # 2: the search ran and cost time whether or not it helped.
    assert result.metadata.retrieval_passes == 2


def test_never_exceeds_the_iteration_cap_regardless_of_the_planner(script) -> None:
    """The cost guarantee, stated as an inequality."""
    script(*["SEARCH hybrid gap %d" % i for i in range(MAX_ITERATIONS + 3)], "Final answer.")

    result = AgenticRAG().run(QUERY)

    assert result.metadata.retrieval_passes <= MAX_ITERATIONS
    assert result.metadata.llm_calls <= MAX_ITERATIONS + 1


# --- The planner is not trusted to be readable -------------------------------


def test_an_unreadable_plan_falls_back_to_the_baseline_and_says_so(script) -> None:
    """Silently falling back looks identical to a confident correct decision."""
    script("I think we should probably look into this further.", "ANSWER", "Final answer.")

    result = AgenticRAG().run(QUERY)

    assert "UNREADABLE PLAN" in result.steps[0].detail
    assert result.metadata.retrieval_passes == 1
    assert result.retrieved_chunks, "the fallback search must still retrieve"


def test_answering_before_retrieving_anything_is_overruled(script) -> None:
    """The one outcome worse than a wasted search: answering from memory, which is
    the failure RAG exists to prevent."""
    script("ANSWER", "ANSWER", "Final answer.")

    result = AgenticRAG().run(QUERY)

    assert "forcing" in result.steps[0].detail
    assert result.retrieved_chunks
    assert result.metadata.retrieval_passes == 1


def test_a_silent_planner_costs_an_iteration_and_says_so(script) -> None:
    """The specific unreadable reply this project keeps producing is no reply.

    `reason=True` draws thinking tokens from the same `num_predict` budget as the
    reply (see core/config.py), so a planner that thinks past the budget returns
    an empty string. Measured at the 1024 that shipped: 7 of 51 planner calls.
    It is the softer half of the same bug that made Multi-Pass's critique a no-op
    — silence there was read as "no gaps" and stopped the loop, silence here is
    caught by `understood` and costs one iteration on the baseline search — but
    the cost is real and it must be legible in the trace rather than inferred
    from an iteration that quietly did nothing.
    """
    script("", "ANSWER", "Final answer.")

    result = AgenticRAG().run(QUERY)

    assert "UNREADABLE PLAN" in result.steps[0].detail
    assert "(empty reply)" in result.steps[0].detail
    assert result.metadata.retrieval_passes == 1
    assert result.retrieved_chunks, "silence must still degrade to the baseline search"


def test_a_repeated_fallback_still_terminates(script) -> None:
    """The unreadable-plan path has no exit of its own — it relies on the repeat
    guard catching the identical fallback. This asserts that wiring."""
    script("mumble", "mumble again", "Final answer.")

    result = AgenticRAG().run(QUERY)

    assert result.metadata.termination_reason == "repeated_action"


# --- What the planner is shown -----------------------------------------------


def test_the_first_plan_is_made_without_passages(script) -> None:
    """It cannot assess evidence it does not have; iteration 1 is planning only."""
    seen = script("SEARCH vector dynamo vector clocks", "ANSWER", "Final answer.")

    AgenticRAG().run(QUERY)
    first_prompt, is_helper = seen[0]

    assert is_helper, "planning must take the tighter helper token cap"
    assert "No passages have been retrieved yet" in first_prompt
    assert QUERY in first_prompt


def test_the_second_plan_sees_the_evidence_and_the_search_history(script) -> None:
    """Without the passages it grades the question against its own pretrained
    knowledge; without the history it re-asks what it already asked."""
    seen = script("SEARCH vector dynamo vector clocks", "ANSWER", "Final answer.")

    result = AgenticRAG().run(QUERY)
    second_prompt, _ = seen[1]

    assert "vector: dynamo vector clocks" in second_prompt
    assert "Searches already run" in second_prompt
    for chunk in result.retrieved_chunks:
        assert chunk.text in second_prompt


def test_the_planner_chooses_the_retriever_not_only_the_query(script) -> None:
    """The one thing this technique has that Multi-Pass does not: every Multi-Pass
    retrieval is dense, and only the text is up to the model."""
    script("SEARCH keyword KRaft", "ANSWER", "Final answer.")

    result = AgenticRAG().run(QUERY)

    assert result.steps[1].name == "Search keyword (iteration 1)"
    # BM25 on a token that occurs in one chunk of the corpus lands that chunk;
    # dense retrieval on the Dynamo question does not go near kafka.md.
    assert any(c.source == "kafka.md" for c in result.retrieved_chunks)


def test_steps_trace_narrates_the_loop(script) -> None:
    """In compare mode the trace next to Standard RAG's three steps IS the lesson."""
    script("SEARCH vector dynamo vector clocks", "SEARCH keyword merkle", "ANSWER", "Answer.")

    result = AgenticRAG().run(QUERY)
    names = [step.name for step in result.steps]

    assert names == [
        "Plan (iteration 1)",
        "Search vector (iteration 1)",
        "Assess & re-plan (iteration 2)",
        "Search keyword (iteration 2)",
        "Assess & re-plan (iteration 3)",
        "Compare against plain dense retrieval",
        f"Answer from {len(result.retrieved_chunks)} chunks",
    ]
    for step in result.steps:
        assert step.detail, f"step '{step.name}' recorded no detail"


def test_evidence_accumulates_without_duplicates(script) -> None:
    script("SEARCH vector dynamo vector clocks", "SEARCH keyword KRaft", "ANSWER", "Answer.")

    ids = [c.chunk_id for c in AgenticRAG().run(QUERY).retrieved_chunks]

    assert len(ids) > 3, "the second search added nothing"
    assert len(ids) == len(set(ids)), "the same chunk was included twice"


def test_the_trace_reports_whether_planning_beat_plain_dense_retrieval(script) -> None:
    """The phase's claim, checked per run rather than asserted once on the page."""
    script("SEARCH keyword KRaft", "ANSWER", "Answer.")

    detail = AgenticRAG().run(QUERY).steps[-2].detail

    assert "NOT in dense top-" in detail


# --- Parsing free text from a small local model ------------------------------


def test_a_plain_search_line_parses() -> None:
    action, understood = parse_plan("SEARCH hybrid spanner truetime external consistency")

    assert understood
    assert action == Action("hybrid", "spanner truetime external consistency")


@pytest.mark.parametrize(
    "reply",
    ["ANSWER", "answer", "ANSWER.", "**ANSWER**", "- ANSWER"],
)
def test_answer_in_any_form_stops_the_loop(reply: str) -> None:
    """Misreading a stop signal as a plan spends an iteration on nothing."""
    assert parse_plan(reply) == (None, True)


def test_reasoning_preamble_before_the_decision_is_skipped() -> None:
    """With thinking on this model narrates before it decides. Retrieving for the
    narration would embed the planner's prose instead of the missing topic —
    the exact shape Multi-Pass's gap parser had to survive."""
    action, understood = parse_plan(
        "The passages cover Dynamo's vector clocks but say nothing about Merkle trees.\n"
        "SEARCH keyword merkle tree anti-entropy"
    )

    assert understood
    assert action == Action("keyword", "merkle tree anti-entropy")


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   \n  ",
        "SEARCH",  # no strategy, no query
        "SEARCH vector",  # a strategy but nothing to search for
        "SEARCH semantic dynamo writes",  # a strategy that does not exist
        "Let me think about what is missing here.",
    ],
)
def test_unusable_replies_are_reported_as_unusable(reply: str) -> None:
    """`(None, False)` is not `(None, True)`: one is a fallback, the other is a
    decision, and they must not look alike in the metadata."""
    assert parse_plan(reply) == (None, False)


def test_quoted_queries_are_unquoted_so_repeat_detection_still_works() -> None:
    """Measured: this model quotes its search string about half the time, so the
    quoted and unquoted spellings of one search must not look like two."""
    quoted, _ = parse_plan('SEARCH keyword "Kafka ZooKeeper replacement protocol"')
    plain, _ = parse_plan("SEARCH keyword Kafka ZooKeeper replacement protocol")

    assert quoted == plain
    assert quoted is not None and quoted.key == plain.key


def test_a_colon_after_the_strategy_is_tolerated() -> None:
    action, _ = parse_plan("SEARCH: keyword: KRaft metadata")

    assert action == Action("keyword", "KRaft metadata")


# --- Registration and setup state --------------------------------------------


def test_registered_and_runnable() -> None:
    from implementations.registry import get_pipeline

    pipeline = get_pipeline("agentic-rag")

    assert pipeline is not None
    assert pipeline.name == AgenticRAG.name


def test_empty_index_is_reported_not_raised(
    script, monkeypatch: pytest.MonkeyPatch
) -> None:
    script("SEARCH vector dynamo vector clocks")
    monkeypatch.setattr("core.retrieval.dense", lambda query, top_k: [])

    metadata = AgenticRAG().run(QUERY).metadata

    assert metadata.termination_reason == "empty_index"
    # The planning call already happened, so it must be reported rather than
    # rounded away — the run cost one LLM call even though it answered nothing.
    assert metadata.llm_calls == 1
