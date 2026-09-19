"""Feedback RAG against the real corpus, with the LLM scripted and votes in a temp DB.

Retrieval stays real, as in test_interactive_rag.py: "a vote moves a passage into
the answer" and "a correct judgment on one query displaces evidence for another"
are claims about the actual index, and a fake retriever would make them vacuous.

Requires the index: run `make index` first.
"""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import llm, vectorstore
from core.config import settings
from core.llm import LLMResponse
from core.pipeline import Chunk
from implementations.feedback_rag import (
    CAP,
    FEEDBACK_CANDIDATES,
    SHIFT,
    FeedbackRAG,
    chunk_hash,
    clear_feedback,
    record_feedback,
    rerank,
)
from implementations.standard_rag import StandardRAG

QUERY = "How does Dynamo handle conflicting concurrent writes?"
# The measured cross-query pair: the only passage mentioning stragglers is
# mapreduce.md#2, and it is also a candidate for the Raft question.
MAPREDUCE = "How are stragglers handled in MapReduce?"
RAFT = "How does Raft elect a leader?"

client = TestClient(app)


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture(autouse=True)
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Never the real store: these tests write votes, and votes have no TTL."""
    path = tmp_path / "feedback.db"
    monkeypatch.setattr(settings, "db_path", path)
    return path


@pytest.fixture(autouse=True)
def script(monkeypatch: pytest.MonkeyPatch) -> None:
    """One canned answer per call, so no test needs Ollama."""

    def fake_generate(system, user, model=None, **kwargs) -> LLMResponse:
        return LLMResponse(
            text="Answer citing dynamo.md.",
            input_tokens=100,
            output_tokens=20,
            model=model or "stub-model",
            backend="ollama",
        )

    monkeypatch.setattr(llm, "generate", fake_generate)


def ids(result) -> list[str]:
    return [chunk.chunk_id for chunk in result.retrieved_chunks]


def candidates(query: str) -> list[Chunk]:
    """The 12 the pipeline would rerank, in dense order."""
    from core import embeddings

    return vectorstore.query(embeddings.embed_query(query), FEEDBACK_CANDIDATES)


def stored_rows(db: Path) -> int:
    """Votes on disk. A missing table means the store was never written to, which
    is exactly what "nothing was stored" looks like before the first vote."""
    with sqlite3.connect(db) as conn:
        try:
            return conn.execute("SELECT count(*) FROM feedback").fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def vote(query: str, chunk_ids: list[str], rating: int, times: int = 1) -> None:
    for _ in range(times):
        record_feedback("feedback-rag", query, chunk_ids, rating)


# --- cold start ------------------------------------------------------------


def test_with_no_votes_it_is_standard_rag() -> None:
    """The cold-start promise: feedback that does not exist changes nothing.

    Over-fetching 12 and cutting to 4 must return exactly what a plain top-4
    query returns, or every later assertion about "what the votes did" would be
    measuring the over-fetch instead.
    """
    feedback = FeedbackRAG().run(QUERY)
    standard = StandardRAG().run(QUERY)

    assert ids(feedback) == ids(standard)
    assert feedback.metadata.feedback_votes == 0
    assert feedback.metadata.termination_reason == "single_pass"
    assert feedback.metadata.llm_calls == 1, "reranking costs no extra call"
    assert feedback.metadata.retrieval_passes == 1
    assert [s.name for s in feedback.steps] == [
        "Embed query",
        "Retrieve candidates",
        "Apply feedback",
        "Generate answer",
    ]
    assert "cold start" in feedback.steps[2].detail


def test_cold_start_holds_for_a_second_query() -> None:
    assert ids(FeedbackRAG().run(RAFT)) == ids(StandardRAG().run(RAFT))


# --- what votes do ---------------------------------------------------------


def test_an_upvoted_passage_moves_into_the_answer() -> None:
    deep = candidates(QUERY)[settings.top_k + 2]  # dense rank 7, outside the answer
    vote(QUERY, [deep.chunk_id], 1, times=CAP)

    result = FeedbackRAG().run(QUERY)

    # Read the position off the returned list, never off the sort key.
    assert deep.chunk_id in ids(result)
    assert result.metadata.feedback_votes == CAP
    assert f"{deep.chunk_id} net +{CAP}" in result.steps[2].detail


def test_a_downvoted_passage_is_pushed_out_of_the_answer() -> None:
    top = candidates(QUERY)[0]
    vote(QUERY, [top.chunk_id], -1, times=CAP)

    result = FeedbackRAG().run(QUERY)

    assert top.chunk_id not in ids(result), "the retriever's own #1, voted out"
    assert f"{top.chunk_id} net -{CAP}" in result.steps[2].detail


def test_votes_beyond_the_cap_change_nothing() -> None:
    top = candidates(QUERY)[0]
    vote(QUERY, [top.chunk_id], -1, times=CAP)
    capped = ids(FeedbackRAG().run(QUERY))

    vote(QUERY, [top.chunk_id], -1, times=7)
    result = FeedbackRAG().run(QUERY)

    assert ids(result) == capped
    # The cap bounds the ranking, not the honesty of the count.
    assert result.metadata.feedback_votes == CAP + 7


def test_a_correct_vote_on_one_query_displaces_evidence_for_another() -> None:
    """The filter bubble, on the real index: votes are global.

    Upvoting the one passage that mentions stragglers is a correct judgment for
    the MapReduce question. Because votes are keyed to the passage and not the
    query, it also reranks an unrelated Raft question — and pushes out a passage
    that question needed.
    """
    straggler = next(c for c in candidates(MAPREDUCE) if "straggler" in c.text.lower())
    before = ids(FeedbackRAG().run(RAFT))

    vote(MAPREDUCE, [straggler.chunk_id], 1, times=CAP)
    after = FeedbackRAG().run(RAFT)

    assert straggler.chunk_id not in before
    assert straggler.chunk_id in ids(after), "a MapReduce vote reranked the Raft question"
    evicted = [chunk_id for chunk_id in before if chunk_id not in ids(after)]
    assert evicted, "something the Raft question had was displaced"
    assert "pushed out of the top" in after.steps[2].detail
    assert evicted[0] in after.steps[2].detail, "the displaced passage is named in the trace"


def test_clearing_feedback_returns_the_ranking_to_cold_start() -> None:
    top = candidates(QUERY)[0]
    cold = ids(FeedbackRAG().run(QUERY))
    vote(QUERY, [top.chunk_id], -1, times=CAP)
    assert ids(FeedbackRAG().run(QUERY)) != cold

    deleted = clear_feedback()

    assert deleted == CAP
    assert ids(FeedbackRAG().run(QUERY)) == cold


# --- the positional-id hazard ----------------------------------------------


def test_a_vote_stops_counting_when_the_text_at_that_id_changes(temp_db: Path) -> None:
    """A re-index can put different text at `dynamo.md#1`. The vote must not follow."""
    top = candidates(QUERY)[0]
    vote(QUERY, [top.chunk_id], -1, times=CAP)
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE feedback SET chunk_hash = ?", (chunk_hash("some other text"),))

    result = FeedbackRAG().run(QUERY)

    assert ids(result) == ids(StandardRAG().run(QUERY))
    assert result.metadata.feedback_votes == 0


def test_a_vote_follows_its_text_to_a_different_id(temp_db: Path) -> None:
    """The positive half: ids are positional, text is not. A passage that merely
    moved keeps its votes, which is why the hash is the key."""
    top = candidates(QUERY)[0]
    vote(QUERY, [top.chunk_id], -1, times=CAP)
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE feedback SET chunk_id = 'renamed.md#99'")

    result = FeedbackRAG().run(QUERY)

    assert top.chunk_id not in ids(result)
    assert result.metadata.feedback_votes == CAP


# --- the reranker itself ---------------------------------------------------


def fake_chunks(count: int) -> list[Chunk]:
    return [
        Chunk(text=f"text {i}", source="x.md", score=1.0 - i / 100, chunk_id=f"x.md#{i}")
        for i in range(1, count + 1)
    ]


def test_rerank_breaks_ties_in_the_retrievers_favour() -> None:
    chunks = fake_chunks(FEEDBACK_CANDIDATES)
    # One vote buys SHIFT places, so this chunk lands level with the one SHIFT
    # ranks above it. Derived from the constant rather than written out, or a
    # change to SHIFT would fail here with arithmetic instead of with a reason.
    voted, tied = 1 + SHIFT, 1
    net = {chunk_hash(f"text {voted}"): 1}

    order = [c.chunk_id for c in rerank(chunks, net)]

    assert order[:2] == [f"x.md#{tied}", f"x.md#{voted}"], "a tie goes to the retriever"


def test_rerank_with_several_voted_chunks_is_not_the_lone_vote_arithmetic() -> None:
    """Reachability is a property of the whole list, not of one passage.

    A lone +3 reaches the top-4 only from rank 9 or better, but with the ranks
    above it voted down, rank 12 gets in — which is why every number the trace
    prints is read off the final list.
    """
    chunks = fake_chunks(FEEDBACK_CANDIDATES)
    net = {chunk_hash("text 1"): -3, chunk_hash("text 2"): -3, chunk_hash("text 3"): -3}

    lone = [c.chunk_id for c in rerank(chunks, {chunk_hash("text 12"): 3})]
    crowded = [c.chunk_id for c in rerank(chunks, {**net, chunk_hash("text 12"): 3})]

    assert "x.md#12" not in lone[: settings.top_k]
    assert "x.md#12" in crowded[: settings.top_k]


# --- the API ---------------------------------------------------------------


def test_posting_feedback_stores_one_row_per_passage(temp_db: Path) -> None:
    chunk_ids = [c.chunk_id for c in candidates(QUERY)[:2]]

    response = client.post(
        "/api/feedback",
        json={"technique": "feedback-rag", "query": QUERY, "chunk_ids": chunk_ids, "rating": -1},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert stored_rows(temp_db) == 2


def test_an_unknown_passage_is_rejected_before_anything_is_written(temp_db: Path) -> None:
    good = candidates(QUERY)[0].chunk_id

    response = client.post(
        "/api/feedback",
        json={
            "technique": "feedback-rag",
            "query": QUERY,
            "chunk_ids": [good, "nosuch.md#9"],
            "rating": 1,
        },
    )

    assert response.status_code == 422
    assert "nosuch.md#9" in response.json()["detail"]
    assert stored_rows(temp_db) == 0, "a partial batch would store an opinion nobody expressed"


def test_duplicate_ids_in_one_request_are_rejected(temp_db: Path) -> None:
    chunk_id = candidates(QUERY)[0].chunk_id

    response = client.post(
        "/api/feedback",
        json={
            "technique": "feedback-rag",
            "query": QUERY,
            "chunk_ids": [chunk_id, chunk_id],
            "rating": 1,
        },
    )

    assert response.status_code == 422
    assert stored_rows(temp_db) == 0


def test_another_technique_cannot_be_voted_on(temp_db: Path) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "technique": "standard-rag",
            "query": QUERY,
            "chunk_ids": [candidates(QUERY)[0].chunk_id],
            "rating": 1,
        },
    )

    assert response.status_code == 409
    assert stored_rows(temp_db) == 0


def test_an_unknown_technique_is_a_404() -> None:
    response = client.post(
        "/api/feedback",
        json={"technique": "nope-rag", "query": QUERY, "chunk_ids": ["a"], "rating": 1},
    )

    assert response.status_code == 404


def test_a_rating_that_is_not_a_thumb_is_rejected() -> None:
    response = client.post(
        "/api/feedback",
        json={"technique": "feedback-rag", "query": QUERY, "chunk_ids": ["a"], "rating": 0},
    )

    assert response.status_code == 422


def test_the_run_endpoint_reports_the_votes_it_applied() -> None:
    chunk_id = candidates(QUERY)[0].chunk_id
    vote(QUERY, [chunk_id], -1, times=2)

    response = client.post("/api/run", json={"technique": "feedback-rag", "query": QUERY})

    assert response.status_code == 200
    assert response.json()["metadata"]["feedback_votes"] == 2


def test_other_techniques_report_no_votes() -> None:
    response = client.post("/api/run", json={"technique": "standard-rag", "query": QUERY})

    assert response.json()["metadata"]["feedback_votes"] == 0
