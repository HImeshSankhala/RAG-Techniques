"""Interactive RAG against the real corpus, with the LLM scripted and drafts in a temp DB.

Retrieval stays real for the same reason as test_agentic_rag.py: "the hint found
new passages" and "an unticked passage is never re-admitted" only mean something
against the actual index.

Requires the index: run `make index` first.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import llm, vectorstore
from core.config import settings
from core.llm import LLMResponse, estimate_cost_usd
from implementations import interactive_rag
from implementations.interactive_rag import (
    DraftNotFoundError,
    InteractiveRAG,
    InvalidSelectionError,
    StaleDraftError,
    finalize,
)
from implementations.standard_rag import StandardRAG

QUERY = "How does Dynamo handle conflicting concurrent writes?"
HINT = "Raft leader election"

client = TestClient(app)


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture(autouse=True)
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "drafts.db"
    monkeypatch.setattr(settings, "db_path", path)
    return path


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch):
    """Queue replies the fake LLM returns in order, and record the prompts it saw."""

    def install(*replies: str, backend: str = "ollama") -> list[str]:
        remaining = list(replies)
        prompts: list[str] = []

        def fake_generate(system, user, model=None, **kwargs) -> LLMResponse:
            prompts.append(user)
            assert remaining, "pipeline made more LLM calls than the script supplied"
            return LLMResponse(
                text=remaining.pop(0),
                input_tokens=100,
                output_tokens=20,
                model=model or "stub-model",
                backend=backend,
            )

        monkeypatch.setattr(llm, "generate", fake_generate)
        return prompts

    return install


def test_the_draft_is_standard_rag_saved_for_review(script, temp_db: Path) -> None:
    script("Draft citing dynamo.md.", "Standard citing dynamo.md.")
    draft = InteractiveRAG().run(QUERY)
    standard = StandardRAG().run(QUERY)

    assert draft.draft_id
    assert draft.metadata.termination_reason == "awaiting_feedback"
    assert draft.metadata.llm_calls == 1
    assert [c.chunk_id for c in draft.retrieved_chunks] == [
        c.chunk_id for c in standard.retrieved_chunks
    ]
    assert draft.steps[-1].name == "Save draft"
    with sqlite3.connect(temp_db) as conn:
        assert conn.execute("SELECT count(*) FROM drafts").fetchone() == (1,)


def test_dropped_passages_leave_the_final_prompt(script) -> None:
    prompts = script("Draft.", "Final citing dynamo.md.")
    draft = InteractiveRAG().run(QUERY)
    keep, drop = draft.retrieved_chunks[:2], draft.retrieved_chunks[2:]

    query, final = finalize(draft.draft_id, [c.chunk_id for c in keep], "")

    assert query == QUERY
    assert final.retrieved_chunks == keep
    for chunk in keep:
        assert chunk.text[:200] in prompts[-1]
    for chunk in drop:
        assert chunk.text[:200] not in prompts[-1]
    assert prompts[-1].endswith(f"Question: {QUERY}")
    # Final leg only: the draft's own panel reports the draft's call.
    assert final.metadata.llm_calls == 1
    assert final.metadata.tokens_in == 100
    assert final.metadata.retrieval_passes == 1
    assert final.metadata.termination_reason == "human_feedback"
    assert [s.name for s in final.steps][0] == "Human review"


def test_a_hint_retrieves_new_passages_but_never_readmits_a_dropped_one(script) -> None:
    prompts = script("Draft.", "Final.")
    draft = InteractiveRAG().run(QUERY)
    dropped = draft.retrieved_chunks[-1]
    keep = [c.chunk_id for c in draft.retrieved_chunks[:-1]]

    _, final = finalize(draft.draft_id, keep, HINT)

    ids = [c.chunk_id for c in final.retrieved_chunks]
    assert dropped.chunk_id not in ids
    assert len(ids) == len(set(ids))
    assert len(ids) > len(keep), "the hint should have found passages the draft lacked"
    assert len(ids) <= len(keep) + settings.top_k
    assert HINT not in prompts[-1], "the hint steers retrieval only"
    assert final.metadata.retrieval_passes == 2
    assert "Retrieve with hint" in [s.name for s in final.steps]


def test_keeping_everything_without_a_hint_makes_no_call(script) -> None:
    prompts = script("Draft citing dynamo.md.")
    draft = InteractiveRAG().run(QUERY)

    _, final = finalize(draft.draft_id, [c.chunk_id for c in draft.retrieved_chunks], "  ")

    assert len(prompts) == 1, "only the draft's call"
    assert final.answer == draft.answer
    assert final.retrieved_chunks == draft.retrieved_chunks
    assert final.metadata.termination_reason == "no_change"
    assert final.metadata.llm_calls == 0


def test_a_draft_can_be_finalized_more_than_once(script) -> None:
    script("Draft.", "First final.", "Second final.")
    draft = InteractiveRAG().run(QUERY)
    ids = [c.chunk_id for c in draft.retrieved_chunks]

    assert finalize(draft.draft_id, ids[:1], "")[1].answer == "First final."
    assert finalize(draft.draft_id, ids[:2], "")[1].answer == "Second final."


def test_unknown_and_expired_drafts_are_not_found(script, monkeypatch: pytest.MonkeyPatch) -> None:
    script("Draft.")
    draft = InteractiveRAG().run(QUERY)

    with pytest.raises(DraftNotFoundError):
        finalize("no-such-draft", [], "hint")

    monkeypatch.setattr(interactive_rag, "DRAFT_TTL_SECONDS", -1)
    with pytest.raises(DraftNotFoundError):
        finalize(draft.draft_id, [draft.retrieved_chunks[0].chunk_id], "")


def test_invalid_selections_are_rejected_before_any_call(script) -> None:
    prompts = script("Draft.")
    draft = InteractiveRAG().run(QUERY)

    with pytest.raises(InvalidSelectionError):
        finalize(draft.draft_id, ["not-from-this-draft.md#0"], "")
    with pytest.raises(InvalidSelectionError):
        finalize(draft.draft_id, [], "   ")

    assert len(prompts) == 1


def test_a_reindex_since_the_draft_is_detected(script, temp_db: Path) -> None:
    """Chunk ids are positional; after a rebuild a draft id can name new text.
    Simulated by rewriting the stored texts, which is the same mismatch."""
    prompts = script("Draft.")
    draft = InteractiveRAG().run(QUERY)
    with sqlite3.connect(temp_db) as conn:
        chunks = json.loads(conn.execute("SELECT chunks_json FROM drafts").fetchone()[0])
        for chunk in chunks:
            chunk["text"] = "text from before the re-index"
        conn.execute("UPDATE drafts SET chunks_json = ?", (json.dumps(chunks),))

    with pytest.raises(StaleDraftError):
        finalize(draft.draft_id, [draft.retrieved_chunks[0].chunk_id], HINT)
    assert len(prompts) == 1


def test_final_cost_is_the_final_legs_own(script) -> None:
    """Ledger arithmetic on a paid-labelled stub. The stub bypasses llm.py, so
    this does NOT exercise .usage.json or /api/usage."""
    script("Draft.", "Final.", backend="anthropic")
    draft = InteractiveRAG().run(QUERY)

    _, final = finalize(draft.draft_id, [draft.retrieved_chunks[0].chunk_id], "")

    assert final.metadata.cost_estimate_usd == round(estimate_cost_usd(100, 20), 6)


def test_an_empty_index_saves_no_draft(
    script, monkeypatch: pytest.MonkeyPatch, temp_db: Path
) -> None:
    script()
    monkeypatch.setattr(vectorstore, "query", lambda embedding, top_k: [])

    result = InteractiveRAG().run(QUERY)

    assert result.draft_id is None
    assert result.metadata.termination_reason == "empty_index"
    assert not temp_db.exists(), "no draft means the store is never opened"


# --- API -----------------------------------------------------------------------


def test_run_passes_the_draft_id_through(script) -> None:
    script("Draft.")
    body = client.post("/api/run", json={"technique": "interactive-rag", "query": QUERY}).json()
    assert body["draft_id"]
    assert body["metadata"]["termination_reason"] == "awaiting_feedback"


def test_final_endpoint_round_trip_and_error_mapping(script) -> None:
    script("Draft.", "Final.")
    draft = client.post("/api/run", json={"technique": "interactive-rag", "query": QUERY}).json()
    first = draft["retrieved_chunks"][0]["chunk_id"]

    final = client.post(
        "/api/run/final", json={"draft_id": draft["draft_id"], "chunk_ids": [first]}
    )
    assert final.status_code == 200
    assert final.json()["query"] == QUERY
    assert final.json()["draft_id"] is None

    missing = client.post("/api/run/final", json={"draft_id": "nope", "chunk_ids": [first]})
    assert missing.status_code == 404

    empty = client.post(
        "/api/run/final", json={"draft_id": draft["draft_id"], "chunk_ids": [], "hint": " "}
    )
    assert empty.status_code == 422
