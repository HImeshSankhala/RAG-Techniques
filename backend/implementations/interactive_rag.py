"""Interactive RAG — a draft, a human's judgment on its evidence, then a final answer.

The first technique that is not a function of its input. Two requests, with an
unbounded human-shaped wait between them:

    run(query)                      -> Standard RAG's answer, saved as a draft
    finalize(draft_id, kept, hint)  -> answer again from the evidence the human kept,
                                       plus whatever the hint retrieves

The draft is Standard RAG, composed rather than copied: without a human this
technique IS the baseline, and saying so in code keeps the two from drifting.

WHERE THE DRAFT LIVES, AND WHY THERE
Between the two requests the draft's passages have to survive somewhere. The
stateless alternative — send them to the browser and have it send them back —
needs no storage, but then the client decides what text reaches a (possibly
paid) prompt. Holding the draft server-side in SQLite means the final prompt can
only contain passages this server retrieved, at the price of a TTL and a
"draft expired" path.

WHAT THE HUMAN'S INPUT DOES
* Kept passages are taken from the stored draft, not re-fetched: the answer is
  built from exactly the text the human read.
* An unticked passage is never re-admitted, even when the hint's search returns
  it again. Re-retrieval silently undoing "this is irrelevant" would make the
  checkbox a suggestion.
* The hint steers retrieval only. The final prompt asks the user's original
  question, so draft and final differ in evidence and nothing else — and a user
  cannot assert a fact into the answer through the hint box.
* Kept everything, no hint: the evidence is byte-identical to the draft's, so a
  second generation could only differ by sampling noise, and the UI would present
  that noise as the human's doing. Returned unchanged, zero calls, `no_change`.
"""

import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, replace

from core import llm, retrieval
from core.config import settings
from core.ledger import LLMLedger
from core.pipeline import Chunk, RAGPipeline, RAGResult, Step, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness
from implementations.standard_rag import StandardRAG

# How long a draft waits for its human. A constant, not config: nothing needs to
# tune it, and a draft older than an hour is a tab someone forgot.
DRAFT_TTL_SECONDS = 60 * 60

# JSON for the chunks rather than a row per chunk: nothing ever queries inside a
# draft — it is written once and read back whole.
_SCHEMA = """CREATE TABLE IF NOT EXISTS drafts (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    query       TEXT NOT NULL,
    model       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    chunks_json TEXT NOT NULL
)"""


class DraftNotFoundError(LookupError):
    """No draft with that id, or it outlived its TTL."""


class InvalidSelectionError(ValueError):
    """The human's selection leaves nothing to answer from, or names foreign chunks."""


class StaleDraftError(RuntimeError):
    """The index was rebuilt after the draft was made, so its chunk ids now name other text."""


@dataclass(frozen=True)
class _Draft:
    created_at: float
    query: str
    model: str
    answer: str
    chunks: list[Chunk]


class InteractiveRAG(RAGPipeline):
    name = "interactive-rag"

    def run(self, query: str, model: str | None = None) -> RAGResult:
        """The draft half: Standard RAG's answer, saved for the human to review."""
        draft = StandardRAG().run(query, model)
        if not draft.retrieved_chunks:
            # Empty index. There is nothing for a human to review, so no draft.
            return draft

        steps = StepRecorder()
        draft_id = uuid.uuid4().hex
        with steps.record("Save draft") as step:
            _save(draft_id, query, draft)
            step.detail = (
                f"draft {draft_id[:8]}, expires in {DRAFT_TTL_SECONDS // 60} min — "
                "untick passages that do not help, add a hint for what is missing"
            )

        return replace(
            draft,
            steps=draft.steps + steps.steps,
            metadata=replace(
                draft.metadata,
                latency_ms=round(draft.metadata.latency_ms + steps.elapsed_ms, 2),
                termination_reason="awaiting_feedback",
            ),
            draft_id=draft_id,
        )


def finalize(draft_id: str, keep_chunk_ids: list[str], hint: str) -> tuple[str, RAGResult]:
    """The final half. Returns the draft's query alongside the result.

    Every rejection happens before any LLM call. `llm_calls`, tokens, cost and
    latency cover this leg only — the draft's panel already reports its own call.
    The exception is `retrieval_passes`, which counts the retrieval behind the
    evidence and so includes the draft's: a click-through final reports 1 pass
    for a leg that retrieved nothing.
    """
    steps = StepRecorder()
    draft = _load(draft_id)
    hint = hint.strip()

    draft_ids = {c.chunk_id for c in draft.chunks}
    foreign = sorted(set(keep_chunk_ids) - draft_ids)
    if foreign:
        raise InvalidSelectionError(f"Not passages from this draft: {', '.join(foreign)}.")

    keep = set(keep_chunk_ids)
    kept = [c for c in draft.chunks if c.chunk_id in keep]
    dropped = [c for c in draft.chunks if c.chunk_id not in keep]
    if not kept and not hint:
        raise InvalidSelectionError(
            "Nothing to answer from: keep at least one passage or add a hint."
        )

    ledger = LLMLedger(draft.model)
    unchanged = not dropped and not hint

    # Recorded by hand: its duration is the human's, measured from the draft's
    # creation (so it also spans any earlier finalize of the same draft), and it
    # must not come from — or count toward — this leg's machine latency.
    waited_s = time.time() - draft.created_at
    steps.steps.append(
        Step(
            name="Human review",
            detail=_review_detail(kept, dropped, hint, waited_s, unchanged),
            duration_ms=round(waited_s * 1000, 2),
        )
    )

    if unchanged:
        return draft.query, RAGResult(
            answer=draft.answer,
            retrieved_chunks=draft.chunks,
            steps=steps.steps,
            metadata=ledger.metadata(
                latency_ms=steps.elapsed_ms,
                retrieval_passes=1,
                termination_reason="no_change",
                groundedness=groundedness(draft.answer, draft.chunks),
            ),
        )

    evidence = list(kept)
    if hint:
        with steps.record("Retrieve with hint") as step:
            new, step.detail = _hint_chunks(draft, hint)
            evidence.extend(new)

    if not evidence:
        raise InvalidSelectionError(
            "The hint found no passages beyond the draft's, and none were kept. "
            "Keep at least one passage or try a different hint."
        )

    with steps.record(f"Final answer from {len(evidence)} chunks") as step:
        response = ledger.record(
            llm.generate(SYSTEM_PROMPT, build_prompt(draft.query, evidence), model=draft.model)
        )
        step.detail = (
            f"{response.model} ({response.backend}): "
            f"{response.input_tokens} in / {response.output_tokens} out"
        )

    return draft.query, RAGResult(
        answer=response.text,
        retrieved_chunks=evidence,
        steps=steps.steps,
        metadata=ledger.metadata(
            latency_ms=steps.elapsed_ms,
            # The retrievals behind this evidence: the draft's, plus the hint's.
            retrieval_passes=1 + (1 if hint else 0),
            termination_reason="human_feedback",
            groundedness=groundedness(response.text, evidence),
        ),
    )


def _hint_chunks(draft: _Draft, hint: str) -> tuple[list[Chunk], str]:
    """Up to top_k passages the draft did not have, for the query plus the hint,
    with the trace detail describing what the search found.

    Over-fetches by the draft's size: a few words appended to the query move the
    embedding only a little, so the plain top_k would mostly be the draft again,
    and filtering it out would leave nothing.
    """
    by_id = {c.chunk_id: c for c in draft.chunks}
    search = f"{draft.query} {hint}"
    found = retrieval.dense(search, settings.top_k + len(draft.chunks))

    for chunk in found:
        # Chunk ids are positional (`source#n`) and `make index` rebuilds them,
        # so after a re-index a draft's id can name text the human never saw.
        if chunk.chunk_id in by_id and by_id[chunk.chunk_id].text != chunk.text:
            raise StaleDraftError(
                "The index was rebuilt after this draft was made, so its passages no "
                "longer match. Run the query again."
            )

    unseen = [c for c in found if c.chunk_id not in by_id]
    new = unseen[: settings.top_k]
    detail = (
        f"{len(found)} fetched for {search!r}: {len(found) - len(unseen)} already in the "
        f"draft (kept or dropped), {len(new)} new taken: "
        f"{', '.join(c.chunk_id for c in new) or 'none'}"
        + (
            f", {len(unseen) - len(new)} more beyond top-{settings.top_k}"
            if len(unseen) > len(new)
            else ""
        )
    )
    return new, detail


def _review_detail(
    kept: list[Chunk], dropped: list[Chunk], hint: str, waited_s: float, unchanged: bool
) -> str:
    """What the human changed, in the trace's words."""
    waited = f"{waited_s:.1f}s since the draft was created (not counted in latency)"
    if unchanged:
        return (
            f"kept all {len(kept)} passages and added no hint, so the evidence is "
            f"unchanged and so is the answer — no LLM call made · {waited}"
        )
    kept_ids = ", ".join(c.chunk_id for c in kept) or "none"
    dropped_ids = ", ".join(c.chunk_id for c in dropped) or "none"
    hint_text = f"hint {hint!r}" if hint else "no hint"
    return (
        f"kept {len(kept)} of {len(kept) + len(dropped)} ({kept_ids}); "
        f"dropped {dropped_ids}; {hint_text} · {waited}"
    )


def _connect() -> sqlite3.Connection:
    """A fresh connection per operation.

    FastAPI runs sync handlers on a thread pool, and a sqlite3 connection belongs
    to the thread that opened it. Callers wrap this in `closing()`: using the
    connection itself as a context manager commits but does NOT close it.
    """
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(_SCHEMA)
    except BaseException:
        conn.close()
        raise
    return conn


def _save(draft_id: str, query: str, result: RAGResult) -> None:
    now = time.time()
    with closing(_connect()) as conn, conn:
        # Lazy expiry: sweep on write instead of running a cleanup job. O(n) over
        # a table that holds an hour of drafts at most.
        conn.execute("DELETE FROM drafts WHERE created_at < ?", (now - DRAFT_TTL_SECONDS,))
        conn.execute(
            "INSERT INTO drafts VALUES (?, ?, ?, ?, ?, ?)",
            (
                draft_id,
                now,
                query,
                result.metadata.model,
                result.answer,
                json.dumps([asdict(c) for c in result.retrieved_chunks]),
            ),
        )


def _load(draft_id: str) -> _Draft:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT created_at, query, model, answer, chunks_json FROM drafts WHERE id = ?",
            (draft_id,),
        ).fetchone()

    # Checked on read too: the sweep only runs when a draft is written, so an
    # expired row can still be sitting there.
    if row is None or time.time() - row[0] > DRAFT_TTL_SECONDS:
        raise DraftNotFoundError("That draft has expired or does not exist. Run the query again.")

    created_at, query, model, answer, chunks_json = row
    return _Draft(
        created_at=created_at,
        query=query,
        model=model,
        answer=answer,
        chunks=[Chunk(**c) for c in json.loads(chunks_json)],
    )
