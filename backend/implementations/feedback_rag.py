"""Feedback-Based RAG — thumbs on passages, stored, reranking every later run.

Interactive RAG asks a human and forgets. This one remembers: a vote is a row in
SQLite, and every future run of this technique reads those rows back.

    run(query)                       -> dense top-12, reranked by stored votes, top-4 answered
    record_feedback(..., rating)     -> one row per passage, from any later request

WHY A RANK SHIFT RATHER THAN `similarity + weight`
The obvious formula adds a weight to the cosine score. It is the same mistake
`core/fusion.py` exists to avoid: the numbers live on different scales. Measured
on this corpus, the top-12 cosine band is 0.136-0.256 for "What is hinted
handoff?" and 0.452-0.762 for "How does Dynamo handle conflicting concurrent
writes?" — so one fixed weight would decide the first query's ranking outright
and do nothing to the second's. Ranks are comparable across queries, so a vote
buys a fixed number of PLACES instead of a fixed number of score points, and the
trace can say exactly that.

WHY THE OVER-FETCH
Reranking needs somewhere to rank from. Fetching the final top-4 and reordering
it could only permute what Standard RAG already had; fetching 12 means feedback
can pull a passage up into the answer, which is the whole point. It also sets a
hard recall limit: a passage below rank 12 is not a candidate, and no number of
votes can reach it. That is a limit of the over-fetch, not of the feedback.

THE FILTER BUBBLE, WHICH IS THE LESSON AND NOT A BUG
Votes are global — keyed to the passage, not the query — so a correct judgment
on one question silently reranks every other question. Measured on this corpus:
upvoting `mapreduce.md#2` (the one passage that mentions stragglers) three times
for the MapReduce question moves it from rank 8 to rank 3 for "How does Raft
elect a leader?", evicting `raft.md#1`, which is the passage headed "Leader
election". Nothing here mitigates that: no decay, no exploration slot, no
normalisation by exposure. The trace names what moved and what was displaced, so
a reader can watch it happen rather than take the learn page's word for it.

WHAT IS DELIBERATELY NOT HERE
No dedupe: every click is a row, so re-voting the same passage amplifies it, which
is how a small number of judgments reaches the cap. No decay and no exploration —
see above. The store is shared with Phase 10's drafts (`settings.db_path`); the
votes have no TTL, because "remembers" is the technique.
"""

import hashlib
import sqlite3
import time
from contextlib import closing
from typing import NamedTuple

from core import embeddings, llm, vectorstore
from core.config import settings
from core.ledger import LLMLedger
from core.pipeline import Chunk, RAGPipeline, RAGResult, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness

# How many passages feedback gets to reorder. Its own constant rather than
# `retrieval.CANDIDATE_MULTIPLIER`: that one is justified for fusion's merge, and
# retuning fusion must not silently change how far a vote can move a passage here.
FEEDBACK_CANDIDATES = 12

# The most a passage's net votes can count, and how many places one net vote buys.
# Together they are the "most consequential number" on the learn page: with
# SHIFT=2 and CAP=3 a lone vote reaches the top-4 from as deep as rank 9, and the
# retriever's own #1 can be voted out. Set deliberately high so the failure mode
# is visible in a demo rather than theoretical.
CAP = 3
SHIFT = 2

_SCHEMA = """CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY,
    created_at  REAL NOT NULL,
    technique   TEXT NOT NULL,
    query       TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    chunk_hash  TEXT NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating IN (-1, 1))
)"""

# Votes are read by hash on every run, one query per run over ≤12 hashes.
_INDEX = "CREATE INDEX IF NOT EXISTS feedback_by_hash ON feedback(chunk_hash)"


class UnknownChunkError(ValueError):
    """A voted passage is not in the index — a stale page, or a bad request."""


class StoredVotes(NamedTuple):
    """What the vote store says about one run's candidates."""

    net: dict[str, int]  # chunk hash -> clamped net rating
    rows: int  # how many stored votes matched, before clamping


class FeedbackRAG(RAGPipeline):
    name = "feedback-rag"

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model
        ledger = LLMLedger(model)

        with steps.record("Embed query") as step:
            vector = embeddings.embed_query(query)
            step.detail = f"{len(vector)}-dimensional vector"

        with steps.record("Retrieve candidates") as step:
            candidates = vectorstore.query(vector, FEEDBACK_CANDIDATES)
            step.detail = (
                f"top {len(candidates)} of {vectorstore.count()} chunks "
                f"(best score {candidates[0].score}) — {settings.top_k} will be answered from"
                if candidates
                else "no chunks found — is the index built? (make index)"
            )

        if not candidates:
            return RAGResult(
                answer=(
                    "Nothing is indexed yet, so there is no context to answer from. "
                    "Run `make index` and try again."
                ),
                steps=steps.steps,
                # Backend derived, not reported — no call was made. See standard_rag.
                metadata=ledger.metadata(
                    latency_ms=steps.elapsed_ms,
                    retrieval_passes=1,
                    termination_reason="empty_index",
                ),
            )

        with steps.record("Apply feedback") as step:
            votes = stored_votes(candidates)
            ordered = rerank(candidates, votes.net)
            chunks = ordered[: settings.top_k]
            step.detail = _feedback_detail(candidates, ordered, votes)

        with steps.record("Generate answer") as step:
            response = ledger.record(
                llm.generate(SYSTEM_PROMPT, build_prompt(query, chunks), model=model)
            )
            step.detail = (
                f"{response.model} ({response.backend}): "
                f"{response.input_tokens} in / {response.output_tokens} out"
            )

        return RAGResult(
            answer=response.text,
            retrieved_chunks=chunks,
            steps=steps.steps,
            metadata=ledger.metadata(
                latency_ms=steps.elapsed_ms,
                # One retrieval. What differs from Standard RAG is the order it is
                # read in, not how many times the index was asked.
                retrieval_passes=1,
                termination_reason="single_pass",
                groundedness=groundedness(response.text, chunks),
                feedback_votes=votes.rows,
            ),
        )


def rerank(candidates: list[Chunk], net: dict[str, int]) -> list[Chunk]:
    """Reorder dense candidates by their stored votes. Pure: no I/O.

        key(c) = (dense_rank(c) - SHIFT * clamp(net(c), -CAP, CAP), dense_rank(c))

    Ties fall back to the dense rank, so feedback nudges the retriever's order and
    never wins a tie against it.

    The key is NOT a rank: a passage whose key is 7 can still finish 6th once the
    ranks it passed are accounted for. Anything shown to a reader is read back off
    the returned list.

    Complexity: O(n log n) over the candidates, which is 12.
    """
    ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(candidates, start=1)}
    return sorted(
        candidates,
        key=lambda chunk: (
            ranks[chunk.chunk_id] - SHIFT * _clamped(net.get(chunk_hash(chunk.text), 0)),
            ranks[chunk.chunk_id],
        ),
    )


def chunk_hash(text: str) -> str:
    """Identity of a passage BY ITS TEXT, which is what a vote is really about.

    The obvious key is `chunk_id`, and it is wrong: ids are positional
    (`dynamo.md#3`, core/index.py) and `make index` rebuilds them, so after an edit
    a stored vote would silently apply to a passage nobody rated. A content hash
    follows the text when it merely moves, and stops counting when it changes.

    Truncated to 16 hex chars (64 bits): collisions are not a security boundary
    here, only a question of whether two different passages could share votes.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stored_votes(candidates: list[Chunk]) -> StoredVotes:
    """Clamped net rating per candidate hash, plus how many rows were counted.

    One indexed query per run over at most 12 hashes. Votes on passages outside
    the candidate set are never loaded — they could not move anything.
    """
    hashes = {chunk_hash(chunk.text) for chunk in candidates}
    placeholders = ",".join("?" * len(hashes))
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT chunk_hash, SUM(rating), COUNT(*) FROM feedback "
            f"WHERE chunk_hash IN ({placeholders}) GROUP BY chunk_hash",
            tuple(hashes),
        ).fetchall()

    return StoredVotes(
        net={chunk_hash_: _clamped(total) for chunk_hash_, total, _ in rows},
        rows=sum(count for _, _, count in rows),
    )


def record_feedback(technique: str, query: str, chunk_ids: list[str], rating: int) -> None:
    """Store one vote per passage. Every click counts — there is no dedupe.

    The passage text is read from the index HERE rather than taken from the
    request, so a client cannot vote on text this server never retrieved. The
    unavoidable race: if the index is rebuilt between a result being shown and its
    vote arriving, the vote is hashed against the new text at that id. Accepted
    rather than mitigated — the API contract carries no text to check against.

    Raises UnknownChunkError before writing anything, so a bad id in a batch never
    leaves half the votes stored.
    """
    texts = {chunk.chunk_id: chunk.text for chunk in vectorstore.all_chunks()}
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in texts]
    if missing:
        raise UnknownChunkError(
            f"Not in the index: {', '.join(missing)}. The index may have been rebuilt — "
            "run the query again."
        )

    now = time.time()
    with closing(_connect()) as conn, conn:
        conn.executemany(
            "INSERT INTO feedback (created_at, technique, query, chunk_id, chunk_hash, rating) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (now, technique, query, chunk_id, chunk_hash(texts[chunk_id]), rating)
                for chunk_id in chunk_ids
            ],
        )


def clear_feedback() -> int:
    """Delete every vote and return how many were deleted (`make reset-feedback`).

    The demo is only repeatable if the state can be wound back, and a reader who
    has demoted a passage has no way to un-demote it: it is no longer shown, so it
    has no thumbs. That is the calcification the learn page describes, so the way
    out is a maintenance command rather than a button in the UI.
    """
    with closing(_connect()) as conn, conn:
        return conn.execute("DELETE FROM feedback").rowcount


def _clamped(total: int) -> int:
    """Net votes, capped. The cap is what keeps one loud passage from pinning itself."""
    return max(-CAP, min(CAP, total))


def _feedback_detail(candidates: list[Chunk], ordered: list[Chunk], votes: StoredVotes) -> str:
    """What the votes did to this ranking, in the trace's words.

    Every position is read off `ordered`, never off the sort key, and displaced
    passages are named too: a chunk that lost its place did not do anything wrong,
    and it would otherwise vanish from the run with no explanation.
    """
    top_k = settings.top_k
    outside = (
        f"votes on passages outside the top {len(candidates)} are not consulted; "
        "not mitigated here: decay, exploration slots, normalisation by exposure"
    )
    if not votes.net:
        return (
            f"no stored votes on these {len(candidates)} candidates — this ranking is "
            f"Standard RAG's (cold start) · {outside}"
        )

    before = {chunk.chunk_id: rank for rank, chunk in enumerate(candidates, start=1)}
    after = {chunk.chunk_id: rank for rank, chunk in enumerate(ordered, start=1)}
    moves = [
        f"{chunk.chunk_id} net {votes.net[chunk_hash(chunk.text)]:+d}: rank "
        f"{before[chunk.chunk_id]} → {after[chunk.chunk_id]}"
        f"{_arrow(before[chunk.chunk_id], after[chunk.chunk_id])}"
        for chunk in candidates
        if chunk_hash(chunk.text) in votes.net
    ]
    displaced = [
        f"{chunk.chunk_id} (rank {before[chunk.chunk_id]} → {after[chunk.chunk_id]})"
        for chunk in candidates[:top_k]
        if after[chunk.chunk_id] > top_k and chunk_hash(chunk.text) not in votes.net
    ]

    detail = f"{votes.rows} stored vote(s) on {len(moves)} candidate(s): {'; '.join(moves)}"
    if displaced:
        detail += f" · pushed out of the top {top_k} by other passages' votes: {', '.join(displaced)}"
    return f"{detail} · {outside}"


def _arrow(before: int, after: int) -> str:
    if after < before:
        return " ↑"
    return " ↓" if after > before else " ="


def _connect() -> sqlite3.Connection:
    """A fresh connection per operation, as in interactive_rag: a sqlite3
    connection belongs to the thread that opened it, and FastAPI runs sync
    handlers on a thread pool. Callers wrap this in `closing()` — using the
    connection as a context manager commits but does NOT close it.
    """
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(_SCHEMA)
        conn.execute(_INDEX)
    except BaseException:
        conn.close()
        raise
    return conn
