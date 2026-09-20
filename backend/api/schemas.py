"""Pydantic schemas — the typed contract between the API and the frontend.

These are the source of truth. `frontend/lib/api.ts` mirrors them by hand; when a
schema changes here, that file changes too or the frontend is lying about the API.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

from core.config import settings

# `min_length` on a plain `str` counts characters, not content, so "   " passed
# validation and reached the model — which then invented its own question and
# answered that. Stripping first makes the length check measure what was actually
# asked. One alias rather than two Field definitions: the two request bodies have
# to agree on what a query is, and duplicating the constraint invites them to drift.
Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Technique(BaseModel):
    """One entry in GET /api/techniques."""

    name: str = Field(description="Slug, e.g. 'fusion-rag'. Matches the MDX filename.")
    display_name: str = Field(description="Human-readable name for cards and selectors.")
    tagline: str = Field(description="One-line summary shown on the technique card.")
    implemented: bool = Field(
        description="True when the technique has a runnable pipeline. False = docs only."
    )
    needs_human: bool = Field(
        description="True when a run pauses for a person (Interactive RAG). "
        "Such techniques run in the playground but cannot be compared."
    )
    docs_only: bool = Field(
        description="True when the technique can never run here, however much is built "
        "(REALM is a pre-training method). Different from implemented=False, which only "
        "means no pipeline exists yet."
    )
    llm_calls_range: str = Field(
        description="Editorial range of model calls per query, e.g. '2-5'. Prose for the "
        "home comparison table, not a measurement — Metadata.llm_calls counts one run."
    )
    retrieval_passes_range: str = Field(
        description="Editorial range of retrieval passes per query. Same caveat as "
        "llm_calls_range."
    )
    upload_note: str = Field(
        description="Why this technique cannot run against uploaded documents, or '' "
        "when it can. Same string the 409 from POST /api/run uses."
    )


class Chunk(BaseModel):
    """A retrieved passage. Mirrors `core.pipeline.Chunk`."""

    text: str
    source: str = Field(description="Filename the passage was chunked out of.")
    score: float = Field(description="Relevance; higher is better.")
    chunk_id: str


class Step(BaseModel):
    """One stage of a run, for the UI trace. Mirrors `core.pipeline.Step`."""

    name: str
    detail: str
    duration_ms: float


class Metadata(BaseModel):
    """Mirrors `core.pipeline.Metadata`. The compare view diffs runs on these."""

    model: str
    backend: str
    latency_ms: float
    llm_calls: int
    retrieval_passes: int
    tokens_in: int
    tokens_out: int
    termination_reason: str
    groundedness: float = Field(
        description="Fraction of retrieved sources cited. Compliance proxy, not accuracy."
    )
    cost_estimate_usd: float = Field(description="Estimated USD; 0.0 on the local backend.")
    feedback_votes: int = Field(
        default=0,
        description="Stored votes counted against this run's candidates (Feedback RAG only; "
        "0 everywhere else). Matched rows, before the per-passage cap — some may have moved "
        "nothing. Non-zero means the result depends on history, not the query alone.",
    )


class ModelInfo(BaseModel):
    """One selectable model in GET /api/models."""

    id: str
    display_name: str
    backend: str = Field(description="'ollama' (local, free) or 'anthropic' (paid).")
    is_paid: bool
    is_default: bool
    available: bool = Field(description="False when the backend is not configured.")
    note: str = ""


class UsageResponse(BaseModel):
    """GET /api/usage — running estimate of Anthropic spend."""

    calls: int
    input_tokens: int
    output_tokens: int
    spend_estimate_usd: float
    session_calls: int = Field(description="Paid calls since the server started.")
    session_call_limit: int
    note: str = "Local list-price estimate, not your bill. The Console spend limit is the cap."


# An opaque token from POST /api/documents. Bounded and character-restricted
# because it becomes part of a Chroma collection name: `secrets.token_hex`
# produces exactly this alphabet, so anything else was not issued here.
SessionId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9]+$"),
]


class RunRequest(BaseModel):
    technique: str = Field(description="Technique slug, e.g. 'standard-rag'.")
    query: Query
    model: str | None = Field(
        default=None,
        description="Model id from GET /api/models. Omit to use the configured default.",
    )
    session_id: SessionId | None = Field(
        default=None,
        description="Uploaded corpus to run against, from POST /api/documents. "
        "Omit for the bundled demo corpus.",
    )


class RunResponse(BaseModel):
    """A completed run: the answer plus everything needed to explain how it got there."""

    technique: str
    query: str
    answer: str
    retrieved_chunks: list[Chunk]
    steps: list[Step]
    metadata: Metadata
    draft_id: str | None = Field(
        default=None,
        description="Set when the run paused for human review; pass it to POST /api/run/final.",
    )
    corpus: str = Field(
        default="demo corpus",
        description="Which corpus answered — the demo corpus, or the uploaded one's label. "
        "Present on every run so a reader never has to infer it.",
    )


# Hints end up embedded (and cost nothing locally), but an unbounded one is an
# unbounded string in a request body. Stripped for the same reason `Query` is.
Hint = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]


class FinalizeRequest(BaseModel):
    """The human's half of an Interactive RAG run.

    No `model`: the final answer uses the draft's model, or the run's metadata
    would describe two models under one field.
    """

    draft_id: str
    chunk_ids: list[str] = Field(description="Draft passages to keep. Omitted ones are dropped.")
    hint: Hint = Field(default="", description="Steers one extra retrieval. Not shown to the model.")


class FeedbackRequest(BaseModel):
    """One thumbs up or down, on the passages a Feedback RAG result showed.

    No text field: the server reads each passage's current text out of the index
    and hashes that, so a client can only rate text this server retrieved.
    """

    technique: str = Field(description="Technique slug the passages were shown under.")
    query: Query = Field(description="The question they were retrieved for. Stored, not scored.")
    chunk_ids: list[str] = Field(
        min_length=1,
        # A panel shows top_k passages, so a larger batch is a malformed client
        # rather than a bigger opinion.
        max_length=settings.top_k,
        description="Passages being rated. Duplicates are rejected.",
    )
    rating: Literal[-1, 1] = Field(description="-1 for thumbs down, 1 for thumbs up.")

    @field_validator("chunk_ids")
    @classmethod
    def _no_duplicates(cls, chunk_ids: list[str]) -> list[str]:
        # Votes are append-only by design, so three copies of one id in one
        # request would be three votes from one click — an amplification the
        # reader never asked for.
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_ids contains duplicates; each passage may be rated once.")
        return chunk_ids


class FeedbackResponse(BaseModel):
    """POST /api/feedback. Nothing to return but acknowledgement — the effect is
    visible on the next run of Feedback RAG, in its trace."""

    ok: Literal[True] = True


class ComparisonSide(BaseModel):
    """One half of a comparison: which technique, on which model."""

    technique: str = Field(description="Technique slug.")
    model: str | None = Field(
        default=None, description="Model id. Omit for the configured default."
    )


class CompareRequest(BaseModel):
    """Two (technique × model) sides against one query.

    Both axes vary independently, which is what supports the two interesting
    comparisons: same model + different techniques (does retrieval differ?), and
    same technique + different models (does the model differ?).

    One `session_id` for both sides, not one each. A comparison whose halves read
    different corpora is comparing corpora, not techniques — making that
    impossible to express beats validating it after the fact.
    """

    query: Query
    a: ComparisonSide
    b: ComparisonSide
    session_id: SessionId | None = Field(
        default=None,
        description="Uploaded corpus both sides run against. Omit for the demo corpus.",
    )


class ComparisonDiff(BaseModel):
    """Computed differences between the two runs.

    Server-side rather than in the UI: this is the part a reader is meant to
    learn from, so it is part of the API contract and identical for any client.
    """

    chunk_overlap: int = Field(description="Chunks both sides retrieved.")
    chunk_overlap_pct: float = Field(
        description="Overlap as a percentage of the larger retrieved set. "
        "0 means the two techniques saw entirely different evidence."
    )
    shared_chunk_ids: list[str]
    only_a_chunk_ids: list[str]
    only_b_chunk_ids: list[str]

    latency_delta_ms: float = Field(description="b minus a. Negative means b was faster.")
    llm_calls_delta: int
    steps_delta: int
    tokens_in_delta: int
    tokens_out_delta: int
    cost_delta_usd: float

    same_technique: bool
    same_model: bool


class CompareResponse(BaseModel):
    query: str
    a: RunResponse
    b: RunResponse
    diff: ComparisonDiff


class UploadResponse(BaseModel):
    """POST /api/documents — the corpus that was just built from uploaded files."""

    session_id: str = Field(description="Pass this on /api/run and /api/compare.")
    label: str = Field(description="What to call this corpus in the UI.")
    documents: int
    chunks: int = Field(
        description="Chunks indexed. Worth showing: a corpus of four chunks and a "
        "top_k of four cannot produce a disagreement between techniques."
    )
    expires_in_seconds: int = Field(description="After this, the corpus is swept and the id 404s.")
