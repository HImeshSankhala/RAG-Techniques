"""Pydantic schemas — the typed contract between the API and the frontend.

These are the source of truth. `frontend/lib/api.ts` mirrors them by hand; when a
schema changes here, that file changes too or the frontend is lying about the API.
"""

from pydantic import BaseModel, Field


class Technique(BaseModel):
    """One entry in GET /api/techniques."""

    name: str = Field(description="Slug, e.g. 'fusion-rag'. Matches the MDX filename.")
    display_name: str = Field(description="Human-readable name for cards and selectors.")
    tagline: str = Field(description="One-line summary shown on the technique card.")
    implemented: bool = Field(
        description="True when the technique has a runnable pipeline. False = docs only."
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


class RunRequest(BaseModel):
    technique: str = Field(description="Technique slug, e.g. 'standard-rag'.")
    query: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(
        default=None,
        description="Model id from GET /api/models. Omit to use the configured default.",
    )


class RunResponse(BaseModel):
    """A completed run: the answer plus everything needed to explain how it got there."""

    technique: str
    query: str
    answer: str
    retrieved_chunks: list[Chunk]
    steps: list[Step]
    metadata: Metadata
