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


class RunRequest(BaseModel):
    technique: str = Field(description="Technique slug, e.g. 'standard-rag'.")
    query: str = Field(min_length=1, max_length=2000)


class RunResponse(BaseModel):
    """A completed run: the answer plus everything needed to explain how it got there."""

    technique: str
    query: str
    answer: str
    retrieved_chunks: list[Chunk]
    steps: list[Step]
    metadata: dict = Field(
        description="latency_ms, llm_calls, token counts — varies by technique."
    )
