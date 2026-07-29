"""Pydantic schemas — the typed contract between the API and the frontend.

These are the source of truth. `frontend/lib/api.ts` mirrors them by hand; when a
schema changes here, that file changes too or the frontend is lying about the API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Technique(BaseModel):
    """One entry in GET /api/techniques."""

    name: str = Field(description="Slug, e.g. 'fusion-rag'. Matches the MDX filename.")
    display_name: str = Field(description="Human-readable name for cards and selectors.")
    tagline: str = Field(description="One-line summary shown on the technique card.")
    implemented: bool = Field(
        description="True when the technique has a runnable pipeline. False = docs only."
    )
