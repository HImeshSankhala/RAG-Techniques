"""GET /api/techniques — the catalog the home page renders its cards from."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import Technique
from implementations.registry import list_techniques

router = APIRouter(prefix="/api", tags=["techniques"])


@router.get("/techniques", response_model=list[Technique])
def get_techniques() -> list[Technique]:
    """List every technique, flagging which ones can actually run."""
    return [
        Technique(
            name=info.name,
            display_name=info.display_name,
            tagline=info.tagline,
            implemented=is_implemented,
        )
        for info, is_implemented in list_techniques()
    ]
