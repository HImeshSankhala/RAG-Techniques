"""POST /api/feedback — store one thumbs up or down on retrieved passages.

Thin, like every other route: it validates the slug and hands the vote to the
engine. The vote store is SQLite behind `feedback_rag`, and this layer never
opens it — the same boundary `run.py` keeps around the draft store.

Only Feedback RAG accepts votes. A vote cast on another technique's panel would
change a technique the reader was not looking at, which is a surprise, not a
feature — so a known-but-wrong slug is a 409, matching how /api/run answers
"known, but not this way".
"""

from fastapi import APIRouter, HTTPException

from api.schemas import FeedbackRequest, FeedbackResponse
from implementations.feedback_rag import FeedbackRAG, UnknownChunkError, record_feedback
from implementations.registry import is_known

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    if not is_known(request.technique):
        raise HTTPException(
            status_code=404, detail=f"Unknown technique: '{request.technique}'."
        )
    if request.technique != FeedbackRAG.name:
        raise HTTPException(
            status_code=409,
            detail=f"'{request.technique}' does not use stored feedback. Only "
            f"'{FeedbackRAG.name}' reads these votes.",
        )

    try:
        record_feedback(
            request.technique, request.query, request.chunk_ids, request.rating
        )
    except UnknownChunkError as exc:
        # 422: well-formed request, but it names passages this index does not have.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FeedbackResponse()
