"""POST /api/run — execute one technique against one query.

The route stays thin: validate, look up the pipeline, call `run`, convert the
dataclasses to Pydantic. All the interesting work is in the engine, which is what
makes /api/compare cheap to add in Phase 5 — it calls the same `run` twice.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import Chunk, RunRequest, RunResponse, Step
from core.llm import MissingAPIKeyError
from implementations.registry import get_pipeline, is_known

router = APIRouter(prefix="/api", tags=["run"])


@router.post("/run", response_model=RunResponse)
def run_technique(request: RunRequest) -> RunResponse:
    pipeline = get_pipeline(request.technique)

    if pipeline is None:
        # Two different mistakes, two different messages: a typo in the slug is not
        # the same as asking for a technique that hasn't been built yet.
        if is_known(request.technique):
            raise HTTPException(
                status_code=409,
                detail=f"'{request.technique}' is documented but not yet runnable.",
            )
        raise HTTPException(
            status_code=404, detail=f"Unknown technique: '{request.technique}'."
        )

    try:
        result = pipeline.run(request.query)
    except MissingAPIKeyError as exc:
        # A missing key is a setup problem, not a bug — 503 with the fix in the body.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RunResponse(
        technique=pipeline.name,
        query=request.query,
        answer=result.answer,
        retrieved_chunks=[Chunk(**vars(c)) for c in result.retrieved_chunks],
        steps=[Step(**vars(s)) for s in result.steps],
        metadata=result.metadata,
    )
