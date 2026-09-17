"""POST /api/run — execute one technique against one query.

The route stays thin: validate, look up the pipeline, call `run`, convert the
dataclasses to Pydantic. All the interesting work is in the engine, which is what
makes /api/compare cheap to add in Phase 5 — it calls the same `run` twice.

POST /api/run/final is the second half of a run that paused for a human
(Interactive RAG). It calls `interactive_rag.finalize` directly rather than going
through the registry: the registry maps a slug to `run()`, and resuming a draft
is not a slug-dispatched operation. The draft store stays behind that function —
this layer never touches SQLite.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from api.schemas import Chunk, FinalizeRequest, Metadata, RunRequest, RunResponse, Step
from core.llm import BudgetExceededError, LLMError, MissingAPIKeyError
from core.pipeline import RAGResult
from implementations.interactive_rag import (
    DraftNotFoundError,
    InvalidSelectionError,
    StaleDraftError,
    finalize,
)
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

    with _llm_errors():
        result = pipeline.run(request.query, model=request.model)

    return _response(pipeline.name, request.query, result)


@router.post("/run/final", response_model=RunResponse)
def finalize_draft(request: FinalizeRequest) -> RunResponse:
    try:
        with _llm_errors():
            query, result = finalize(request.draft_id, request.chunk_ids, request.hint)
    except DraftNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidSelectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except StaleDraftError as exc:
        # 409: the draft conflicts with the index as it now is.
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _response("interactive-rag", query, result)


@contextmanager
def _llm_errors() -> Iterator[None]:
    """Map LLM failures to status codes, identically for both handlers."""
    try:
        yield
    except MissingAPIKeyError as exc:
        # Setup problem, not a bug — 503 with the fix in the body.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BudgetExceededError as exc:
        # 429: the request was well-formed, the caller just has to slow down.
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMError as exc:
        # Unknown model, refusal, unreachable Ollama — all caller-actionable.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _response(technique: str, query: str, result: RAGResult) -> RunResponse:
    return RunResponse(
        technique=technique,
        query=query,
        answer=result.answer,
        retrieved_chunks=[Chunk(**vars(c)) for c in result.retrieved_chunks],
        steps=[Step(**vars(s)) for s in result.steps],
        metadata=Metadata(**asdict(result.metadata)),
        draft_id=result.draft_id,
    )
