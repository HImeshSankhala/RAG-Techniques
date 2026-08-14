"""POST /api/compare — run two (technique × model) sides against one query.

This is the endpoint the whole engine contract was designed for. Because every
technique returns the same `RAGResult`, comparing two is calling `run` twice and
subtracting — no per-pair adapter code, and adding a technique in Phase 6 makes it
comparable against all the others for free.
"""

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter

from api.routes.run import run_technique
from core import llm
from core.config import settings
from core.llm import LLMError
from api.schemas import (
    CompareRequest,
    CompareResponse,
    ComparisonDiff,
    ComparisonSide,
    RunRequest,
    RunResponse,
)

router = APIRouter(prefix="/api", tags=["compare"])


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    # Reuse the /api/run handler rather than calling pipelines directly, so both
    # endpoints resolve techniques and map errors to status codes identically.
    # A 404/409/503/429 from either side surfaces unchanged.
    if _both_local(request):
        # Sequential on purpose. Two concurrent local generations each want the
        # full model resident — 5.2GB for qwen3:8b — so running them together
        # doubles resident memory and thrashes swap on a 16GB machine. Measured:
        # ~11s each sequentially, versus not completing within 600s in parallel.
        # Ollama serialises same-model requests anyway, so there was never any
        # throughput to win here.
        a, b = _run_side(request.query, request.a), _run_side(request.query, request.b)
    else:
        # At most one side is local; the other is network-bound, so they genuinely
        # overlap and the comparison finishes in max(a, b) instead of a + b.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_run_side, request.query, side)
                for side in (request.a, request.b)
            ]
            # .result() re-raises in this thread, so HTTPException still reaches
            # FastAPI with its original status code.
            a, b = (future.result() for future in futures)

    return CompareResponse(query=request.query, a=a, b=b, diff=_diff(request, a, b))


def _both_local(request: CompareRequest) -> bool:
    """True when both sides would run on the local (Ollama) backend."""
    try:
        return all(
            llm.resolve_backend(side.model or settings.default_model) == "ollama"
            for side in (request.a, request.b)
        )
    except LLMError:
        # An unknown model id — let the per-side run raise the real 400 rather
        # than failing here with a less useful message.
        return False


def _run_side(query: str, side: ComparisonSide) -> RunResponse:
    return run_technique(RunRequest(technique=side.technique, query=query, model=side.model))


def _diff(request: CompareRequest, a: RunResponse, b: RunResponse) -> ComparisonDiff:
    a_ids = [c.chunk_id for c in a.retrieved_chunks]
    b_ids = [c.chunk_id for c in b.retrieved_chunks]
    shared = set(a_ids) & set(b_ids)

    # Denominator is the larger set, so the percentage answers "how much of the
    # evidence did they agree on" rather than being inflated when one side
    # retrieved fewer chunks.
    largest = max(len(a_ids), len(b_ids))

    return ComparisonDiff(
        chunk_overlap=len(shared),
        chunk_overlap_pct=round(len(shared) / largest * 100, 1) if largest else 0.0,
        # Sorted for a stable UI ordering; sets are unordered.
        shared_chunk_ids=sorted(shared),
        only_a_chunk_ids=sorted(set(a_ids) - shared),
        only_b_chunk_ids=sorted(set(b_ids) - shared),
        latency_delta_ms=round(b.metadata.latency_ms - a.metadata.latency_ms, 2),
        llm_calls_delta=b.metadata.llm_calls - a.metadata.llm_calls,
        steps_delta=len(b.steps) - len(a.steps),
        tokens_in_delta=b.metadata.tokens_in - a.metadata.tokens_in,
        tokens_out_delta=b.metadata.tokens_out - a.metadata.tokens_out,
        cost_delta_usd=round(
            b.metadata.cost_estimate_usd - a.metadata.cost_estimate_usd, 6
        ),
        same_technique=request.a.technique == request.b.technique,
        # Compare the models actually used, not what was requested — either side
        # may have been None, meaning "the configured default".
        same_model=a.metadata.model == b.metadata.model,
    )
