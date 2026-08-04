"""GET /api/models and GET /api/usage — model selection and spend visibility.

These exist together because they answer the same question from two sides: which
models can I run, and what has the paid one cost me so far.
"""

from fastapi import APIRouter

from api.schemas import ModelInfo, UsageResponse
from core import llm
from core.config import settings

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """Selectable models. Local first — it is the default and the free one."""
    return [
        ModelInfo(
            id=settings.ollama_model,
            display_name=f"{settings.ollama_model} (local)",
            backend="ollama",
            is_paid=False,
            is_default=settings.llm_backend == "ollama",
            available=True,
            note=f"Runs on your machine. Free and uncapped. Context {settings.ollama_num_ctx}.",
        ),
        ModelInfo(
            id=settings.anthropic_model,
            display_name=f"{settings.anthropic_model} (hosted)",
            backend="anthropic",
            is_paid=True,
            is_default=settings.llm_backend == "anthropic",
            # Reported rather than hidden: the selector should show why it is
            # unavailable instead of silently omitting the option.
            available=bool(settings.anthropic_api_key),
            note=(
                f"Paid. Output capped at {settings.anthropic_max_tokens_answer} tokens."
                if settings.anthropic_api_key
                else "No ANTHROPIC_API_KEY set — add one to backend/.env to enable."
            ),
        ),
    ]


@router.get("/usage", response_model=UsageResponse)
def get_usage() -> UsageResponse:
    """Running estimate of Anthropic spend, for the UI badge."""
    usage = llm.read_usage()
    return UsageResponse(
        calls=usage["calls"],
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        spend_estimate_usd=usage["spend_estimate_usd"],
        session_calls=llm.session_calls(),
        session_call_limit=settings.anthropic_max_session_calls,
    )
