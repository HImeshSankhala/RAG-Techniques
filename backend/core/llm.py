"""Thin wrapper over the Anthropic Messages API.

Pipelines call `complete()` and get back text plus token counts. Keeping the
provider behind one function means swapping models — or providers — is a change
here, not across eight technique classes.
"""

from dataclasses import dataclass
from functools import lru_cache

import anthropic

from core.config import settings


class MissingAPIKeyError(RuntimeError):
    """Raised when generation is attempted without a configured key."""


@dataclass(frozen=True)
class LLMResponse:
    """Text plus the usage numbers the UI shows in its metadata row."""

    text: str
    input_tokens: int
    output_tokens: int


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    if not settings.anthropic_api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy backend/.env.example to backend/.env "
            "and add your key, then restart the server."
        )
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def complete(system: str, user: str) -> LLMResponse:
    """Send one message and return the text response.

    No `temperature` or `top_p`: the current models reject them. Response shape is
    steered by the prompt instead.
    """
    message = _client().messages.create(
        model=settings.answer_model,
        max_tokens=settings.answer_max_tokens,
        # Grounded answering from supplied context is not deep reasoning. Low effort
        # keeps latency down, which matters because the playground times every run.
        output_config={"effort": settings.answer_effort},
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    # Check before reading content: a refusal returns HTTP 200 with empty or partial
    # content, so indexing into content[0] would raise instead of reporting why.
    if message.stop_reason == "refusal":
        raise RuntimeError(
            "The model declined to answer this query. Try rephrasing it."
        )

    # Content is a list of blocks; thinking blocks appear alongside text ones.
    text = "".join(block.text for block in message.content if block.type == "text")

    return LLMResponse(
        text=text.strip(),
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
