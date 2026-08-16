"""Two LLM backends behind one `generate()`.

Local (Ollama) is free and uncapped. Anthropic is paid, Haiku-only, and wrapped in
every guardrail that protects the $5 ceiling: an allowlist enforced in config, a
hard output-token cap, a per-session call cap, and a running spend estimate.

Routing is by model id, and the important property is that it is *closed*: an id
that is neither the configured local model nor a Haiku model raises rather than
falling back to a paid call.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from core.config import HAIKU_MARKER, settings

Backend = Literal["ollama", "anthropic"]


class LLMError(RuntimeError):
    """Any backend failure the API should surface to the user as-is."""


class MissingAPIKeyError(LLMError):
    """Anthropic was requested but no key is configured."""


class BudgetExceededError(LLMError):
    """The per-session paid-call cap was hit."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    backend: Backend


# Paid calls made in this process. Reset on restart — this is a development
# safety valve against a stuck loop, not an accounting record. The cumulative
# record is .usage.json.
_session_calls = 0


def resolve_backend(model: str) -> Backend:
    """Which backend serves `model`. Raises on anything unrecognized.

    Deliberately not a `try local, else paid` fallback: an unknown id must be an
    error, because the failure mode of guessing wrong is spending money.
    """
    if model == settings.ollama_model:
        return "ollama"
    if HAIKU_MARKER in model.lower():
        return "anthropic"
    raise LLMError(
        f"Unknown model {model!r}. Available: {settings.ollama_model} (local), "
        f"{settings.anthropic_model} (paid). See GET /api/models."
    )


def generate(system: str, user: str, model: str | None = None, *, helper: bool = False) -> LLMResponse:
    """Answer a prompt on the requested model, defaulting to the local backend.

    `helper` marks cheap internal calls (routers, critiques in later phases) so
    they get the tighter output cap.
    """
    model = model or settings.default_model

    if resolve_backend(model) == "ollama":
        return _generate_ollama(system, user, model, helper=helper)
    return _generate_anthropic(system, user, model, helper=helper)


# --- Local backend ---------------------------------------------------------


def _generate_ollama(system: str, user: str, model: str, *, helper: bool = False) -> LLMResponse:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        # qwen3 is a reasoning model, and whether reasoning helps depends on the
        # job. Answering from supplied passages is extraction: thinking only slows
        # the trace and leaks a stray "/think" control token into the answer.
        #
        # Helper calls are judgements, and there thinking is load-bearing. Measured
        # on Multi-Pass's critique: asked whether passages about Dynamo and Bigtable
        # cover a question about Spanner, this model answers "COMPLETE" with
        # thinking off — deterministically, 3 times out of 3, and regardless of how
        # the prompt is worded or whether it sits in the system or user turn. With
        # thinking on it correctly reports the gap. A self-critique that cannot
        # critique makes the whole loop a no-op, so the flag follows the call type.
        "think": helper,
        "options": {
            # The load-bearing option — see core/config.py.
            "num_ctx": settings.ollama_num_ctx,
            # Not the Anthropic helper cap: that is a spend guardrail, and thinking
            # tokens come out of this same budget. See core/config.py.
            "num_predict": (
                settings.ollama_helper_num_predict
                if helper
                else settings.anthropic_max_tokens_answer
            ),
        },
    }

    try:
        request = urllib.request.Request(
            f"{settings.ollama_host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise LLMError(
            f"Could not reach Ollama at {settings.ollama_host}. "
            f"Is it running? (`ollama serve`, then `ollama pull {model}`) — {exc}"
        ) from exc

    return LLMResponse(
        text=body["message"]["content"].strip(),
        # Ollama's names for prompt/completion tokens.
        input_tokens=body.get("prompt_eval_count", 0),
        output_tokens=body.get("eval_count", 0),
        model=model,
        backend="ollama",
    )


# --- Anthropic backend (paid) ----------------------------------------------


@lru_cache(maxsize=1)
def _anthropic_client():
    if not settings.anthropic_api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. The local model works without it — "
            "add a key to backend/.env only if you want to run Haiku."
        )
    import anthropic

    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _generate_anthropic(system: str, user: str, model: str, *, helper: bool) -> LLMResponse:
    global _session_calls

    # config already rejects non-Haiku at startup; this catches a model id that
    # arrives on the request itself.
    if HAIKU_MARKER not in model.lower():
        raise LLMError(f"Refusing to call {model!r}: only Haiku models are permitted.")

    if _session_calls >= settings.anthropic_max_session_calls:
        raise BudgetExceededError(
            f"Session cap of {settings.anthropic_max_session_calls} paid calls reached. "
            "Restart the server to reset, or use the local model."
        )

    max_tokens = (
        settings.anthropic_max_tokens_helper if helper else settings.anthropic_max_tokens_answer
    )

    client = _anthropic_client()
    _session_calls += 1

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    # A refusal returns HTTP 200 with empty or partial content, so this must be
    # checked before indexing into content.
    if message.stop_reason == "refusal":
        raise LLMError("The model declined to answer this query. Try rephrasing it.")

    text = "".join(block.text for block in message.content if block.type == "text")

    record_usage(message.usage.input_tokens, message.usage.output_tokens)

    return LLMResponse(
        text=text.strip(),
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        model=model,
        backend="anthropic",
    )


# --- Spend meter -----------------------------------------------------------


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """List-price estimate for one Haiku call."""
    return (
        input_tokens / 1_000_000 * settings.anthropic_input_usd_per_mtok
        + output_tokens / 1_000_000 * settings.anthropic_output_usd_per_mtok
    )


def read_usage() -> dict:
    """Cumulative paid usage. Returns zeros before the first Anthropic call."""
    try:
        return json.loads(settings.usage_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "spend_estimate_usd": 0.0}


def record_usage(input_tokens: int, output_tokens: int) -> dict:
    """Add one paid call to the running total in .usage.json.

    Read-modify-write with no locking: this is a single-process dev server, and
    the file is an observability aid rather than a ledger. The real cap is the
    Console spend limit.
    """
    usage = read_usage()
    usage["calls"] += 1
    usage["input_tokens"] += input_tokens
    usage["output_tokens"] += output_tokens
    usage["spend_estimate_usd"] = round(
        usage["spend_estimate_usd"] + estimate_cost_usd(input_tokens, output_tokens), 6
    )

    settings.usage_file.write_text(json.dumps(usage, indent=2))
    return usage


def session_calls() -> int:
    """Paid calls made since this process started."""
    return _session_calls
