"""The $5 ceiling is a correctness requirement, so it gets tests.

Each of these asserts that an *accident* is impossible — a wrong model, an
uncapped output, a runaway loop. None of them make a paid call.
"""

import inspect

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from core import llm
from core.config import Settings, settings
from core.ledger import LLMLedger
from core.llm import LLMResponse

client = TestClient(app)


def _response(backend: llm.Backend, tokens_in: int, tokens_out: int) -> LLMResponse:
    return LLMResponse(
        text="", input_tokens=tokens_in, output_tokens=tokens_out, model="m", backend=backend
    )


# --- Allowlist: only Haiku is reachable ------------------------------------


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"])
def test_config_rejects_non_haiku_models(model: str) -> None:
    """Startup validation: an expensive model cannot even be configured."""
    with pytest.raises(ValidationError, match="Haiku"):
        Settings(anthropic_model=model)


def test_config_accepts_haiku() -> None:
    assert Settings(anthropic_model="claude-haiku-4-5").anthropic_model == "claude-haiku-4-5"


@pytest.mark.parametrize("model", ["claude-opus-5", "gpt-4", "llama3.2:latest"])
def test_unknown_or_expensive_model_is_refused_at_the_router(model: str) -> None:
    """Routing is closed: an unrecognized id raises rather than falling back to paid."""
    with pytest.raises(llm.LLMError):
        llm.resolve_backend(model)


def test_local_and_haiku_route_to_their_backends() -> None:
    assert llm.resolve_backend(settings.ollama_model) == "ollama"
    assert llm.resolve_backend("claude-haiku-4-5") == "anthropic"


def test_run_rejects_an_expensive_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard holds at the HTTP boundary, not just in the library."""
    response = client.post(
        "/api/run",
        json={"technique": "standard-rag", "query": "hi", "model": "claude-opus-5"},
    )
    assert response.status_code == 400
    assert "opus" in response.json()["detail"].lower()


# --- Caps -------------------------------------------------------------------


def test_output_caps_are_tight() -> None:
    """Output tokens are 5x input on Haiku, so they carry the hard cap."""
    assert settings.anthropic_max_tokens_answer <= 512
    assert settings.anthropic_max_tokens_helper <= 256


def test_top_k_is_small() -> None:
    """Every retrieved chunk is prompt tokens, and on the paid backend that is money."""
    assert settings.top_k <= 5


def test_multi_pass_iterations_are_capped() -> None:
    """An uncapped self-terminating loop is the #1 way to drain credit.

    PLAN fixes Multi-Pass at <= 3 passes. At 3 that is at most 5 paid calls per
    query; the cap is what makes the worst case a number rather than a hope.
    """
    assert settings.multi_pass_max_passes <= 3


def test_local_is_the_default_backend() -> None:
    """Nothing reaches the paid backend without explicitly asking for it."""
    assert settings.llm_backend == "ollama"
    assert settings.default_model == settings.ollama_model


def test_ollama_context_is_large_enough_for_rag() -> None:
    """Ollama's 2048 default silently truncates RAG prompts — see LEARNINGS."""
    assert settings.ollama_num_ctx >= 8192


# --- Spend meter ------------------------------------------------------------


def test_cost_estimate_uses_haiku_rates() -> None:
    # 1M in + 1M out at $1 / $5.
    assert llm.estimate_cost_usd(1_000_000, 1_000_000) == pytest.approx(6.0)
    assert llm.estimate_cost_usd(0, 0) == 0.0


def test_a_paid_call_cannot_be_reported_as_free() -> None:
    """The ledger costs what it counted, from the response that carried both."""
    ledger = LLMLedger("claude-haiku-4-5")
    ledger.record(_response("anthropic", 1_000_000, 1_000_000))

    assert ledger.backend == "anthropic"
    assert ledger.cost_estimate_usd == pytest.approx(6.0)
    assert ledger.metadata(
        latency_ms=1.0, retrieval_passes=1, termination_reason="single_pass"
    ).cost_estimate_usd == pytest.approx(6.0)


def test_a_later_free_call_does_not_erase_an_earlier_paid_one() -> None:
    """Six pipelines used to read the backend off the LAST response.

    Nothing routes two backends through one run today, which is exactly why this
    is worth pinning: the day something does, the total must not round to free.
    """
    ledger = LLMLedger(settings.ollama_model)
    ledger.record(_response("anthropic", 1_000_000, 1_000_000))
    ledger.record(_response("ollama", 10, 10))

    assert ledger.backend == "anthropic"
    assert ledger.cost_estimate_usd > 0


def test_a_pipeline_has_no_way_to_report_its_own_cost() -> None:
    """Safety by construction, not by every author remembering the ternary.

    `metadata()` takes no backend, token or cost argument, so a pipeline cannot
    pass `cost_estimate_usd=0.0` — there is nowhere to pass it.
    """
    parameters = set(inspect.signature(LLMLedger.metadata).parameters)

    assert not parameters & {"backend", "llm_calls", "tokens_in", "tokens_out"}
    assert "cost_estimate_usd" not in parameters


def test_local_calls_are_exactly_free() -> None:
    ledger = LLMLedger(settings.ollama_model)
    ledger.record(_response("ollama", 5000, 500))

    assert ledger.backend == "ollama"
    assert ledger.cost_estimate_usd == 0.0


def test_usage_endpoint_reports_the_session_cap() -> None:
    body = client.get("/api/usage").json()

    assert body["session_call_limit"] == settings.anthropic_max_session_calls
    assert body["spend_estimate_usd"] >= 0
    assert "not your bill" in body["note"]


def test_session_cap_blocks_further_paid_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck loop hits a wall instead of quietly spending."""
    monkeypatch.setattr(llm, "_session_calls", settings.anthropic_max_session_calls)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-not-a-real-key")

    with pytest.raises(llm.BudgetExceededError):
        llm._generate_anthropic("sys", "user", "claude-haiku-4-5", helper=False)


# --- Model catalogue --------------------------------------------------------


def test_models_endpoint_lists_both_backends() -> None:
    body = client.get("/api/models").json()
    by_backend = {m["backend"]: m for m in body}

    assert by_backend["ollama"]["is_paid"] is False
    assert by_backend["anthropic"]["is_paid"] is True
    assert by_backend["ollama"]["is_default"] is True


def test_paid_model_reports_availability_rather_than_hiding() -> None:
    """The selector should explain why an option is unusable, not omit it."""
    anthropic = next(m for m in client.get("/api/models").json() if m["backend"] == "anthropic")

    assert "haiku" in anthropic["id"].lower()
    if not anthropic["available"]:
        assert "ANTHROPIC_API_KEY" in anthropic["note"]
