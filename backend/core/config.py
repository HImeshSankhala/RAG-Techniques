"""Runtime settings, loaded from backend/.env (see .env.example).

Two LLM backends live here. Local (Ollama) is the default for everything; Anthropic
is opt-in per request and restricted to Haiku by an allowlist that fails at startup.
That asymmetry is deliberate — see PLAN.md "Cost Guardrails".
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Only Haiku may ever be called. A single Opus request can cost more than a
# thousand Haiku ones, so the guard is a substring check on the model id rather
# than a fixed list — a future `claude-haiku-5` passes, `claude-opus-5` cannot.
HAIKU_MARKER = "haiku"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cors_origins: str = "http://localhost:3000"

    # --- Backend selection -------------------------------------------------
    # Local by default. Anthropic is reached only when a request names a Haiku
    # model explicitly, never by falling through to it.
    llm_backend: str = "ollama"

    # --- Local backend -----------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    # Ollama's default context is 2048 tokens. A RAG prompt (4 chunks + question)
    # exceeds that, and Ollama truncates silently — the model simply never sees
    # some of what was retrieved, and the answer looks plausible. This is a
    # correctness setting, not a tuning knob.
    ollama_num_ctx: int = 8192

    # --- Anthropic backend (opt-in, paid) ----------------------------------
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"

    # Output tokens are the expensive side (5x input on Haiku), so they are capped
    # hard rather than trusted to the prompt.
    anthropic_max_tokens_answer: int = 512
    anthropic_max_tokens_helper: int = 256

    # Stops a stuck loop during development from quietly draining credit.
    anthropic_max_session_calls: int = 50

    # --- Iteration caps ----------------------------------------------------
    # Multi-Pass (Phase 6) loops until its own critique says it is done, and a
    # model asked "what is missing?" can nearly always find something. On the
    # local backend an uncapped loop costs latency; on the paid one it is the
    # fastest way to drain the $5. That makes this a spend guarantee rather than
    # a tuning knob, which is why it lives here beside the other caps instead of
    # inside the pipeline that happens to use it.
    multi_pass_max_passes: int = 3

    # Haiku 4.5 list price, USD per million tokens. Used for the local estimate
    # in .usage.json — the Console spend limit is the real cap.
    anthropic_input_usd_per_mtok: float = 1.00
    anthropic_output_usd_per_mtok: float = 5.00

    # --- Retrieval ---------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Small on purpose: every chunk is prompt tokens, and on the paid backend
    # that is money. 4 is enough to answer from this corpus.
    top_k: int = 4

    # Truncate any single chunk before it reaches a paid call, so one oversized
    # document cannot inflate an Anthropic request.
    max_chunk_chars: int = 1500

    # Output budget for local helper calls. Sized for the ones that also reason
    # (`reason=True`, see core/llm.py): Ollama draws thinking tokens from the same
    # num_predict budget as the reply, so a budget sized for the reply alone gets
    # eaten by the reasoning and the reply comes back EMPTY. Multi-Pass reads an
    # empty critique as "no gaps" and silently stops looping — measured at 233 of
    # 256 tokens, i.e. intermittently.
    #
    # Separate from the Anthropic helper cap on purpose: that one is a spend
    # guardrail on a paid call, this one is free local generation and bounds
    # latency only. Reusing one constant for both is what turned a cost control
    # into a correctness bug.
    ollama_helper_num_predict: int = 1024

    @field_validator("anthropic_model")
    @classmethod
    def _reject_non_haiku(cls, value: str) -> str:
        """Fail at import time rather than at first paid call.

        A misconfigured model is the single most expensive mistake available in
        this project, so it must be impossible to reach a request handler.
        """
        if HAIKU_MARKER not in value.lower():
            raise ValueError(
                f"ANTHROPIC_MODEL must be a Haiku model; got {value!r}. "
                "Opus and Sonnet are not callable from this project — see the "
                "$5 ceiling guardrails in PLAN.md."
            )
        return value

    @field_validator("llm_backend")
    @classmethod
    def _known_backend(cls, value: str) -> str:
        if value not in {"ollama", "anthropic"}:
            raise ValueError(f"LLM_BACKEND must be 'ollama' or 'anthropic'; got {value!r}")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sample_docs_dir(self) -> Path:
        return BACKEND_ROOT / "data" / "sample_docs"

    @property
    def chroma_dir(self) -> Path:
        return BACKEND_ROOT / ".chroma"

    @property
    def usage_file(self) -> Path:
        return BACKEND_ROOT / ".usage.json"

    @property
    def default_model(self) -> str:
        """The model used when a request does not name one."""
        return self.anthropic_model if self.llm_backend == "anthropic" else self.ollama_model


settings = Settings()
