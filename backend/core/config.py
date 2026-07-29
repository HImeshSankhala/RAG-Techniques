"""Runtime settings, loaded from backend/.env (see .env.example)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cors_origins: str = "http://localhost:3000"

    # Blank is a valid state: retrieval works without a key, only generation needs one.
    # core/llm.py raises a clear error rather than letting the SDK fail obscurely.
    anthropic_api_key: str = ""
    answer_model: str = "claude-opus-5"

    # Thinking counts against max_tokens on this model, so leave room for both.
    answer_max_tokens: int = 4096

    # Grounded answering from supplied context is not a deep-reasoning task, and the
    # playground shows a latency badge. "low" keeps the trace responsive.
    answer_effort: str = "low"

    # all-MiniLM-L6-v2: 384-dim, ~90MB, runs on CPU in milliseconds. Good enough to
    # show retrieval differences between techniques, which is what this project needs.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ~1200 chars ≈ 300 tokens: large enough to hold a whole idea, small enough that
    # top-k of them fits comfortably in a prompt. Overlap keeps a sentence that
    # straddles a boundary retrievable from both sides.
    chunk_size: int = 1200
    chunk_overlap: int = 200

    top_k: int = 4

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sample_docs_dir(self) -> Path:
        return BACKEND_ROOT / "data" / "sample_docs"

    @property
    def chroma_dir(self) -> Path:
        return BACKEND_ROOT / ".chroma"


settings = Settings()
