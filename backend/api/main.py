"""FastAPI app. Thin layer: it validates input and delegates to the engine.

It must never touch Chroma, embeddings, or the LLM directly — those live behind
pipelines in implementations/. Keeping that boundary is what makes /api/compare
cheap to build later.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import compare, documents, feedback, models, run, techniques
from core import uploads
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Drop uploaded corpora that outlived their TTL while the server was down.

    The only other caller of `sweep()` is the next upload, so without this the
    LAST corpus of a session would sit on disk indefinitely — the one case where
    the UI's "deleted after an hour" would be a lie.
    """
    uploads.sweep()
    yield


app = FastAPI(
    title="RAG Lab API",
    description="Run and compare 9 RAG techniques over the same corpus.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(techniques.router)
app.include_router(run.router)
app.include_router(models.router)
app.include_router(compare.router)
app.include_router(feedback.router)
app.include_router(documents.router)


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe — also the quickest way to tell if the dev server is up."""
    return {"status": "ok"}
