"""FastAPI app. Thin layer: it validates input and delegates to the engine.

It must never touch Chroma, embeddings, or the LLM directly — those live behind
pipelines in implementations/. Keeping that boundary is what makes /api/compare
cheap to build later.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import models, run, techniques
from core.config import settings

app = FastAPI(
    title="RAG Lab API",
    description="Run and compare 9 RAG techniques over the same corpus.",
    version="0.1.0",
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


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe — also the quickest way to tell if the dev server is up."""
    return {"status": "ok"}
