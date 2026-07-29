"""Turn text into vectors with a local sentence-transformers model.

Local rather than an embedding API: embeddings are computed for every chunk at
index time and for every query at run time, so a hosted model would bill on the
hottest path in the project. all-MiniLM-L6-v2 runs on CPU in milliseconds and is
free, which matters more here than the last few points of retrieval quality.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from core.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the model once per process.

    Loading costs seconds and hundreds of MB. Without this cache the first
    /api/run of every request would pay it — and with eight more techniques
    coming, each would load its own copy.
    """
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Batching is much faster than looping one at a time."""
    if not texts:
        return []
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
