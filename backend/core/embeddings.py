"""Turn text into vectors with a local sentence-transformers model.

Local rather than an embedding API: embeddings are computed for every chunk at
index time and for every query at run time, so a hosted model would bill on the
hottest path in the project. all-MiniLM-L6-v2 runs on CPU in milliseconds and is
free, which matters more here than the last few points of retrieval quality.
"""

import threading

from sentence_transformers import SentenceTransformer

from core.config import settings


_instance: SentenceTransformer | None = None
_load_lock = threading.Lock()


def _model() -> SentenceTransformer:
    """Load the model once per process, safely under concurrency.

    Loading costs seconds and hundreds of MB, so it is cached — but the cache
    alone is not enough. `functools.lru_cache` only guarantees that a *completed*
    result is reused; two threads that miss simultaneously will both run the
    body. For most factories that is merely wasteful. For SentenceTransformer it
    is a crash: torch loads weights lazily onto a `meta` device and materialises
    them on first use, and two concurrent loads race that transition, producing

        NotImplementedError: Cannot copy out of meta tensor; no data!

    This surfaced only in Phase 5, because /api/compare is the first caller to
    run two pipelines in genuine parallel — every earlier phase embedded from a
    single thread.

    Double-checked locking: the fast path after warm-up is one `is None` test,
    and only the first concurrent callers pay for the lock.
    """
    global _instance
    if _instance is None:
        with _load_lock:
            if _instance is None:
                _instance = SentenceTransformer(settings.embedding_model)
    return _instance


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Batching is much faster than looping one at a time."""
    if not texts:
        return []
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
