"""Chroma wrapper: add chunks, query for the nearest ones.

The rest of the engine talks to this module, never to Chroma directly, so the
vector store stays swappable. Chroma returns *distances*; everything above this
layer works in *scores* (higher is better), and the conversion happens here.

Phase 13 added a second kind of corpus: a visitor's uploaded documents. Those get
their own collection rather than joining the curated one, because chunk ids are
positional (`source#n`) and would collide, and because `make eval` asserts the
curated corpus is exactly 43 chunks from 9 documents. Which collection a request
reads is held in a ContextVar rather than threaded as a parameter through
retrieval, keyword, graph and every pipeline — five modules that have no business
knowing that more than one corpus exists. That is hidden state, and the price is
that a code path which forgets to set it silently reads the demo corpus; the
mitigation is that the default IS the demo corpus, so forgetting is the safe
direction.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import settings
from core.pipeline import Chunk

COLLECTION_NAME = "rag_lab"

# Every uploaded corpus is named `upload_<token>`. The prefix is what makes the
# sweep possible without a second registry: Chroma already knows which
# collections exist, so asking it is cheaper than keeping a list that can drift
# out of sync with the store it describes (and that a restart would lose while
# the on-disk collections survived).
UPLOAD_PREFIX = "upload_"

_active: ContextVar[str] = ContextVar("rag_lab_collection", default=COLLECTION_NAME)


def active() -> str:
    """The collection this request reads from. The curated corpus unless set."""
    return _active.get()


@contextmanager
def using(name: str) -> Iterator[None]:
    """Read and write `name` for the duration of the block.

    A ContextVar rather than a module global because FastAPI runs sync handlers
    in a threadpool and /api/compare fans out to two more threads: a plain global
    would let one request's corpus leak into another's. ContextVars copy into
    both, and the token reset restores whatever was active before.
    """
    token = _active.set(name)
    try:
        yield
    finally:
        _active.reset(token)


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk on its way into the store, before it has a relevance score."""

    chunk_id: str
    text: str
    source: str


def _collection(
    name: str | None = None, metadata: dict[str, str | int | float] | None = None
) -> chromadb.api.models.Collection.Collection:
    # Cosine, not Chroma's default L2. The embeddings are normalized, so cosine
    # distance measures orientation only — two passages about the same topic score
    # alike whether one is a sentence or a paragraph. L2 would let length dominate.
    return _client().get_or_create_collection(
        name=name or active(),
        metadata={"hnsw:space": "cosine", **(metadata or {})},
    )


def reset_collection() -> None:
    """Drop the active collection so an index run starts clean.

    Without this, re-running `make index` would add a second copy of every chunk
    under the same ids — Chroma upserts by id, but chunk ids are positional, so
    edited documents would leave orphaned chunks from the previous run behind.
    """
    drop(active())


def drop(name: str) -> None:
    """Delete a collection if it exists. Deleting a missing one is not an error."""
    try:
        _client().delete_collection(name)
    except Exception:
        # Nothing indexed yet — the first run has no collection to drop.
        pass


def create(name: str, metadata: dict[str, str | int | float]) -> None:
    """Create an empty collection carrying `metadata`.

    Uploads keep their creation time and label in the collection's own metadata,
    so the TTL sweep needs no side file and survives a server restart.
    """
    _collection(name=name, metadata=metadata)


def uploads() -> list[tuple[str, dict[str, str | int | float]]]:
    """Every uploaded corpus: its collection name and the metadata it was made with."""
    return [
        (c.name, dict(c.metadata or {}))
        for c in _client().list_collections()
        if c.name.startswith(UPLOAD_PREFIX)
    ]


def add_chunks(chunks: list[IndexedChunk], embeddings: list[list[float]]) -> None:
    """Store chunks and their vectors."""
    if not chunks:
        return

    _collection().add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source} for c in chunks],
    )


def query(embedding: list[float], top_k: int) -> list[Chunk]:
    """Return the `top_k` chunks nearest to `embedding`, best first."""
    result = _collection().query(query_embeddings=[embedding], n_results=top_k)

    # Chroma nests results one list per query embedding; we only ever send one.
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        Chunk(
            text=text,
            source=str(metadata.get("source", "unknown")),
            # Cosine distance runs 0 (identical) to 2 (opposite); flip it so the
            # UI can sort and display "higher is better" like every other score.
            score=round(1.0 - distance, 4),
            chunk_id=chunk_id,
        )
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def count() -> int:
    """How many chunks are indexed. Used to fail loudly when the index is empty."""
    return _collection().count()


def all_chunks() -> list[IndexedChunk]:
    """Every stored chunk, without scores.

    BM25 needs the whole corpus to compute term statistics — unlike vector search,
    which asks the index for neighbours, a keyword retriever has to see everything
    to know how rare a term is. Chroma is the store of record for chunk text, so
    the corpus is read back from it rather than kept in a second file that could
    drift out of sync with the index.
    """
    stored = _collection().get()
    return [
        IndexedChunk(
            chunk_id=chunk_id,
            text=text,
            source=str(metadata.get("source", "unknown")),
        )
        for chunk_id, text, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"]
        )
    ]
