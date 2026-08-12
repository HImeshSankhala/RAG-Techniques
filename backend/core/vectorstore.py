"""Chroma wrapper: add chunks, query for the nearest ones.

The rest of the engine talks to this module, never to Chroma directly, so the
vector store stays swappable. Chroma returns *distances*; everything above this
layer works in *scores* (higher is better), and the conversion happens here.
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import settings
from core.pipeline import Chunk

COLLECTION_NAME = "rag_lab"


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk on its way into the store, before it has a relevance score."""

    chunk_id: str
    text: str
    source: str


def _collection() -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    # Cosine, not Chroma's default L2. The embeddings are normalized, so cosine
    # distance measures orientation only — two passages about the same topic score
    # alike whether one is a sentence or a paragraph. L2 would let length dominate.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drop the collection so an index run starts clean.

    Without this, re-running `make index` would add a second copy of every chunk
    under the same ids — Chroma upserts by id, but chunk ids are positional, so
    edited documents would leave orphaned chunks from the previous run behind.
    """
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Nothing indexed yet — the first run has no collection to drop.
        pass


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
