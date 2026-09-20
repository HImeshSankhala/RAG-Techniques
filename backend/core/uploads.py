"""Uploaded corpora: build one, read from it, expire it.

This is the only module that knows an uploaded corpus exists. The API layer calls
`create`/`corpus`/`delete` and never names a Chroma collection itself, and the
pipelines never learn that more than one corpus is possible — they read whatever
`vectorstore.active()` points at.

Isolation is a separate Chroma collection per upload, not a shared one with a
filter. Two reasons, both load-bearing: chunk ids are positional (`raft.md#0`), so
an upload named like a bundled document would collide with it; and `make eval`
asserts the curated corpus is exactly 43 chunks from 9 documents, which a shared
collection breaks for everyone the moment one visitor uploads anything.
"""

import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from core import embeddings, keyword, vectorstore
from core.config import settings
from core.index import chunks_for
from core.ingest import Document, UnsupportedDocumentError, read_upload, safe_source


class UnknownCorpusError(LookupError):
    """No such uploaded corpus — a wrong id, or one that outlived its TTL."""


class UploadRejected(ValueError):
    """The upload broke a limit. The message names the file and the limit."""


@dataclass(frozen=True)
class Corpus:
    """An uploaded corpus, as the API reports it."""

    session_id: str
    label: str
    documents: int
    chunks: int
    expires_in_seconds: int


def create(files: list[tuple[str, bytes]]) -> Corpus:
    """Chunk, embed and index uploaded files into a collection of their own.

    Blocking rather than queued. Embedding a few hundred chunks locally takes
    seconds, and a job queue for a wait that short is a second state machine —
    with its own polling endpoint, its own failure states and its own UI — bought
    to hide a spinner.
    """
    _check_limits(files)
    sweep()

    documents: list[Document] = []
    rejected: list[str] = []
    for filename, data in files:
        try:
            documents.append(read_upload(filename, data))
        except UnsupportedDocumentError as exc:
            rejected.append(str(exc))

    if not documents:
        # Every per-file reason is reported, not just the first: an upload of five
        # scanned PDFs has five identical causes and one of them is the answer.
        raise UploadRejected(" ".join(rejected) or "Nothing could be read from that upload.")

    # Sorted for the same reason `load_documents` sorts: chunk ids are positional,
    # and multipart parts arrive in whatever order the browser sent them.
    documents.sort(key=lambda d: d.source)
    chunks = chunks_for(documents)

    # `token_hex`, not `token_urlsafe`: the id becomes part of a Chroma collection
    # name, and Chroma rejects a name that ends in `-` or `_` — which urlsafe
    # base64 produces roughly one time in thirty. A bug that shows up in 3% of
    # uploads is worse than one that shows up in all of them, so the id is drawn
    # from an alphabet that cannot express it.
    session_id = secrets.token_hex(12)
    name = f"{vectorstore.UPLOAD_PREFIX}{session_id}"
    label = _label(documents)

    vectorstore.create(name, {"created_at": time.time(), "label": label})
    with vectorstore.using(name):
        vectorstore.add_chunks(chunks, embeddings.embed_texts([c.text for c in chunks]))

    return Corpus(
        session_id=session_id,
        label=label,
        documents=len(documents),
        chunks=len(chunks),
        expires_in_seconds=settings.upload_ttl_seconds,
    )


@contextmanager
def corpus(session_id: str | None) -> Iterator[str]:
    """Read from `session_id`'s corpus inside the block, or the curated one if None.

    Yields the label every run reports, so the reader is never left guessing which
    corpus answered. An expired or unknown id raises rather than falling back to
    the demo corpus: a silent fallback would answer a question about the visitor's
    documents out of somebody else's, which is the worst failure this phase has.
    """
    if session_id is None:
        yield "demo corpus"
        return

    name = f"{vectorstore.UPLOAD_PREFIX}{session_id}"
    stored = dict(_find(name) or {})
    if not stored or _expired(stored):
        # Expired but still on disk: drop it now rather than waiting for a sweep,
        # so the 404 and the storage agree.
        vectorstore.drop(name)
        keyword.reset()
        raise UnknownCorpusError(
            "That uploaded corpus has expired or was reset. Upload the documents again."
        )

    with vectorstore.using(name):
        yield str(stored.get("label", "your documents"))


def delete(session_id: str) -> None:
    """Drop an uploaded corpus now. Deleting one that is already gone is fine."""
    vectorstore.drop(f"{vectorstore.UPLOAD_PREFIX}{session_id}")
    # The BM25 index caches per collection; a deleted corpus must not be reachable
    # through a cache entry that outlives it.
    keyword.reset()


def sweep() -> int:
    """Delete every uploaded corpus past its TTL. Returns how many went."""
    gone = 0
    for name, metadata in vectorstore.uploads():
        if _expired(metadata):
            vectorstore.drop(name)
            gone += 1
    if gone:
        keyword.reset()
    return gone


def _expired(metadata: dict[str, str | int | float]) -> bool:
    # A collection with no timestamp is from an older build or was made by hand;
    # treating it as expired is the conservative reading, and it is exactly the
    # case the sweep exists to clean up.
    created_at = metadata.get("created_at")
    if not isinstance(created_at, (int, float)):
        return True
    return time.time() - created_at > settings.upload_ttl_seconds


def _find(name: str) -> dict[str, str | int | float] | None:
    return next((metadata for stored, metadata in vectorstore.uploads() if stored == name), None)


def _label(documents: list[Document]) -> str:
    """What the UI calls this corpus: the filename, or a count once there are more."""
    if len(documents) == 1:
        return documents[0].source
    return f"{len(documents)} uploaded documents"


def _check_limits(files: list[tuple[str, bytes]]) -> None:
    """Refuse oversized uploads before any parsing happens.

    Checked here, not after extraction: the point of a byte cap is to bound the
    work, so it has to run before the work does.
    """
    if not files:
        raise UploadRejected("No files were uploaded.")

    if len(files) > settings.upload_max_files:
        raise UploadRejected(
            f"{len(files)} files uploaded; at most {settings.upload_max_files} at a time."
        )

    for filename, data in files:
        if len(data) > settings.upload_max_file_bytes:
            # `safe_source`, not the raw filename: this string is echoed straight
            # back to the browser, and a name from a form is not a name we chose.
            raise UploadRejected(
                f"{safe_source(filename)}: {len(data):,} bytes, over the "
                f"{settings.upload_max_file_bytes:,} byte limit for one file."
            )

    total = sum(len(data) for _, data in files)
    if total > settings.upload_max_total_bytes:
        raise UploadRejected(
            f"{total:,} bytes in total, over the "
            f"{settings.upload_max_total_bytes:,} byte limit for one upload."
        )
