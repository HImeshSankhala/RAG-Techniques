"""POST /api/documents, DELETE /api/documents/{session_id} — bring your own corpus.

The route does three things and delegates the rest: enforce the request-shaped
limits (how many parts), read the bytes, and map engine errors to status codes.
Everything about chunking, embedding, isolation and expiry lives in
`core.uploads`, which is also the only module that names a Chroma collection.

Bytes are read into memory and never written to disk. No filename from this
request is ever opened, joined to a path, or used as one — `core.ingest.safe_source`
reduces it to a display label before it reaches a chunk id.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import SessionId, UploadResponse
from core.config import settings
from core.uploads import UploadRejected, create, delete

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/documents", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Index uploaded .txt/.md/.pdf files into a corpus of their own."""
    # Checked before any part is read: the point of a file-count cap is to bound
    # the reading, so counting after reading would bound nothing.
    if len(files) > settings.upload_max_files:
        raise HTTPException(
            status_code=413,
            detail=f"{len(files)} files uploaded; at most {settings.upload_max_files} at a time.",
        )

    payload = [(file.filename or "document", await file.read()) for file in files]

    try:
        corpus = create(payload)
    except UploadRejected as exc:
        # 413 rather than 422: every one of these is "too big / unreadable input",
        # which is a size-and-content complaint, not a malformed request body.
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    return UploadResponse(**vars(corpus))


@router.delete("/documents/{session_id}", status_code=204)
def delete_documents(session_id: SessionId) -> None:
    """Drop an uploaded corpus now, rather than waiting for its TTL.

    Idempotent: deleting a corpus that has already expired is a 204, because the
    caller asked for it to be gone and it is gone. The id is constrained to the
    `token_urlsafe` alphabet by `SessionId`, so nothing else reaches the store.
    """
    delete(session_id)
