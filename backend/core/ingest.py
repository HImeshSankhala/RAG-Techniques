"""Load source documents — off disk for the curated corpus, from bytes for uploads.

The disk path stays as thin as it was. The upload path is the one that reads
*untrusted* input, and the difference shows in the code: every limit is checked
before the work it bounds, the filename is used only as a label, and nothing
here opens or writes a path. (The web layer is what touches disk on this path:
Starlette spools a large multipart part to a temp file of its own before the
route sees it.)
"""

import io
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from core.config import settings

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})

# PDF joins the list only for uploads. Putting it in SUPPORTED_SUFFIXES would make
# `make index` try to read a PDF in the sample corpus as UTF-8 text and crash on a
# file nobody put there.
UPLOAD_SUFFIXES = frozenset({".md", ".txt", ".pdf"})


class UnsupportedDocumentError(ValueError):
    """The upload is not a format this can read, or holds no extractable text."""


@dataclass(frozen=True)
class Document:
    """One source file: its full text plus the filename used to cite it."""

    source: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Read every supported file in `directory`, sorted by name for stable chunk ids.

    Sorted because chunk ids are positional (`<source>#<index>`), and an unstable
    read order would silently reshuffle ids between indexing runs.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"No such directory: {directory}")

    documents = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(source=path.name, text=text))

    return documents


def safe_source(filename: str) -> str:
    """The label a chunk id cites, stripped of anything that could name a path.

    The filename arrives from a browser form and is never opened, joined, or
    written — but it *is* echoed back to the reader and embedded in chunk ids, so
    it gets the basename treatment anyway. `PurePosixPath` first because a POSIX
    server still receives Windows separators, and a `..\\..\\etc` that survives as
    a display string invites the next person to pass it to `open()`.
    """
    base = PurePosixPath(filename.replace("\\", "/")).name.strip()
    cleaned = "".join(c for c in base if c.isalnum() or c in "._- ").lstrip(".")
    return cleaned[:80] or "document"


def read_upload(filename: str, data: bytes) -> Document:
    """One uploaded file as a Document, or raise `UnsupportedDocumentError`.

    Bounded twice, on purpose. The byte cap is checked by the caller before this
    is reached; the character cap below is checked on the *extracted* text,
    because a 2 MB PDF can decompress into far more text than that, and the cap
    that matters is the one on what actually gets embedded.
    """
    source = safe_source(filename)
    suffix = Path(source).suffix.lower()

    if suffix not in UPLOAD_SUFFIXES:
        raise UnsupportedDocumentError(
            f"{source}: only {', '.join(sorted(UPLOAD_SUFFIXES))} files can be read."
        )

    text = _extract_pdf(source, data) if suffix == ".pdf" else _extract_text(source, data)
    text = text.strip()

    if not text:
        raise UnsupportedDocumentError(
            f"{source}: no text could be extracted. A scanned PDF is images of text, "
            "which needs OCR — not something this reads."
        )

    if len(text) > settings.upload_max_text_chars:
        raise UnsupportedDocumentError(
            f"{source}: {len(text):,} characters of text, over the "
            f"{settings.upload_max_text_chars:,} limit."
        )

    return Document(source=source, text=text)


def _extract_text(source: str, data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDocumentError(f"{source}: not valid UTF-8 text.") from exc


def _extract_pdf(source: str, data: bytes) -> str:
    # Imported here rather than at module scope: `make index` loads this module and
    # has no PDF to read, and pypdf is the one dependency only the upload path needs.
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        # Encrypted PDFs raise on page access rather than returning empty text, so
        # the refusal is stated here where it can name the reason.
        if reader.is_encrypted:
            raise UnsupportedDocumentError(f"{source}: password-protected PDFs cannot be read.")
        pages = reader.pages[: settings.upload_max_pdf_pages]
        return "\n\n".join(page.extract_text() or "" for page in pages)
    except PyPdfError as exc:
        raise UnsupportedDocumentError(f"{source}: not a readable PDF ({exc}).") from exc
