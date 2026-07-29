"""Load source documents off disk.

Deliberately thin. The plan lists pdf/md/txt loaders; only md and txt are here
because the sample corpus is Markdown and a PDF loader with no PDF to read is a
dependency and a code path that nothing exercises. Add it when a PDF arrives.
"""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})


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
