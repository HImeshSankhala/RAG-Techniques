"""Split documents into retrievable chunks.

Chunk size is the quiet lever under every technique in this project. Too large and
a chunk covers several ideas, so its embedding is an average that matches nothing
sharply. Too small and the chunk that matches the query no longer contains the
sentence that answers it. Everything downstream inherits this choice.

The strategy here is *recursive* splitting: prefer to break on the largest natural
boundary that fits, and only fall back to a finer one when a piece is still too
big. Splitting blindly every N characters is one line shorter and routinely cuts
mid-sentence, which is exactly the text a retriever needs intact.
"""

# Coarsest boundary first. The empty string is the terminal fallback: a hard cut,
# used only for text with no whitespace at all (minified data, long identifiers).
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters.

    Consecutive chunks share `chunk_overlap` trailing characters so a sentence
    that straddles a boundary is still retrievable whole from one side.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    return _pack(_split_recursively(text, _SEPARATORS, chunk_size), chunk_size, chunk_overlap)


def _split_recursively(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """Break `text` into pieces that each fit in `chunk_size`, on the coarsest boundary."""
    if len(text) <= chunk_size:
        return [text]

    separator, *finer = separators

    # No boundary left to try: cut at exactly chunk_size.
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces: list[str] = []
    parts = text.split(separator)
    for i, part in enumerate(parts):
        # Put the separator back, so joining chunks later reproduces readable text.
        piece = part + separator if i < len(parts) - 1 else part
        if not piece:
            continue
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursively(piece, tuple(finer), chunk_size))

    return pieces


def _pack(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks, carrying an overlap between them."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            tail = _overlap_tail(current, chunk_overlap)
            # Carry the overlap only if the next piece still fits behind it. A piece
            # that already fills a chunk on its own has to start one cleanly —
            # otherwise overlap + piece exceeds chunk_size, and the size limit is a
            # hard constraint (it is the prompt budget) while overlap is a nicety.
            current = tail if len(tail) + len(piece) <= chunk_size else ""
        current += piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _overlap_tail(text: str, chunk_overlap: int) -> str:
    """The last `chunk_overlap` characters of `text`, trimmed to a word boundary.

    A raw character slice opens the next chunk mid-word ("it of access control"
    from "unit of access control"). These chunks are shown to the reader in the
    playground and fed to the model as context, so both audiences do better with a
    clean first word.
    """
    if not chunk_overlap:
        return ""

    tail = text[-chunk_overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space != -1 else tail
