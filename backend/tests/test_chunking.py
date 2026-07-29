"""Chunking is pure text manipulation, so it can be tested exactly."""

import pytest

from core.chunking import split_text


def test_short_text_is_one_chunk() -> None:
    assert split_text("A short sentence.", chunk_size=100, chunk_overlap=20) == [
        "A short sentence."
    ]


def test_empty_text_produces_no_chunks() -> None:
    assert split_text("", chunk_size=100, chunk_overlap=20) == []
    assert split_text("   \n  ", chunk_size=100, chunk_overlap=20) == []


def test_every_chunk_respects_the_size_limit() -> None:
    text = " ".join(f"word{i}" for i in range(2000))
    for chunk in split_text(text, chunk_size=200, chunk_overlap=40):
        assert len(chunk) <= 200


def test_prefers_paragraph_boundaries() -> None:
    """Two paragraphs that each fit should not be cut mid-paragraph."""
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = split_text(text, chunk_size=30, chunk_overlap=5)

    assert any(c.startswith("First paragraph") for c in chunks)
    assert any("Second paragraph" in c for c in chunks)


def test_chunks_do_not_start_mid_word() -> None:
    """The overlap must snap to a word boundary — these strings are shown to users."""
    text = " ".join(f"token{i}" for i in range(400))
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1, "test needs multiple chunks to exercise the overlap"
    for chunk in chunks[1:]:
        assert chunk.split()[0].startswith("token"), f"chunk opens mid-word: {chunk[:30]!r}"


def test_consecutive_chunks_overlap() -> None:
    """Overlap is what keeps a boundary-straddling sentence retrievable."""
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, chunk_overlap=60)

    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words, "adjacent chunks share no text"


def test_text_with_no_whitespace_still_splits() -> None:
    """The terminal fallback: a hard cut when there is no boundary to find."""
    chunks = split_text("x" * 500, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """Otherwise packing cannot make forward progress and would loop forever."""
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=100, chunk_overlap=100)
