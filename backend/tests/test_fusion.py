"""Reciprocal Rank Fusion is pure arithmetic over ranked lists, so it tests exactly."""

from core.fusion import RRF_K, reciprocal_rank_fusion
from core.pipeline import Chunk


def chunk(chunk_id: str, score: float = 0.0) -> Chunk:
    return Chunk(text=f"text of {chunk_id}", source="doc.md", score=score, chunk_id=chunk_id)


def ids(chunks: list[Chunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def test_agreement_wins() -> None:
    """Ranked highly by both retrievers is the strongest possible signal."""
    a = [chunk("both"), chunk("only-a")]
    b = [chunk("both"), chunk("only-b")]

    assert ids(reciprocal_rank_fusion([a, b], 3))[0] == "both"


def test_a_chunk_from_one_retriever_still_surfaces() -> None:
    """If agreement were required, fusion could only return the intersection —
    which would defeat the entire purpose of running two retrievers."""
    a = [chunk("only-a")]
    b = [chunk("only-b")]

    assert set(ids(reciprocal_rank_fusion([a, b], 2))) == {"only-a", "only-b"}


def test_scores_are_ignored_entirely() -> None:
    """The point of RRF: BM25's 12.0 and cosine's 0.7 are not comparable, so only
    position is used. Wildly different scores at the same ranks must fuse the same."""
    modest = [chunk("x", score=0.01), chunk("y", score=0.005)]
    huge = [chunk("x", score=9999.0), chunk("y", score=5000.0)]

    assert ids(reciprocal_rank_fusion([modest, modest], 2)) == ids(
        reciprocal_rank_fusion([huge, huge], 2)
    )


def test_rrf_score_formula() -> None:
    """Rank 1 in both lists scores exactly 2/(k+1)."""
    single = [chunk("a")]
    fused = reciprocal_rank_fusion([single, single], 1)

    assert fused[0].score == round(2.0 / (RRF_K + 1), 6)


def test_found_by_both_beats_found_by_one_at_every_k() -> None:
    """The known limitation, pinned deliberately.

    A chunk ranked #1 by one retriever and absent from the other LOSES to a chunk
    ranked #1 and #8. The extra term is always positive, so no k rescues it.

    This is a real failure mode observed on the query `reversed hostnames`, where
    BM25 alone had the right answer. The test exists so the behaviour stays
    visible rather than being rediscovered as a bug.
    """
    dense = [chunk("both-mediocre"), chunk("filler-1")]
    sparse = [chunk("lonely-but-right")] + [chunk(f"filler-{i}") for i in range(2, 8)]
    sparse.append(chunk("both-mediocre"))  # rank 8 in the sparse list

    for k in (1, 5, 20, 60, 200):
        winner = ids(reciprocal_rank_fusion([dense, sparse], 1, k=k))[0]
        assert winner == "both-mediocre", f"k={k} unexpectedly changed the outcome"


def test_respects_top_k() -> None:
    many = [chunk(f"c{i}") for i in range(20)]
    assert len(reciprocal_rank_fusion([many, many], 4)) == 4


def test_empty_input() -> None:
    assert reciprocal_rank_fusion([[], []], 4) == []
