"""Reciprocal Rank Fusion — merging ranked lists from retrievers that disagree.

The problem: two retrievers each return a ranked list, and you need one list.

The obvious approach is to combine their scores, and it is wrong. Cosine
similarity runs roughly 0 to 1 and clusters tightly; BM25 is unbounded and depends
on corpus statistics — a BM25 score of 12 and a cosine score of 0.7 are not
measurements of the same thing. Averaging them lets whichever scale happens to be
larger dominate, which is an artefact of the scoring functions rather than a claim
about relevance. Normalising first (min-max, z-score) only moves the problem: it
makes the numbers comparable in range without making them comparable in meaning,
and it is unstable when one retriever returns a narrow band of scores.

RRF sidesteps all of it by throwing the scores away and using only position.
Rank 1 means "this retriever's best guess" no matter which retriever said it.
"""

from collections import defaultdict

from core.pipeline import Chunk

# Damping constant. The standard value from Cormack et al. (2009).
#
# It flattens the difference between the very top ranks: with k=60, rank 1 scores
# 1/61 and rank 2 scores 1/62 — nearly equal. Without it (k=0) rank 1 would score
# 1.0 and rank 2 only 0.5, letting a single retriever's top pick dominate the
# merged list. Large k means "trust that a document appears at all more than you
# trust exactly where"; that is the right prior when the retrievers are known to
# disagree, which is why we are fusing them.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]], top_k: int, k: int = RRF_K
) -> list[Chunk]:
    """Merge ranked lists into one.

        RRF(d) = sum over lists of  1 / (k + rank(d))

    A chunk ranked highly by both retrievers beats one ranked highly by only one —
    but a chunk found by a single retriever still places well, which is what makes
    fusion able to surface something the other retriever missed entirely. That is
    the whole point: if agreement were required, the merge could only ever return
    the intersection.

    Complexity: O(n) over the total entries to score, then O(n log n) to sort.
    """
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] += 1.0 / (k + rank)
            # Keep the first sighting. The Chunk objects differ only in `score`,
            # which is the incomparable per-retriever value being discarded here;
            # the fused score replaces it below.
            chunks.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

    return [
        Chunk(
            text=chunks[chunk_id].text,
            source=chunks[chunk_id].source,
            # The RRF score, not either retriever's. Small by construction —
            # with two lists the ceiling is 2/(k+1) ≈ 0.033 — and meaningful only
            # relative to other entries in this same merge.
            score=round(score, 6),
            chunk_id=chunk_id,
        )
        for chunk_id, score in ordered[:top_k]
    ]
