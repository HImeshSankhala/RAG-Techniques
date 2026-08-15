"""BM25 keyword retrieval — the other half of Fusion RAG.

Dense retrieval matches on meaning and blurs exact tokens. BM25 does the opposite:
it scores on term overlap, so a rare literal term in the query is exactly what it
is best at finding. The two have complementary blind spots, which is the entire
premise of fusing them.

BM25 scores a document by, for each query term it contains: how often the term
appears here (saturating, so the tenth occurrence adds little), how rare the term
is across the corpus (IDF — a term in every document carries no signal), and how
long this document is relative to average (so long documents don't win by
containing more of everything).
"""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from core import vectorstore
from core.pipeline import Chunk
from core.vectorstore import IndexedChunk

# Split on anything that isn't a letter or digit. Deliberately crude: no stemming,
# no stopword list. Stemming would map "retrieves" and "retrieval" together but
# would also blunt exactly the precise-token matching BM25 is here to provide.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, tuple[IndexedChunk, ...]]:
    """Build the BM25 index over the whole corpus, once per process.

    Rebuilt only on restart. That is correct today because `make index` is a
    separate step from serving — but it does mean a re-index while the server is
    running leaves this stale. Worth revisiting if indexing ever moves online
    (Phase 13 would force that).
    """
    chunks = tuple(vectorstore.all_chunks())
    if not chunks:
        # Unreachable through `query`, which guards on the count first. Kept as an
        # invariant for any other caller: a BM25 index over nothing is not a usable
        # object, so building one must fail rather than return silently.
        raise RuntimeError("Nothing indexed — run `make index` first.")
    return BM25Okapi([tokenize(c.text) for c in chunks]), chunks


def query(text: str, top_k: int) -> list[Chunk]:
    """Return the `top_k` chunks with the highest BM25 score, best first.

    Scores are raw BM25: unbounded, corpus-dependent, and NOT comparable to the
    cosine scores from vector search. That incomparability is the reason fusion
    merges by rank rather than by score.

    An unbuilt index returns [] rather than raising, matching what dense retrieval
    does. The two retrievers have to agree on this: Fusion runs both and reports
    "nothing is indexed" from the merged result, so one retriever raising while the
    other returns empty turns a setup state into a 500.
    """
    if vectorstore.count() == 0:
        return []

    bm25, chunks = _index()
    scores = bm25.get_scores(tokenize(text))

    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)

    return [
        Chunk(
            text=chunk.text,
            source=chunk.source,
            score=round(float(score), 4),
            chunk_id=chunk.chunk_id,
        )
        for score, chunk in ranked[:top_k]
    ]


def reset() -> None:
    """Drop the cached index. Used by tests that re-index between cases."""
    _index.cache_clear()
