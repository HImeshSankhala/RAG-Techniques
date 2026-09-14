"""The retrieval strategies themselves, separated from the techniques that pick one.

Every technique so far hard-coded its retrieval: Standard RAG is always dense,
Fusion RAG is always dense+BM25+RRF. Auto RAG is the first one that decides at
runtime, and it needs all three paths callable by name — which is what turns
"how do we retrieve" into its own vocabulary rather than a detail inside a
pipeline.

What justifies `hybrid` living here is duplication that actually existed: Fusion
RAG and Auto RAG's hybrid route are the same scatter-gather, the same candidate
multiplier and the same RRF merge. Both now call it; nobody else does.

`dense` is a convenience, not an extraction. Standard RAG and Multi-Pass still
call `vectorstore.query(embeddings.embed_query(q), k)` directly, and they should:
one line is not duplication worth an indirection, and routing it through here
would buy a layer that hides where the embedding happens. `hybrid` needs it, so
it is defined here — that is the whole reason.

Nothing here makes an LLM call or records a step: these are the workers, and the
pipeline above stays responsible for narrating what it chose and why.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from core import embeddings, keyword, vectorstore
from core.fusion import reciprocal_rank_fusion
from core.pipeline import Chunk

# Each retriever returns more than the final top_k so fusion has room to work.
# If both returned exactly top_k, a chunk ranked 5th by one and 1st by the other
# could never enter the merge — the very case fusion exists to catch.
CANDIDATE_MULTIPLIER = 3


def dense(query: str, top_k: int) -> list[Chunk]:
    """Vector similarity only. Strong on paraphrase, weak on exact tokens."""
    return vectorstore.query(embeddings.embed_query(query), top_k)


class HybridResult(NamedTuple):
    """The fused list plus the two inputs that produced it.

    The inputs are returned, not discarded, because how much the retrievers
    disagreed is the interesting number for the trace — full overlap means the
    merge changed nothing and a single retriever would have answered identically.
    """

    fused: list[Chunk]
    dense: list[Chunk]
    sparse: list[Chunk]


def hybrid(query: str, top_k: int) -> HybridResult:
    """Dense and BM25 in parallel, merged by reciprocal rank.

    Both retrievers are I/O- and C-bound (Chroma's HNSW walk, numpy in rank_bm25),
    so threads genuinely overlap here despite the GIL. Scatter-gather is only worth
    the machinery if the branches run at the same time.
    """
    candidates = top_k * CANDIDATE_MULTIPLIER

    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(dense, query, candidates)
        sparse_future = pool.submit(keyword.query, query, candidates)
        dense_hits, sparse_hits = dense_future.result(), sparse_future.result()

    return HybridResult(
        fused=reciprocal_rank_fusion([dense_hits, sparse_hits], top_k),
        dense=dense_hits,
        sparse=sparse_hits,
    )
