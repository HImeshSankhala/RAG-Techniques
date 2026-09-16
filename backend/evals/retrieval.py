"""Re-run the retrieval claims this project publishes, against the live index.

Every learn page and every LEARNINGS file here states measured facts about
retrieval: *dense misses this term*, *KRaft occurs in exactly one chunk of 43*,
*the fused top 4 loses this chunk at k = 7*. Each of those is a claim about a
**specific index**, and this project has invalidated one of them in every phase so
far — by adding a document, by moving chunk boundaries — and noticed two phases
late, when a human happened to re-read the prose.

Prose cannot notice that the index moved underneath it. This can.

    cd backend && .venv/bin/python -m evals.retrieval

Retrieval only: local embeddings, Chroma and BM25. No LLM call and no network, so
it runs in seconds and is cheap enough to hang off every commit.

## The two kinds of claim, and why they fail differently

**Corpus-invariant** claims are properties of the retrievers — "BM25 finds a
literal term that dense blurs". Adding a document should not touch them, so a
failure means retrieval regressed or the claim was never true. Prefer these.

**Corpus-specific** claims are examples chosen *for* this 43-chunk index — "dense's
top 4 for the multi-hop question is kafka.md#4, chubby.md#4, kafka.md#1,
kafka.md#0". Some claims are irreducibly of this kind: a multi-hop example is a
statement about a corpus, not about a technique. Those expire, and expiring is not
a bug — it means a page now describes an index that no longer exists, and a human
has to go re-measure. The failure message names the file and the section.

## What is deliberately not pinned

Anything needing an LLM — answer quality, groundedness, which route the classifier
picks, latency. Those are the expensive, non-deterministic claims, and a harness
that made them would be neither fast nor repeatable.

And every floating-point score. Ranks are stable across a library upgrade; cosine
scores are not. Every assertion below is on chunk ids, ranks, counts, or substring
presence, all compared exactly. There are no tolerances here on purpose: a
tolerance on a rank is just a vaguer claim, and the claims in the docs are not
vague.
"""

import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from core import embeddings, graph, keyword, retrieval, vectorstore
from core.config import settings
from core.fusion import reciprocal_rank_fusion
from core.pipeline import Chunk
from core.vectorstore import IndexedChunk

TOP_K = settings.top_k
# What each retriever hands to the merge. Phase 4's k-sweep was measured over
# exactly these lists ("12 per retriever"), so it has to read the same constant
# rather than hardcode 12 — otherwise changing the multiplier silently rewrites
# the claim instead of failing it.
CANDIDATES = TOP_K * retrieval.CANDIDATE_MULTIPLIER


class Skipped(Exception):
    """A check that cannot run here — e.g. the graph was never built."""


@dataclass(frozen=True)
class Claim:
    """One published statement, and the measurement that decides whether it holds.

    `check` returns None when the claim still holds, or a one-line description of
    what was found instead. It may raise `Skipped`.
    """

    claim_id: str
    where: str  # file and section — this is what a failure tells you to go fix
    says: str  # the claim in one line, as the doc states it
    corpus_specific: bool
    check: Callable[[], str | None]


# --- Retrieval helpers -----------------------------------------------------
# Cached because the claims overlap heavily: `reversed hostnames` is measured by
# five of them. Chunk lists are read-only here, and returned as tuples so the
# cache cannot be mutated from under a later claim.


@lru_cache(maxsize=None)
def dense(query: str, k: int = TOP_K) -> tuple[Chunk, ...]:
    return tuple(vectorstore.query(embeddings.embed_query(query), k))


@lru_cache(maxsize=None)
def sparse(query: str, k: int = TOP_K) -> tuple[Chunk, ...]:
    return tuple(keyword.query(query, k))


@lru_cache(maxsize=None)
def fused(query: str, k: int = TOP_K) -> tuple[Chunk, ...]:
    return tuple(retrieval.hybrid(query, k).fused)


@lru_cache(maxsize=1)
def corpus() -> tuple[IndexedChunk, ...]:
    return tuple(vectorstore.all_chunks())


def ids(chunks: Iterable[Chunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def docs(chunks: Iterable[Chunk]) -> set[str]:
    return {c.source for c in chunks}


def containing(*terms: str) -> list[str]:
    """Chunk ids whose text contains every one of `terms`, case-insensitively.

    Substring, not token match: the docs make claims about `understandab*` and
    about phrases like `hinted handoff` that a tokenizer would split.
    """
    lowered = [t.lower() for t in terms]
    return sorted(c.chunk_id for c in corpus() if all(t in c.text.lower() for t in lowered))


def _expect(label: str, found: object, expected: object) -> str | None:
    if found == expected:
        return None
    return f"{label} is {found!r}, documented as {expected!r}"


def _problems(*results: str | None) -> str | None:
    found = [r for r in results if r]
    return "; ".join(found) if found else None


# --- The queries the docs measure on ---------------------------------------

MULTI_HOP = (
    "What replaced ZooKeeper in newer Kafka, and what was that protocol "
    "designed to be easier than?"
)
ISR = "Explain how the ISR mechanism keeps Kafka replicas consistent."

# The three rare exact terms in the fusion-rag.mdx table, with the dense and BM25
# top hits it publishes for each.
EXACT_TERMS: list[tuple[str, str, str]] = [
    ("hinted handoff", "raft.md#0", "cassandra.md#4"),
    ("commit wait", "chubby.md#2", "spanner.md#4"),
    ("reversed hostnames", "dynamo.md#3", "bigtable.md#0"),
]

# The eight-query scorecard, gold documents recorded up front exactly as both
# fusion-rag.mdx and LEARNINGS/phase-4-fusion-rag.md list them.
SCORECARD: list[tuple[str, set[str]]] = [
    ("hinted handoff", {"dynamo.md", "cassandra.md"}),
    ("reversed hostnames", {"bigtable.md"}),
    ("commit wait", {"spanner.md"}),
    ("vector clocks", {"dynamo.md"}),
    ("How does a leader keep followers up to date?", {"raft.md"}),
    ("What happens when a worker machine fails mid-job?", {"mapreduce.md"}),
    ("How is a large file split up for storage?", {"gfs.md"}),
    ("How do writes stay available when a replica is down?", {"dynamo.md", "cassandra.md"}),
]

# Phase 7 measured route divergence over 23 queries and **did not write them
# down** — only their shape ("one full question per document plus six more...")
# plus the eight bare terms. The counts 2/23 and 4/23 are therefore not
# reproducible by anyone, including their author, which is the sharpest argument
# in this file for a harness existing at all: a measurement whose inputs were
# never recorded is a number, not a result.
#
# So this set is recorded. Every query in it is already published somewhere in
# the repo — the eight bare terms and the showcase question from phase 7, the
# four paraphrased questions from the fusion scorecard, the compare-view presets,
# the multi-hop question — and the only claims asserted over it are the two the
# phase 7 file says are robust to the choice of set: nobody agrees three ways,
# and vector never equals keyword.
ROUTE_SET: list[str] = [
    "reversed hostnames",
    "hinted handoff",
    "commit wait",
    "SSTable",
    "KRaft",
    "chunkserver lease",
    "TrueTime",
    "log compaction",
    "How does a leader keep followers up to date?",
    "What happens when a worker machine fails mid-job?",
    "How is a large file split up for storage?",
    "How do writes stay available when a replica is down?",
    ISR,
    MULTI_HOP,
    "What is hinted handoff?",
    "What is commit wait?",
    "What are reversed hostnames used for?",
    "How does Raft elect a leader?",
]


# --- Corpus ----------------------------------------------------------------


def _corpus_shape() -> str | None:
    chunks = corpus()
    return _problems(
        _expect("chunk count", len(chunks), 43),
        _expect("document count", len({c.source for c in chunks}), 9),
    )


# --- Fusion RAG ------------------------------------------------------------


def _exact_terms_invariant() -> str | None:
    """The reason Fusion RAG exists, stated so a corpus change cannot expire it."""
    problems = []
    for term, _, _ in EXACT_TERMS:
        if term in dense(term)[0].text.lower():
            problems.append(f"dense's top hit for {term!r} now contains the phrase")
        if term not in sparse(term)[0].text.lower():
            problems.append(f"BM25's top hit for {term!r} no longer contains the phrase")
    return "; ".join(problems) if problems else None


def _exact_terms_table() -> str | None:
    return _problems(
        *[
            _problems(
                _expect(f"dense top hit for {term!r}", dense(term)[0].chunk_id, want_dense),
                _expect(f"BM25 top hit for {term!r}", sparse(term)[0].chunk_id, want_sparse),
            )
            for term, want_dense, want_sparse in EXACT_TERMS
        ]
    )


def _bm25_misses_paraphrase() -> str | None:
    """The mirror-image failure: vocabulary the question and the answer do not share."""
    query = "How is a large file split up for storage?"
    return _problems(
        _expect("dense top hit", dense(query)[0].chunk_id, "gfs.md#1"),
        _expect("BM25 top hit", sparse(query)[0].chunk_id, "bigtable.md#1"),
    )


def _rrf_consensus_beats_conviction() -> str | None:
    """RRF's documented failure, on the query both Fusion pages use for it."""
    query = "reversed hostnames"
    d, s = ids(dense(query, CANDIDATES)), ids(sparse(query, CANDIDATES))
    top = ids(fused(query))
    return _problems(
        _expect("bigtable.md#0 in dense's candidates", "bigtable.md#0" in d, False),
        _expect("BM25 rank of bigtable.md#0", s.index("bigtable.md#0") + 1, 1),
        _expect("dense rank of cassandra.md#0", d.index("cassandra.md#0") + 1, 5),
        _expect("BM25 rank of cassandra.md#0", s.index("cassandra.md#0") + 1, 5),
        _expect("fused winner", top[0], "cassandra.md#0"),
        _expect("bigtable.md#0 in the fused top 4", "bigtable.md#0" in top, False),
    )


def _rrf_k_sweep() -> str | None:
    """Phase 4's k sweep: both bands, and the tie that separates them.

    An earlier version of this pinned only the *membership* boundary, on the
    reasoning that the pairwise flip sits on an exact tie and is therefore too
    fragile to assert. That was the same mistake the section is about — arguing
    from the arithmetic instead of running the sweep — and it left the harness
    green while both docs printed the flip one step late. The fragile claim is
    exactly where an error hides, so it is pinned now.

    Three things are asserted:

    * membership: bigtable.md#0 is in the fused top 4 for k <= 6, gone from k = 7
    * ordering: it outranks cassandra.md#0 for k = 0..2 and is below it from k = 3
    * the tie at k = 3: both chunks score exactly 1/4, which is *why* the flip is
      one step earlier than `2/(k+5) > 1/(k+1)` predicts. That inequality is
      strict, so it says nothing at its own boundary; what resolves the order
      there is `sorted()` being stable over a dict the dense list populated first.

    The tie is pinned as exact equality of the two fused scores rather than as
    "cassandra wins the tie-break". The equality is a fact about RRF; who wins it
    is a fact about CPython's sort being stable and about the order
    `core/fusion.py` happens to merge its inputs. Pinning the second would make
    this harness enforce an implementation detail the docs explicitly say is not
    a property of the algorithm.
    """
    query = "reversed hostnames"
    lists = [list(dense(query, CANDIDATES)), list(sparse(query, CANDIDATES))]
    d, s = ids(lists[0]), ids(lists[1])

    def ranking(k: int) -> list[str]:
        """The whole merged order, not the top 4 — ordering outlives the window."""
        return ids(reciprocal_rank_fusion(lists, len(d) + len(s), k))

    present = [
        k for k in range(0, 16) if "bigtable.md#0" in ids(reciprocal_rank_fusion(lists, TOP_K, k))
    ]
    above = [
        k
        for k in range(0, 16)
        if ranking(k).index("bigtable.md#0") < ranking(k).index("cassandra.md#0")
    ]

    scores = {c.chunk_id: c.score for c in reciprocal_rank_fusion(lists, len(d) + len(s), 3)}

    return _problems(
        _expect("k values keeping bigtable.md#0 in the fused top 4", present, list(range(0, 7))),
        _expect("k values ranking bigtable.md#0 above cassandra.md#0", above, [0, 1, 2]),
        _expect(
            "the two scores at k = 3",
            (scores["bigtable.md#0"], scores["cassandra.md#0"]),
            (0.25, 0.25),
        ),
        _expect(
            "cassandra.md#0 in the top 4 at k = 0",
            "cassandra.md#0" in ids(reciprocal_rank_fusion(lists, TOP_K, 0)),
            False,
        ),
        _expect("the chunk that evicts it", ids(fused(query))[-2], "cassandra.md#3"),
        _expect("dense rank of cassandra.md#3", d.index("cassandra.md#3") + 1, 9),
        _expect("BM25 rank of cassandra.md#3", s.index("cassandra.md#3") + 1, 8),
    )


def _scorecard(retriever: Callable[[str], Sequence[Chunk]]) -> tuple[int, int]:
    """Precision@1 and recall@4 over the eight recorded queries."""
    precision = sum(1 for q, gold in SCORECARD if retriever(q)[0].source in gold)
    recall = sum(1 for q, gold in SCORECARD if docs(retriever(q)) & gold)
    return precision, recall


def _scorecard_table() -> str | None:
    return _problems(
        _expect("dense (P@1, R@4)", _scorecard(dense), (3, 5)),
        _expect("BM25 (P@1, R@4)", _scorecard(sparse), (7, 8)),
        _expect("fused (P@1, R@4)", _scorecard(fused), (3, 7)),
    )


def _fusion_buys_recall_not_precision() -> str | None:
    """The conclusion the page actually rests on, independent of the exact counts."""
    d_p, d_r = _scorecard(dense)
    f_p, f_r = _scorecard(fused)
    problems = []
    if f_r <= d_r:
        problems.append(f"fusion no longer improves recall@4 over dense ({f_r} vs {d_r})")
    if f_p > d_p:
        problems.append(f"fusion now beats dense on precision@1 too ({f_p} vs {d_p})")
    return "; ".join(problems) if problems else None


# --- Graph RAG -------------------------------------------------------------


def _kraft_is_a_single_chunk() -> str | None:
    kraft = containing("kraft")
    if kraft != ["kafka.md#4"]:
        return f"'KRaft' occurs in {kraft}, documented as exactly one chunk: ['kafka.md#4']"
    return _problems(
        _expect("kafka.md#4 contains 'Paxos'", "kafka.md#4" in containing("paxos"), False),
        _expect(
            "kafka.md#4 contains 'understandab*'",
            "kafka.md#4" in containing("understandab"),
            False,
        ),
    )


def _no_chunk_joins_paxos_and_kafka() -> str | None:
    both = containing("paxos", "kafka")
    if both:
        return f"{both} now contains both 'Paxos' and 'Kafka' — the question has a shortcut"
    return None


def _second_hop_chunks() -> str | None:
    hop_two = containing("paxos", "understandab")
    missing = {"raft.md#1", "chubby.md#3"} - set(hop_two)
    if missing:
        return f"{sorted(missing)} no longer states Raft was an understandable alternative to Paxos"
    mentions_kafka = sorted(set(hop_two) & set(containing("kafka")))
    if mentions_kafka:
        return f"the second-hop chunks {mentions_kafka} now mention Kafka"
    return None


def _dense_top4_on_the_multi_hop_question() -> str | None:
    return _expect(
        "dense top 4",
        ids(dense(MULTI_HOP)),
        ["kafka.md#4", "chubby.md#4", "kafka.md#1", "kafka.md#0"],
    )


def _similarity_cannot_reach_hop_two() -> str | None:
    """Corpus-invariant form of the page's whole argument.

    Not "dense returns these four chunks" but "no single-pass similarity retriever
    returns any chunk holding the second hop". That is what the page claims, and
    it survives a corpus change in a way the chunk ids do not.
    """
    hop_two = set(containing("paxos", "understandab"))
    problems = []
    for label, hits in (("dense", dense(MULTI_HOP)), ("hybrid", fused(MULTI_HOP))):
        reached = hop_two & set(ids(hits))
        if reached:
            problems.append(f"{label} now reaches the second hop via {sorted(reached)}")
    return "; ".join(problems) if problems else None


def _bm25_rank_two_caveat() -> str | None:
    """graph-rag.mdx's own honest caveat. If it stops being true, the caveat goes."""
    return _expect(
        "BM25's rank 2 on the multi-hop question", ids(sparse(MULTI_HOP))[1], "raft.md#1"
    )


def _the_retired_example_stays_retired() -> str | None:
    """Why the page no longer uses the Bigtable/Dynamo question.

    `cassandra.md` states both halves in one sentence, so one passage answers it
    outright. Pinned so the page's explanation of its own history keeps matching
    the corpus — and so that if the corpus ever splits them again, someone is told
    that the retired example became usable.
    """
    both = containing("dynamo", "bigtable")
    if not both:
        return (
            "no chunk states Dynamo and Bigtable together any more — the retired "
            "Bigtable/Dynamo example may be multi-hop again"
        )
    return None


def _traversal_ball() -> str | None:
    """The 81%-of-corpus measurement. Needs the extracted graph, which is optional."""
    stored = graph.load()
    if stored is None:
        raise Skipped("backend/.graph.json absent (build it with `python -m core.graph`)")
    if stored.stale:
        raise Skipped("backend/.graph.json is stale for this index (re-run `python -m core.graph`)")

    walk = graph.traverse(stored.graph, graph.match_entities(stored.graph, MULTI_HOP))
    reached = {
        chunk_id
        for node in walk.depth_of
        for chunk_id in stored.graph.nodes[node].get("chunks", ())
    }
    share = len(reached) / len(corpus())
    if share < 0.6:
        return (
            f"the 2-hop ball now touches {len(reached)} of {len(corpus())} chunks "
            f"({share:.0%}), not the ~81% the page calls 'not a filter'"
        )
    return None


# --- Auto RAG --------------------------------------------------------------


def _routes() -> list[tuple[str, set[str], set[str], set[str]]]:
    return [
        (q, set(ids(dense(q))), set(ids(sparse(q))), set(ids(fused(q)))) for q in ROUTE_SET
    ]


def _routes_diverge() -> str | None:
    """The correction phase 7 made: at chunk-id granularity the routes never agree."""
    rows = _routes()
    all_three = [q for q, v, k, h in rows if v == k == h]
    vector_keyword = [q for q, v, k, _ in rows if v == k]
    return _problems(
        _expect("queries where all three routes agree", all_three, []),
        _expect("queries where vector == keyword", vector_keyword, []),
    )


def _filenames_hide_divergence() -> str | None:
    """The measurement bug itself: filename granularity reports agreement that is not there.

    Pinned as "at least one such query exists", not as a count. The count was
    2/23 over a query set nobody recorded; the *existence* is the finding, and it
    is what makes chunk-id granularity the right ruler.
    """
    hidden = [
        q
        for q in ROUTE_SET
        if docs(dense(q)) == docs(sparse(q)) == docs(fused(q))
        and not (set(ids(dense(q))) == set(ids(sparse(q))) == set(ids(fused(q))))
    ]
    if not hidden:
        return "no query now looks identical by filename while differing by chunk id"
    return None


def _isr_example() -> str | None:
    return _problems(
        _expect(
            "vector", ids(dense(ISR)), ["kafka.md#1", "kafka.md#4", "kafka.md#0", "kafka.md#2"]
        ),
        _expect(
            "keyword", ids(sparse(ISR)), ["kafka.md#3", "kafka.md#2", "kafka.md#1", "kafka.md#0"]
        ),
        _expect(
            "hybrid", ids(fused(ISR)), ["kafka.md#1", "kafka.md#3", "kafka.md#2", "kafka.md#0"]
        ),
    )


def _reversed_hostnames_misroute() -> str | None:
    """The cost of the router's worst deterministic misroute, measured on retrieval alone."""
    query = "reversed hostnames"
    return _problems(
        _expect("keyword's documents", docs(sparse(query)), {"bigtable.md"}),
        _expect("bigtable.md chunks in vector's top 4", "bigtable.md" in docs(dense(query)), False),
        _expect("bigtable.md chunks in hybrid's top 4", "bigtable.md" in docs(fused(query)), False),
    )


def _hybrid_is_not_a_superset() -> str | None:
    """Corpus-invariant: RRF re-ranks and truncates, so hybrid can drop a specialist's #1."""
    query = "hinted handoff"
    best = sparse(query)[0].chunk_id
    if best in ids(fused(query)):
        return f"hybrid now keeps BM25's top hit {best} — the page's counter-example is gone"
    return None


# --- Compare view presets --------------------------------------------------


def _preset_hinted_handoff() -> str | None:
    query = "What is hinted handoff?"
    return _problems(
        _expect("the document dense leads with", dense(query)[0].source, "raft.md"),
        _expect("the document fusion leads with", fused(query)[0].source, "dynamo.md"),
    )


def _preset_commit_wait() -> str | None:
    """Note: `dense leads with chubby.md and misses BM25's #1, spanner.md#2 — one of
    two chunks that say the words`.

    The note this replaced said "only the literal term finds spanner.md", which was
    false: dense returns `spanner.md#4` — a chunk that does contain the phrase — at
    rank 4. The document was never the thing dense missed; a specific chunk was.

    Note the claim is scoped to the top 4, which is what the compare view puts in
    front of a reader. `spanner.md#2` is dense's rank **9** in a wider candidate
    window, so "never returns it" would repeat the original overclaim one window
    further out. "Misses" means "is not in the evidence", and that is asserted as
    written.
    """
    query = "What is commit wait?"
    literal = containing("commit wait")
    return _problems(
        _expect("the document dense leads with", dense(query)[0].source, "chubby.md"),
        _expect("BM25's top hit", sparse(query)[0].chunk_id, "spanner.md#2"),
        _expect("spanner.md#2 in dense's top 4", "spanner.md#2" in ids(dense(query)), False),
        _expect("the chunks that say 'commit wait'", literal, ["spanner.md#2", "spanner.md#4"]),
    )


def _preset_reversed_hostnames() -> str | None:
    query = "What are reversed hostnames used for?"
    return _problems(
        _expect("bigtable.md chunks in dense's top 4", "bigtable.md" in docs(dense(query)), False),
        _expect("bigtable.md chunks in fusion's top 4", "bigtable.md" in docs(fused(query)), True),
    )


def _preset_control() -> str | None:
    """The control preset promises 100% overlap. A control that diverges teaches nothing."""
    query = "How does Raft elect a leader?"
    d, f = ids(dense(query)), ids(fused(query))
    return _problems(
        _expect("fusion's top 4", set(f), set(d)),
        _expect("the documents in it", docs(fused(query)), {"raft.md"}),
    )


# --- The claims ------------------------------------------------------------

CLAIMS: list[Claim] = [
    Claim(
        claim_id="corpus.shape",
        where="frontend/content/graph-rag.mdx and LEARNINGS/phase-4, -7, -8 — "
        "'43 chunks', '9 documents'",
        says="the corpus is 43 chunks across 9 documents",
        corpus_specific=True,
        check=_corpus_shape,
    ),
    Claim(
        claim_id="fusion.exact-terms-invariant",
        where="frontend/content/fusion-rag.mdx — 'The problem it solves'",
        says="for a rare literal term, dense's top hit does not contain the phrase and BM25's does",
        corpus_specific=False,
        check=_exact_terms_invariant,
    ),
    Claim(
        claim_id="fusion.exact-terms-table",
        where="frontend/content/fusion-rag.mdx — the three-row table in 'The problem it solves' "
        "(repeated in LEARNINGS/phase-4-fusion-rag.md, 'The problem, measured')",
        says="the published dense and BM25 top hits for hinted handoff / commit wait / "
        "reversed hostnames",
        corpus_specific=True,
        check=_exact_terms_table,
    ),
    Claim(
        claim_id="fusion.bm25-misses-paraphrase",
        where="frontend/content/fusion-rag.mdx — 'The problem it solves', final paragraph",
        says="'How is a large file split up for storage?' gives gfs.md#1 from dense and "
        "bigtable.md#1 from BM25",
        corpus_specific=True,
        check=_bm25_misses_paraphrase,
    ),
    Claim(
        claim_id="fusion.rrf-consensus",
        where="frontend/content/fusion-rag.mdx — 'Where RRF itself fails' "
        "(and LEARNINGS/phase-4-fusion-rag.md, same heading)",
        says="on 'reversed hostnames' BM25 ranks bigtable.md#0 first, dense never returns it, "
        "cassandra.md#0 is 5th in both, and the fused top 4 keeps the mediocre chunk",
        corpus_specific=True,
        check=_rrf_consensus_beats_conviction,
    ),
    Claim(
        claim_id="fusion.rrf-k-sweep",
        where="LEARNINGS/phase-4-fusion-rag.md — 'Where RRF itself fails', the corrected sweep "
        "(published as a code block in frontend/content/fusion-rag.mdx too)",
        says="bigtable.md#0 outranks cassandra.md#0 for k = 0..2 and is below it from k = 3 (an "
        "exact tie at 1/4 each), stays in the fused top 4 for k <= 6, and is gone from k = 7, "
        "evicted by cassandra.md#3 at dense #9 / BM25 #8",
        corpus_specific=True,
        check=_rrf_k_sweep,
    ),
    Claim(
        claim_id="fusion.scorecard",
        where="frontend/content/fusion-rag.mdx — 'Fusion is not strictly better', the 8-query "
        "precision/recall table (repeated in LEARNINGS/phase-4-fusion-rag.md)",
        says="dense 3/8 and 5/8, BM25 7/8 and 8/8, fused 3/8 and 7/8",
        corpus_specific=True,
        check=_scorecard_table,
    ),
    Claim(
        claim_id="fusion.recall-not-precision",
        where="frontend/content/fusion-rag.mdx — 'Fusion is not strictly better', the two "
        "bolded conclusions under the table",
        says="fusion improves recall@4 over dense and does not improve precision@1",
        corpus_specific=False,
        check=_fusion_buys_recall_not_precision,
    ),
    Claim(
        claim_id="graph.kraft-single-chunk",
        where="frontend/content/graph-rag.mdx — 'The problem it solves' "
        "(and LEARNINGS/phase-8-graph-rag.md, 'The problem')",
        says="KRaft occurs in exactly one chunk of 43, kafka.md#4, which holds neither Paxos "
        "nor understandab*",
        corpus_specific=True,
        check=_kraft_is_a_single_chunk,
    ),
    Claim(
        claim_id="graph.no-shortcut-passage",
        where="frontend/content/graph-rag.mdx — 'The problem it solves' "
        "(and LEARNINGS/phase-8-graph-rag.md, 'The problem')",
        says="no chunk in the corpus contains both Paxos and Kafka",
        corpus_specific=True,
        check=_no_chunk_joins_paxos_and_kafka,
    ),
    Claim(
        claim_id="graph.second-hop-chunks",
        where="LEARNINGS/phase-8-graph-rag.md — 'The problem', second bullet",
        says="raft.md#1 and chubby.md#3 hold the second hop and neither mentions Kafka",
        corpus_specific=True,
        check=_second_hop_chunks,
    ),
    Claim(
        claim_id="graph.dense-top4",
        where="frontend/content/graph-rag.mdx — 'The problem it solves', 'Dense retrieval duly "
        "fails it'",
        says="dense's top 4 for the multi-hop question is kafka.md#4, chubby.md#4, kafka.md#1, "
        "kafka.md#0",
        corpus_specific=True,
        check=_dense_top4_on_the_multi_hop_question,
    ),
    Claim(
        claim_id="graph.similarity-misses-hop-two",
        where="frontend/content/graph-rag.mdx — 'The problem it solves', 'Hybrid fusion fails "
        "the same way'",
        says="neither dense nor hybrid retrieves any chunk holding the second hop",
        corpus_specific=False,
        check=_similarity_cannot_reach_hop_two,
    ),
    Claim(
        claim_id="graph.bm25-rank-two-caveat",
        where="frontend/content/graph-rag.mdx — 'The problem it solves', the parenthesised "
        "'One honest caveat'",
        says="BM25 alone lands raft.md#1 at rank 2 on the multi-hop question",
        corpus_specific=True,
        check=_bm25_rank_two_caveat,
    ),
    Claim(
        claim_id="graph.retired-example",
        where="frontend/content/graph-rag.mdx — 'A note on how this example was chosen'",
        says="cassandra.md states the Dynamo and Bigtable halves together, which is why the "
        "old example stopped being multi-hop",
        corpus_specific=True,
        check=_the_retired_example_stays_retired,
    ),
    Claim(
        claim_id="graph.traversal-ball",
        where="LEARNINGS/phase-8-graph-rag.md — 'Why, precisely' "
        "(and frontend/content/graph-rag.mdx, 'Measured on this corpus, it loses')",
        says="the 2-hop ball touches 35 of 43 chunks — 81% of the corpus, so the graph is not "
        "acting as a filter",
        corpus_specific=True,
        check=_traversal_ball,
    ),
    Claim(
        claim_id="auto.routes-diverge",
        where="LEARNINGS/phase-7-auto-rag.md — 'Correction: measuring at the wrong granularity "
        "got this backwards'",
        says="measured by chunk id, no query has all three routes agree and none has "
        "vector == keyword",
        corpus_specific=False,
        check=_routes_diverge,
    ),
    Claim(
        claim_id="auto.filenames-hide-divergence",
        where="LEARNINGS/phase-7-auto-rag.md — 'Correction', the kafka.md worked example",
        says="some queries look identical at filename granularity while returning different "
        "chunks — which is the measurement bug that section corrects",
        corpus_specific=False,
        check=_filenames_hide_divergence,
    ),
    Claim(
        claim_id="auto.isr-example",
        where="LEARNINGS/phase-7-auto-rag.md — 'Correction', the three-line kafka.md code block",
        says="the published vector / keyword / hybrid top 4 for the ISR question",
        corpus_specific=True,
        check=_isr_example,
    ),
    Claim(
        claim_id="auto.reversed-hostnames-misroute",
        where="LEARNINGS/phase-7-auto-rag.md — 'The actual failure mode: the classifier, not "
        "the pattern'",
        says="on 'reversed hostnames' keyword returns 4/4 bigtable.md and neither vector nor "
        "hybrid returns any of it",
        corpus_specific=True,
        check=_reversed_hostnames_misroute,
    ),
    Claim(
        claim_id="auto.hybrid-not-a-superset",
        where="LEARNINGS/phase-7-auto-rag.md — 'Why this is optimal', 'hybrid is the "
        "best-hedged default, not a superset'",
        says="on 'hinted handoff' the fused top 4 drops BM25's own top hit",
        corpus_specific=False,
        check=_hybrid_is_not_a_superset,
    ),
    Claim(
        claim_id="compare.preset-hinted-handoff",
        where="frontend/components/CompareView.tsx — PRESETS[0].note",
        says="dense leads with raft.md and fusion surfaces dynamo.md",
        corpus_specific=True,
        check=_preset_hinted_handoff,
    ),
    Claim(
        claim_id="compare.preset-commit-wait",
        where="frontend/components/CompareView.tsx — PRESETS[1].note",
        says="dense leads with chubby.md and misses BM25's #1, spanner.md#2 — one of two "
        "chunks that say the words",
        corpus_specific=True,
        check=_preset_commit_wait,
    ),
    Claim(
        claim_id="compare.preset-reversed-hostnames",
        where="frontend/components/CompareView.tsx — PRESETS[2].note",
        says="dense misses bigtable.md entirely and fusion pulls it back into the evidence",
        corpus_specific=True,
        check=_preset_reversed_hostnames,
    ),
    Claim(
        claim_id="compare.preset-control",
        where="frontend/components/CompareView.tsx — PRESETS[3].note",
        says="the control query returns the same four raft.md chunks from dense and fusion — "
        "100% overlap",
        corpus_specific=True,
        check=_preset_control,
    ),
]


# --- Running ---------------------------------------------------------------

CORPUS_SPECIFIC_ADVICE = [
    "This is an example measured on one index, not a property of the technique.",
    "Adding or editing a document expires it legitimately — and a claim that was",
    "overstated when written fails here the same way. Either way the prose is what",
    "needs re-measuring, not this harness. Go open the file named above.",
]

INVARIANT_ADVICE = [
    "This claim was supposed to survive a corpus change, so either retrieval",
    "regressed or the claim was never true. Check backend/core/ before touching",
    "the file named above.",
]


@dataclass(frozen=True)
class Outcome:
    claim: Claim
    status: str  # "ok" | "fail" | "skip"
    detail: str


def run_claim(claim: Claim) -> Outcome:
    try:
        found = claim.check()
    except Skipped as reason:
        return Outcome(claim, "skip", str(reason))
    return Outcome(claim, "fail", found) if found else Outcome(claim, "ok", "")


def report(outcome: Outcome) -> str:
    """What someone who just added a document and broke a page needs to read.

    The audience has no idea which page they broke, so the first thing after the
    claim id is the file and section to open. The measured-vs-documented diff
    comes after that, because on its own a diff of chunk ids tells that reader
    nothing they can act on.
    """
    advice = CORPUS_SPECIFIC_ADVICE if outcome.claim.corpus_specific else INVARIANT_ADVICE
    lines = [
        f"FAIL  {outcome.claim.claim_id}",
        f"      claims : {outcome.claim.says}",
        f"      stated : {outcome.claim.where}",
        f"      found  : {outcome.detail}",
        f"      fix    : {advice[0]}",
    ]
    lines += [f"               {line}" for line in advice[1:]]
    return "\n".join(lines)


def main() -> int:
    if vectorstore.count() == 0:
        print("Nothing indexed — run `make index` first.")
        return 1

    print("Retrieval claims, re-measured against the live index.\n")
    started = time.monotonic()
    outcomes: list[Outcome] = []

    for claim in CLAIMS:
        outcome = run_claim(claim)
        outcomes.append(outcome)
        suffix = f" — {outcome.detail}" if outcome.status == "skip" else ""
        print(f"{outcome.status:<4}  {claim.claim_id}{suffix}")

    failures = [o for o in outcomes if o.status == "fail"]
    skipped = [o for o in outcomes if o.status == "skip"]

    for outcome in failures:
        print()
        print(report(outcome))

    elapsed = time.monotonic() - started
    print(
        f"\n{len(CLAIMS)} claims: {len(outcomes) - len(failures) - len(skipped)} ok, "
        f"{len(failures)} failed, {len(skipped)} skipped  ({elapsed:.1f}s)"
    )

    if failures and any(o.claim.claim_id == "corpus.shape" for o in failures):
        print(
            "\nThe corpus itself changed, so every corpus-specific failure above is "
            "probably downstream of that one."
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
