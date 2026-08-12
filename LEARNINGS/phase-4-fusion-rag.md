# Phase 4 — Fusion RAG

Two retrievers with opposite blind spots, merged by rank. The first technique that
changes *how* retrieval happens rather than how many times.

## The problem, measured

Standard RAG inherits dense retrieval's weaknesses wholesale. The clearest cases in this
corpus are rare exact terms, where dense retrieval returns **the wrong document entirely**:

| Query | Answer lives in | Dense top hit | BM25 top hit |
|---|---|---|---|
| `hinted handoff` | `dynamo.md` | `raft.md` ✗ | `dynamo.md` ✓ |
| `Chubby` | `bigtable.md` | `mapreduce.md` ✗ | `bigtable.md` ✓ |
| `reversed hostnames` | `bigtable.md` | `dynamo.md` ✗ | `bigtable.md` ✓ |

Dense is not being stupid here. `hinted handoff` embeds into the region of "distributed
systems failure handling", which genuinely neighbours Raft's discussion of leader failure.
The embedding has the *topic* right. It simply cannot represent "this exact phrase occurs
in this exact chunk", which is the entirety of what the query asked.

## Why rank, not score

The obvious merge is to combine scores, and it is wrong. Cosine similarity runs ~0–1 and
clusters tightly; BM25 is unbounded and corpus-dependent. In the runs above, BM25 scores
reached 5.28 while cosine sat at 0.42. Averaging those lets the larger scale win — an
artefact of the scoring functions, not a claim about relevance.

Normalising first only moves the problem. Min-max makes the ranges match without making
the *meanings* match, and it is unstable when one retriever returns a narrow band.

RRF discards the scores and uses position:

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

**Complexity:** O(n) to accumulate, O(n log n) to sort. `k = 60` (Cormack et al., 2009)
damps the top ranks — at k=60, rank 1 scores 1/61 and rank 2 scores 1/62, nearly equal, so
no single retriever's top pick can unilaterally decide the merge.

## What I got wrong, and had to correct

The Phase 3 learn page claimed dense retrieval mis-ranks `What is a memtable?`, based on a
Phase 1 observation. **Re-testing in this phase, it does not** — dense now ranks the
memtable chunk first.

The reason is that Phase 1's chunking fixes (word-boundary overlap, and the packing bug
that let chunks exceed `chunk_size`) changed where chunk boundaries fall. Re-indexing
moved the definition into a chunk that dense retrieval ranks well.

The claim was true when written and false by the time it shipped. I replaced it with the
three measured examples above, which I verified in this phase rather than recalling.

The lesson is not "be careful with docs." It is that **a retrieval example is a fact about
a specific index, not about a technique.** Chunk size, overlap, and boundary rules all
change which retriever wins on a given query. Any claim of the form "dense fails at X"
needs the index configuration attached, or it silently expires.

## Where RRF itself fails

Summing reciprocal ranks means **a document found by both retrievers beats a document found
by only one — at every `k`.** Usually right. Sometimes exactly wrong.

Measured on `reversed hostnames`:

- BM25 ranks the correct chunk `bigtable.md#0` **#1**; dense does not return it in 12
- Dense ranks `dynamo.md#3` **#1**; BM25 has it at **#8**
- Fused: `dynamo.md#3` wins, correct answer loses

```
dynamo.md#3    = 1/(k+1) + 1/(k+8)     found by both
bigtable.md#0  = 1/(k+1)               found by one
```

The second term is always positive. I swept `k` from 1 to 200 — the winner never changes.
This is structural, not a tuning failure.

**RRF rewards consensus over conviction.** When one retriever is authoritative for a *kind*
of query, unweighted fusion dilutes it with the other's confident wrongness. The fix is
per-query-type retriever weighting, which is where fusion starts to need Auto RAG's router.

A test pins this behaviour so it stays visible rather than being rediscovered as a bug.

## Fusion is not strictly better

Precision@1 over eight queries:

| | dense | BM25 | fused |
|---|---|---|---|
| correct top hit | 4/8 | **8/8** | 7/8 |

Fusion nearly doubles dense. It also **loses to BM25 alone** on this set.

That result needs its caveat stated plainly: I chose several of these queries specifically
to find divergence, so the set is biased toward rare exact terms — BM25's home ground. A
set weighted toward paraphrased questions would invert it. The honest summary is that
**fusion buys robustness across query types, not peak precision on any one type.** If you
know your traffic is all exact-term lookups, ship BM25 and skip the vector index.

## Why the end-to-end answers barely changed

Running Standard vs Fusion through `/api/run` on `hinted handoff`, both produced correct
answers. Fusion promoted `dynamo.md#2` from rank 3 to rank 1, but Standard's top-4 already
contained it.

At `top_k=4` over an 18-chunk corpus, every retrieval pulls **22% of the entire corpus**.
Ranking has to be badly wrong before the right chunk falls out of the window entirely.

This is a property of the demo corpus, not of the technique, and it is worth being explicit
about: ranking improvements show up in *answers* when `top_k` covers a small fraction of the
corpus. On a realistic index the same rank-3-to-rank-1 promotion is often the difference
between a grounded answer and a confidently wrong one.

## Scatter-gather

The two retrievers run in a `ThreadPoolExecutor`, and they genuinely overlap: Chroma's HNSW
walk and `rank_bm25`'s numpy scoring both release the GIL. They share no state, which is
what makes the pattern extend — a third retriever is one more entry in the list handed to
`reciprocal_rank_fusion`.

Each retriever returns `top_k * 3` candidates. If both returned exactly `top_k`, a chunk
ranked 5th by one and 1st by the other could never enter the merge — precisely the case
fusion exists to catch.

## Edge case: BM25 needs the whole corpus

Vector search asks an index for neighbours. BM25 must see every document to compute IDF —
how rare a term is *across the corpus* is the core of the score. So `core/keyword.py` reads
all chunks back out of Chroma and builds the index in memory, cached per process.

The consequence: **re-indexing while the server runs leaves BM25 stale.** Correct today,
because `make index` is a separate step from serving. It would break the moment indexing
moves online — which is exactly what the optional Phase 13 (document upload) would do.
