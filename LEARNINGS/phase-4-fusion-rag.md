# Phase 4 — Fusion RAG

Two retrievers with opposite blind spots, merged by rank. The first technique that
changes *how* retrieval happens rather than how many times.

## The problem, measured

Standard RAG inherits dense retrieval's weaknesses wholesale. The clearest cases in this
corpus are rare exact terms, where dense retrieval returns **the wrong document entirely**:

| Query | Answer lives in | Dense top hit | BM25 top hit |
|---|---|---|---|
| `hinted handoff` | `dynamo.md`, `cassandra.md` | `raft.md#0` ✗ | `cassandra.md#4` ✓ |
| `commit wait` | `spanner.md` | `chubby.md#2` ✗ | `spanner.md#4` ✓ |
| `reversed hostnames` | `bigtable.md` | `dynamo.md#3` ✗ | `bigtable.md#0` ✓ |

(Re-measured after the corpus grew from 4 documents to 9. The original table's `Chubby` row
is gone: `chubby.md` now exists, dense ranks it first, and the row had started disproving
its own point. This is the same expiry the section below is about, arriving on schedule.)

Dense is not being stupid here. `hinted handoff` embeds into the region of "distributed
systems failure handling", which genuinely neighbours Raft's discussion of leader failure.
The embedding has the *topic* right. It simply cannot represent "this exact phrase occurs
in this exact chunk", which is the entirety of what the query asked.

## Why rank, not score

The obvious merge is to combine scores, and it is wrong. Cosine similarity runs ~0–1 and
clusters tightly; BM25 is unbounded and corpus-dependent. In the runs above, BM25's top
scores ran 5.7–6.1 while the corresponding cosine scores sat at 0.06–0.21. Averaging those
lets the larger scale win — an artefact of the scoring functions, not a claim about
relevance.

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

Summing reciprocal ranks means **agreement outweighs confidence.** Usually right. Sometimes
exactly wrong.

Measured on `reversed hostnames`:

- BM25 ranks the correct chunk `bigtable.md#0` **#1**; dense does not return it in 12
- `cassandra.md#0` is **#5 in both** lists — mediocre twice
- Fused: `cassandra.md#0` wins, and `bigtable.md#0` falls out of the top 4 altogether

```
cassandra.md#0 = 1/(k+5) + 1/(k+5)     found by both, unimpressively
bigtable.md#0  = 1/(k+1)               found by one, decisively
```

Two fifth places beat one first place whenever `2/(k+5) > 1/(k+1)` — that is, whenever
`k > 3`.

**Correction (re-measured against the 43-chunk index).** I originally wrote that the sweep
confirms this directly: "at `k ≤ 3` the correct chunk is back in the merged top 4, and from
`k = 4` up it is gone." That is off by three, and the off-by-three is the actual lesson.
Sweeping `k` over the real candidate lists (12 per retriever):

```
k = 0…2   bigtable.md#0 in the fused top 4, ranked above cassandra.md#0
          (at k = 0 cassandra.md#0 is not in the top 4 at all)
k = 3…6   bigtable.md#0 still in the top 4, now ranked below cassandra.md#0
k ≥ 7     bigtable.md#0 gone
```

The pairwise flip lands at `k = 3`, not `k = 4`, because `k > 3` is *strict* and `k = 3` is
an exact tie at `1/4` each. RRF defines no tie-break, so the order there is settled by
`sorted()` being stable and the dense list being merged first — an implementation detail of
`core/fusion.py`, not a property of the algorithm. The inequality's own boundary is the one
value of `k` it cannot answer for.

The inequality is arithmetically right and answers the wrong question. `2/(k+5) > 1/(k+1)`
governs the **pairwise order** of two chunks; what the pipeline consumes is **membership in
the top 4**, which is decided by the chunk in *fourth* place, not the one in first. Losing a
rank costs nothing while the window still holds you. The chunk that actually evicts
`bigtable.md#0` is `cassandra.md#3` — dense **#9**, BM25 **#8**, scoring `1/(k+9) + 1/(k+8)`
— which overtakes `1/(k+1)` at `k = 7`.

Worth keeping as a habit: an ordering inequality between two items tells you nothing about a
top-*k* cutoff until you have named the item at position *k*. I derived a threshold and did
not check it against a sweep; the sweep disagreed by three.

That said, `k` is *not* the escape hatch it looks like. `k` is the damping that stops one retriever
unilaterally deciding the merge — the reason the standard value is 60. Setting `k` anywhere in
the rescue band (`≤ 6`) re-creates the failure RRF's damping exists to prevent. The tuning knob
trades one failure mode for its mirror image rather than removing either.

(On the original 4-document corpus this section read differently: `dynamo.md#3` was dense #1
*and* BM25 #8, so it beat `bigtable.md#0` at every `k` by sharing the same first term. The
9-document corpus broke that specific arithmetic while leaving the lesson intact — worth
noting, because the *unconditional* version of the claim is what `test_fusion.py` pins, and
that test uses synthetic ranks precisely so it does not expire with the index.)

**RRF rewards consensus over conviction.** When one retriever is authoritative for a *kind*
of query, unweighted fusion dilutes it with the other's confident wrongness. The fix is
per-query-type retriever weighting, which is where fusion starts to need Auto RAG's router.

A test pins this behaviour so it stays visible rather than being rediscovered as a bug.

## Fusion is not strictly better

Eight queries — four rare exact terms, four paraphrased questions — with the answering
document written down before measuring, so the set is auditable rather than remembered:

| Query | Answer lives in |
|---|---|
| `hinted handoff` | `dynamo.md`, `cassandra.md` |
| `reversed hostnames` | `bigtable.md` |
| `commit wait` | `spanner.md` |
| `vector clocks` | `dynamo.md` |
| `How does a leader keep followers up to date?` | `raft.md` |
| `What happens when a worker machine fails mid-job?` | `mapreduce.md` |
| `How is a large file split up for storage?` | `gfs.md` |
| `How do writes stay available when a replica is down?` | `dynamo.md`, `cassandra.md` |

| | dense | BM25 | fused |
|---|---|---|---|
| Precision@1 | 3/8 | **7/8** | 3/8 |
| Recall@4 | 5/8 | **8/8** | 7/8 |

**Fusion does not improve precision@1 on this set at all.** That surprised me, and the
reason is the section above: RRF reorders by consensus, and where the two retrievers
disagree hardest — the exact-term queries fusion is supposed to fix — consensus lands on a
chunk neither retriever would have put first.

**It does improve recall@4, 5/8 to 7/8.** That is the metric that matters, and I had been
reporting the wrong one. All four chunks go into the prompt; the model never sees the
ranking. Precision@1 is the right measure for a search results page and the wrong measure
for a RAG context window.

Same caveat as before, stated plainly: half these queries were chosen to find divergence, so
the set leans toward BM25's home ground. **Fusion buys robustness across query types, not
peak precision on any one type.** If you know your traffic is all exact-term lookups, ship
BM25 and skip the vector index.

## Why the end-to-end answers barely changed — and then stopped barely changing

On the original 4-document, 18-chunk corpus, Standard and Fusion produced the same answer to
`hinted handoff`. Fusion promoted `dynamo.md#2` from rank 3 to rank 1, but Standard's top-4
already contained it, so the model saw the same evidence either way.

The reason was corpus size. At `top_k=4` over 18 chunks, every retrieval pulls **22% of the
entire corpus**. Ranking has to be badly wrong before the right chunk falls out of a window
that wide.

Re-measured on the 9-document, 43-chunk corpus, `top_k=4` is **9%**, and the same query now
behaves completely differently:

- Dense top-4: `raft.md#0`, `raft.md#1`, `chubby.md#2`, `chubby.md#0` — **no Dynamo at all**
- `dynamo.md#2` is dense **#7**, outside the window
- Fused top-4: `dynamo.md#2`, `chubby.md#0`, `chubby.md#1`, `raft.md#0`

Fusion is no longer reordering evidence Standard already had; it is supplying evidence
Standard never saw. That is the difference between a ranking improvement and a retrieval
improvement, and the only thing that changed is how much of the corpus fits in `top_k`.

The general lesson: **ranking improvements show up in answers only when `top_k` covers a
small fraction of the corpus.** A demo index small enough to be convenient is also small
enough to hide the technique it is demonstrating.

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
