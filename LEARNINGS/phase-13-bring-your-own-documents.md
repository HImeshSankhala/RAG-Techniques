# Phase 13 — Bring your own documents

You know Phase 12: the home page compares nine techniques, and every number on it
is either measured against the bundled 43-chunk corpus or labelled as editorial
prose. This phase lets a visitor replace that corpus with their own files.

## The problem, and the problem with solving it

RAG's actual proposition is *your* documents. A demo on a fixed corpus is
something you watch; a demo on your own files is something you use. That is the
upside, and it is real.

The risk is equally real, and it is the reason this phase was optional for twelve
phases. The bundled corpus is curated so the techniques visibly disagree — Phase
12 pinned the shipped presets at 25% top-4 overlap, and `make eval` fails if that
stops being true. An arbitrary uploaded document has no such property. Somebody
uploads a three-page résumé, all five available techniques return the same
passages and the same answer, and the honest conclusion available to them is
"these techniques are identical".

So the design problem was never "how do I index a PDF". It was **what the reader
sees when their corpus produces the same result five times**. Three ways to
handle that:

1. Hide it. Show the result and say nothing. This is the default if you don't
   think about it, and it is what makes the project look pointless.
2. Explain it. Generate a sentence about *why* the techniques agreed.
3. State it. Report the identical result as a measurement, name the corpus, and
   set the expectation before the upload rather than after.

(2) is the tempting one and it is a trap. The compare row has exactly one number
about evidence — `chunk_overlap_pct`, computed at `api/routes/compare.py:96-98`
from retrieved chunk ids. That number cannot distinguish "your corpus is too
small to disagree about" from "your query's terms are in every chunk" from "dense
and keyword retrieval genuinely ranked alike this time". Any sentence naming a
cause is a claim the measurement does not support — which is the exact failure the
whole repo is organised against. So the wording ships (3), and
`DiffSummary.test.ts` asserts the *absence* of a causal clause, because the next
person to read that sentence will want to be more helpful than it is.

## Why a separate collection per upload

The naive version adds the uploaded chunks to the existing Chroma collection with
a `session` field in the metadata and filters on it. It is fewer lines and it is
wrong here, for three reasons that are specific to this codebase:

- Chunk ids are positional — `f"{document.source}#{position}"`, `core/index.py:30`.
  An upload named `raft.md` produces `raft.md#0`, which is already taken. Chroma
  upserts by id, so the collision is silent data loss.
- `evals/retrieval.py` asserts the corpus is exactly 43 chunks from 9 documents.
  A shared collection breaks `make eval` for everyone the moment one visitor
  uploads anything, and every corpus-specific claim in the docs with it.
- A filter is a thing every query must remember to apply. A collection is a thing
  a query cannot forget: the isolation is structural rather than disciplined.

The cost is that "which collection" has to reach `vectorstore`, and the honest
options were a parameter threaded through five modules that have no business
knowing two corpora exist, or a `ContextVar`. This picked the ContextVar
(`core/vectorstore.py`), which is hidden state and is flagged as such in the
module docstring. The mitigating property: the default is the demo corpus, so a
code path that forgets to set it reads the wrong corpus in the *safe* direction.
It is a ContextVar rather than a module global specifically because `/api/compare`
fans two runs into a `ThreadPoolExecutor` — a global would let one request's
corpus leak into another's, while a ContextVar is copied into each worker.

## The algorithm that had to change: BM25's cache

BM25 scores a chunk on term frequency × inverse document frequency — how often a
query term appears here, against how rare it is **in this corpus**. IDF is a
corpus-wide statistic, which is why `core/keyword.py` builds an index over every
chunk rather than asking the store for neighbours.

That index was memoised with `@lru_cache(maxsize=1)` on a zero-argument function.
With one corpus, correct. With two, whichever corpus was queried first serves its
term statistics to the other — and because IDF only *reweights*, the result is a
plausible ranking of the right chunks in the wrong order. No exception, no empty
result, just a worse answer. This is the single most likely silent-wrong-answer
bug the phase could have shipped, and the test for it
(`test_bm25_does_not_score_an_upload_with_demo_corpus_statistics`) queries the
demo corpus, then an upload, then the demo corpus again in one process.

Complexities, for the record:
- Building the index: O(N · L) in chunks and their average token length — every
  chunk is tokenised once. Space O(N · L) for the postings.
- Scoring one query: O(Q · N) with `rank_bm25`'s dense scoring, Q query terms.
- Cache sizing: `maxsize=4`. Unbounded would hold a BM25 index per upload for the
  whole TTL; 1 would re-tokenise the entire corpus on every alternation between
  the demo corpus and an upload. 4 bounds the memory and absorbs the alternation.
- Chunking and embedding an upload: O(total characters), which is what the
  character cap bounds — deliberately checked on *extracted* text, because a 2 MB
  PDF decompresses into far more than 2 MB of it.

## The gate: three techniques that cannot honestly run

Graph, Interactive and Feedback RAG are refused against an uploaded corpus at the
route, before dispatch (`implementations/registry.py`, `api/routes/run.py`).

Graph RAG is why this is a gate rather than a pipeline's own empty state, and it
is worth being precise about, because "it already reports no-graph-yet, so it
degrades gracefully" was the *wrong* answer that nearly shipped. Under an active
upload collection, `core/graph.py` compares the stored graph's fingerprint against
the uploaded corpus, finds a mismatch, and marks a perfectly good graph stale —
and the pipeline then resolves demo chunk ids against the uploaded collection.
The reader gets either "re-run `make graph`" (false: the graph is fine) or a
traversal citing passages their documents do not contain. A pipeline's graceful
degradation is only graceful for the state it was written for.

The other two are simpler: a draft records passages and positional ids but not
which corpus they came from, and an uploaded corpus can expire between the two
halves of an interactive run; votes are permanent and content-hashed, while an
uploaded corpus lasts an hour. In all three cases the registry declares the
sentence and both the 409 and the disabled `<option>` use it, so the refusal and
the explanation cannot drift apart.

## Cost

Nothing in the upload path calls a model. Embeddings are local
(`sentence-transformers`), chunking is string work, and entity extraction — the
one per-chunk LLM cost in the codebase — is exactly what the gate above excludes.

Spend per run is unchanged by the size of the corpus, because `top_k = 4` and
`max_chunk_chars = 1500` bound the prompt before it is built. Those two settings
were precautionary for twelve phases; an upload is the first unbounded input in
the project, so they are now load-bearing. Worst case on Haiku is roughly $0.015
for one Agentic run, and the per-process call cap keeps a session near $0.24 —
though that counter is per uvicorn process and shared between visitors, which is
a distinction that did not matter until this phase made "more than one person
uses this" plausible.

## One failure mode

A scanned PDF is images of text. `pypdf` extracts nothing from it and raises
nothing — `extract_text()` returns `""` for every page, perfectly successfully.
Without a check, that upload "succeeds" with zero chunks, every technique
retrieves nothing, and the reader is left with five empty answers and no
explanation. `core/ingest.py` refuses it by name and says that reading it needs
OCR. The general shape is worth keeping: with untrusted input, the dangerous
result is not the error, it is the success that produced nothing.
