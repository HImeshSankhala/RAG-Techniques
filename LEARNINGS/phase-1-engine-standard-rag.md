# Phase 1 — Engine: core + Standard RAG

Phase 0 built layers with nothing behind them. This phase fills them in: a corpus
on disk becomes vectors in a store, and `POST /api/run` walks a query through
retrieval to an answer.

## The problem RAG solves

A language model knows what was in its training data. It does not know your
documents, anything published after its cutoff, or anything private. Asked about
them, it has two failure modes, and the second is the dangerous one: it says it
doesn't know, or it produces fluent, plausible, wrong text.

Fine-tuning is the obvious fix and usually the wrong one. It costs a training run
per update, needs far more examples than most corpora contain, and bakes facts
into weights where they cannot be inspected, corrected, or attributed. A fact
learned in fine-tuning has no source you can point at.

RAG separates knowledge from the model. Keep documents outside, retrieve the
relevant passages at question time, and put them in the prompt. Updating knowledge
becomes re-running `make index`. Every claim traces to a filename. And the whole
apparatus reduces to one sentence to the model: *answer using only these
passages*.

The cost is that the answer is now only as good as the retrieval. If the right
passage doesn't come back, no amount of model capability recovers it — the model
never sees the fact. **This is why the remaining eight phases exist.** They are all
different answers to "retrieval brought back the wrong thing."

## The pipeline

```
query → embed → nearest-neighbour search → top-k chunks → prompt → answer
```

Three stages, one LLM call, no loops. Standard RAG is the baseline precisely
because it is the least you can do and still be doing RAG.

## Why chunk at all

Documents are not the unit of retrieval; passages are. Two reasons, and the second
matters more than it first appears.

The obvious one is budget — you cannot put a corpus in a prompt.

The subtle one is that **embedding quality degrades with length**. An embedding is
a single fixed-size vector, so a long document's vector is roughly an average of
everything in it. `bigtable.md` covers the data model, tablets, SSTables, and
compaction; averaged into one vector, it sits at the centroid of those topics and
is a sharp match for none of them. A query about memtables should match the
paragraph about memtables, not a blur of the whole file.

### The chunking algorithm

Naive chunking cuts every N characters, which routinely severs sentences. This
project uses **recursive splitting**: try to break on the largest natural boundary
that fits — paragraph, then line, then sentence, then word — and only fall back to
a hard cut when a piece still doesn't fit.

Then the pieces are greedily packed into chunks up to `chunk_size`, with the tail
of each chunk repeated at the head of the next. That overlap exists because a
boundary is a guess: whatever sentence straddles it would otherwise be retrievable
from neither side. With overlap it is intact in one of them.

**Complexity:** O(n) time in the characters of the document — each level of the
separator ladder touches a given character at most once, and the ladder has a
fixed depth of five. O(n) space for the output, roughly n·(1 + overlap/chunk_size)
because overlapping text is stored twice. At `chunk_size=1200, overlap=200` that's
about 17% duplication, which is the price of not losing boundary-straddling
sentences.

## Why embeddings, and why cosine

An embedding maps text to a vector positioned so that semantically similar text
lands nearby. This is what lets "how does Dynamo handle conflicting writes?"
retrieve a passage about vector clocks that shares almost no words with the query.
Keyword search cannot do that — which is also the setup for Phase 4, where it
turns out keyword search can do something embeddings can't.

The model here is `all-MiniLM-L6-v2`: 384 dimensions, runs on CPU in milliseconds,
free. Embeddings are computed for every chunk at index time and every query at run
time, so a hosted embedding API would bill on the hottest path in the project.

Distance is **cosine**, not Chroma's default L2, and the vectors are normalized.
Cosine measures the angle between vectors and ignores magnitude, so a one-sentence
chunk and a paragraph about the same topic score alike. Under L2, length
differences contribute to distance directly, so a long chunk can lose to a short
one purely for being long. Angle is the thing that means "about the same subject."

### Nearest-neighbour search

Finding the closest vectors by brute force is O(n·d) per query — every chunk, every
dimension. At n=18 that is nothing. At a million chunks it is the whole latency
budget.

Chroma indexes with **HNSW** (Hierarchical Navigable Small World): a layered graph
where upper layers are sparse and let a search take large jumps, lower layers are
dense and refine locally. Search descends the layers, greedily walking toward the
query. Roughly **O(log n)** per query, **O(n·d)** space.

The trade-off is in the first letter: HNSW is *approximate*. It can miss a true
nearest neighbour. In exchange it makes the search sublinear, and for retrieval
feeding an LLM that is nearly always right — the fifth-best passage instead of the
fourth changes little, and the alternative is a linear scan.

At n=18, none of this matters. The habit of knowing what your index does before
n=18 becomes n=10⁶ is the point.

## Grounding is a prompt, not a guarantee

The last stage puts the retrieved passages in the prompt and instructs the model to
use only those, to cite the source filename, and to say so when the passages don't
contain the answer.

Worth being clear-eyed: **this is an instruction, not an enforcement mechanism.**
Nothing prevents the model from answering from its own knowledge of Dynamo, which
it certainly has. Grounding is strong in practice and it is not a hard guarantee —
which is exactly why `retrieved_chunks` ships in the response. A reader who can see
the passages can check the answer against them.

Passages go before the question in the prompt: they are the bulk of it and are
identical across every query over this corpus, so keeping them first makes the
prefix cacheable later. It also puts the question adjacent to the answer.

## Failure mode: dense retrieval is weak at exact terms

The clearest example came out of testing this phase. Querying **"What is a
memtable?"** against the corpus returns, as its top hit, `bigtable.md#2` — a chunk
about the *tablet location hierarchy*. The chunk that actually defines a memtable
ranks third, at a modest score.

The query has one load-bearing word. Embedded into 384 dimensions, "memtable"
becomes a point near "Bigtable storage internals" generally, and every chunk of
that document is near that point. The signal that this specific chunk contains the
literal token gets averaged away with everything else in it.

This is the structural weakness of dense retrieval: it is excellent at meaning and
mediocre at exact terms — identifiers, error codes, proper nouns, version numbers.
A keyword index has the opposite profile: it would nail `memtable` and completely
miss the Dynamo/vector-clocks query that dense retrieval handles easily.

Neither is better. **That observation is Phase 4** — Fusion RAG runs both and merges
the rankings, and the reason it needs rank-based fusion rather than score averaging
is that these two scores are not on comparable scales.

## Edge case the tests caught

`test_text_with_no_whitespace_still_splits` failed on first run: chunks came back
110 characters against a limit of 100.

The packer emitted a full chunk, carried the overlap forward as the next chunk's
head, then appended the next piece — without checking whether *overlap + piece*
still fit. Whenever a piece was already near `chunk_size` (which is exactly what
the hard-cut fallback produces), the result overflowed.

Harmless-looking, genuinely not: `chunk_size` is the prompt budget. A chunker that
silently exceeds it is the kind of bug that shows up much later as a truncated
prompt on the one document that happened to contain a long unbroken string.

The fix carries the overlap only when the next piece still fits behind it. When it
doesn't, the piece starts a chunk cleanly and that boundary loses its overlap —
**the size limit is a hard constraint and the overlap is an optimization**, so when
they conflict, the optimization yields.

A second, cosmetic version of the same class of bug: the overlap originally sliced
raw characters, so chunks opened mid-word (`"it of access control"` cut out of
"unit of access control"). Those strings are shown to the reader in the playground
*and* fed to the model as context. Now the overlap snaps to a word boundary.

## What "done" means here, and what is still unverified

Retrieval is verified end-to-end: 18 chunks from 4 documents, queries route to the
correct source document, chunks come back ordered, and the full pipeline —
including step timings and metadata — is exercised by tests with the LLM call
stubbed.

**The generation step is not verified.** It needs an `ANTHROPIC_API_KEY`, which is
not set in this environment. The request shape was checked against the installed
SDK (`output_config.effort` accepts `"low"`), and the missing key degrades to a 503
with instructions rather than a stack trace — but no real answer has been produced.
Phase 1's "done when" is met only once that call runs.
