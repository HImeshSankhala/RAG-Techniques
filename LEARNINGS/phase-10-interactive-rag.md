# Phase 10 — Interactive RAG

The first technique that is not a function of its input. Every pipeline through Phase 9 is
`run(query) -> RAGResult`; this one stops halfway, waits for a person for an unbounded time,
and resumes with new input. Most of this phase is what that does to the architecture.

## The problem

Multi-Pass, Auto and Agentic RAG all have the model guess what the user meant, and they share
one blind spot: a model cannot notice that a fact it never retrieved is missing. A person
looking at four passages often can, in seconds. So:

    POST /api/run        -> Standard RAG's answer + its passages, saved as a draft (draft_id)
    (human unticks off-topic passages, optionally types a hint)
    POST /api/run/final  -> answer again from the kept passages + what the hint retrieves

What the human's input does, precisely:

- **Kept passages come from the stored draft**, not a fresh retrieval, so the answer is built
  from exactly the text the person read.
- **An unticked passage is never re-admitted**, even if the hint's search returns it again.
  Otherwise the checkbox is only a suggestion.
- **The hint steers retrieval only.** The final prompt asks the original question, so draft and
  final differ in evidence and nothing else, and nobody can type a "fact" into the answer.
- **Kept everything, no hint → `no_change`, zero LLM calls, draft returned.** The prompt would be
  byte-identical to the draft's. A second generation could only differ by sampling noise, and the
  UI would present that noise as the human's doing. This is also the path the learn page predicts
  most users take, so it should cost nothing and the trace should say so.

## Why this design, against the naive alternatives

**Server-held draft (SQLite) vs a stateless round-trip.** The stateless version needs no
storage: send the passages to the browser and have it send the kept ones back. But then the
*client* decides what text goes into the prompt, which could be a paid one. Anything can be
pasted in, at any length. Holding the draft server-side means the final prompt only ever
contains passages this server retrieved. The price is a TTL, an "expired" 404, and one table.
The file (`backend/rag_lab.db`, gitignored) is shared with Phase 11's feedback store.

**One nullable field vs a new pipeline interface.** The obvious design is an
`InteractivePipeline` with `draft()` and `resume()` and a `run()` that raises. But `/api/run` and
`/api/compare` hold a `RAGPipeline` and call `run()`, so a `run()` that raises breaks both (the
Liskov substitution principle: a subclass must work wherever its parent does). Instead `run()` *is*
the draft, and `RAGResult` gained `draft_id: str | None`, which the other six techniques leave
`None`. One technique with two steps is not a pattern, and rule of three says don't build an
interface for it.

**The draft composes Standard RAG** (`StandardRAG().run()`), no copy. Without a human, this
technique *is* the baseline. Saying so in code means the two cannot drift.

**Excluded from compare.** A technique that needs a person mid-run has no honest unattended
result. A compare side would show either a paused draft, which looks identical to Standard RAG,
or a stubbed human, which fakes the lesson. `/api/techniques` exposes `needs_human`, compare
returns 409, and the selectors disable it. The flag is derived in the registry with `isinstance`,
not declared on `RAGPipeline`, so the engine contract carries no per-technique special case.

**Metadata spans two requests, so each panel reports only its own leg.** The Final panel's
`llm_calls`, tokens, cost and latency cover the final request only. The Draft panel above it
already shows the draft's. Cumulative numbers were possible (replay the draft call into the
ledger), but then the two panels on screen don't add up. The human's wait is a `Human review` step
with its real duration, and it is excluded from `latency_ms`. The trace bars are relative to the
slowest step, so a 30-second review shrinks every machine step to a sliver. "Human-bounded
latency" becomes visible instead of being hidden in a number.

**Cost.** Two answer calls per round-trip (one for `no_change`), both capped at 512 output tokens.
Context stays at most 4 kept plus 4 hint chunks at 1500 chars each, about 3k tokens, inside
`num_ctx` 8192. Worst case on Haiku is about $0.010 per round-trip. Each re-finalize of the same
draft is one more capped call.

## Algorithms and complexity

- **Hint retrieval over-fetches.** `dense(query + " " + hint, top_k + len(draft))`, then drop the
  draft's chunks and take up to `top_k` new ones. Appending a few words moves a MiniLM embedding
  only a little, so the plain top-k is mostly the draft again. **Measured on the live check:**
  hint "vector clocks reconciliation" on the Dynamo query brought back **all 4 draft chunks**
  in its top 8. A plain top-4 would have found nothing new; the over-fetch found 4. Cost: one
  HNSW query with k+d, roughly O(log N) per lookup plus O(k+d) to filter. Exclusion is a set
  lookup, so O(k) overall.
- **Lazy TTL expiry.** No cleanup job. Each insert runs `DELETE … WHERE created_at < now - TTL`, a
  full scan (O(n), where n is at most an hour of drafts, so no index), and each read also checks
  age, because a row can expire between sweeps. The trade-off: an index or a background task buys
  nothing at this size, and adds a moving part.
- **Draft ids** are `uuid4` (122 random bits): unguessable, but **not** authentication. Anyone
  holding the id can trigger a final call. That's acceptable for a local app, and it's why the id
  is never exposed through compare.
- **One SQLite connection per operation.** FastAPI runs sync handlers on a thread pool, and a
  `sqlite3` connection belongs to its opening thread. Gotcha: `with sqlite3.connect(...)` commits
  on exit but does **not** close. Use `contextlib.closing`.

## Failure mode: chunk ids are positional

`chunk_id` is `source#n`, and `make index` rebuilds the collection. Edit a document, re-index, and
`raft.md#1` can now name text the person never saw. A draft made before the re-index would then
exclude or dedupe the wrong passage, silently. `finalize` catches this on the hint path: a search
hit whose id is in the draft but whose text differs raises `StaleDraftError` → 409 "run the query
again". Without a hint no retrieval happens and the stored texts are used, which matches what the
person read. The general lesson: an id derived from position is only stable for as long as the
list it indexes.

A second, human failure mode is built into the technique: **people click straight through.** All
passages ticked and no hint is Standard RAG with extra friction. `no_change` makes that path free
and honest, but it can't make the technique useful to someone who doesn't review.
