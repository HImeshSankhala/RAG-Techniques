# Phase 7 — Auto RAG

A cheap classifier in front of expensive workers. The first technique here that
picks its own strategy at runtime instead of having one baked in at design time.

## The problem

Phase 4 asked "dense or BM25?" and answered "always both". That is the *safe*
answer, and it is not free: every query pays two retrievals plus a merge, including
the many queries where one retriever alone would have returned the identical list.
Phase 1 made the opposite bet — always dense — and loses whenever the answer hides
behind a literal token an embedding blurs.

Both are the same mistake in different directions: a choice made once, at design
time, for a population of queries that is not uniform. Auto RAG moves the choice to
run time.

    router (one cheap LLM call) -> vector | keyword | hybrid -> answer

This is the router pattern, and it recurs everywhere: a small model in front of
large ones, a rule engine in front of a service call, a cache lookup in front of a
computation. What makes it a *pattern* rather than just a conditional is that the
classifier is a different, much cheaper kind of thing than the work it dispatches.

## Why this is optimal — and the inequality it rests on

The pattern is only worth it while

    cost(router) << cost(always running the most expensive worker)

and it is easy to violate that by accident. The concrete trap in this repo:
`core/llm.py` exposes `helper` (tighter output budget) and `reason` (let the model
think) as two independent flags. Phase 6's critique needs `reason=True` — without
thinking it answers COMPLETE to everything and the loop is a no-op. It is tempting
to reason "the critique is a helper call and it needs thinking, so helper calls
should think." Measured on this machine, on the identical router prompt:

| | latency | output tokens |
|---|---|---|
| `helper=True, reason=False` | **0.07 – 0.23 s** | 2 – 3 |
| `helper=True, reason=True`  | **13.6 / 16.4 / 26.1 s** | 407 |

A ~90–140× difference for the same one-word verdict. The work being dispatched is a
sub-second retrieval. A thinking router would cost twenty times the retrieval it is
choosing between, and Auto RAG would become a strictly slower Fusion RAG — the
technique would demonstrate the opposite of its own lesson. That is why the flags
were split in the commit before this one, and why `test_auto_rag.py` and
`test_llm_backends.py` both assert `reason is False`.

Measured across five end-to-end runs on the local backend, routing was **2.4 %–3.0 %
of total latency** (67–72 ms of a 6.6–7.9 s run; generation is 5.9–7.7 s of it).
That ratio is the whole justification for the technique.

**What routing does NOT buy: accuracy.** Its ceiling is whatever the best single
path would have returned, and `hybrid` already sits at that ceiling on every query
by construction — its result set is a superset of either specialist's. So routing
can only ever *lose* recall relative to always-hybrid, in exchange for spending less.
Auto RAG is a cost optimisation wearing an intelligence costume, and reading it any
other way will make the compare view against Fusion RAG look like a bug.

## The algorithm and its cost

The router is not an algorithm so much as a classifier call plus a tolerant parse.
The parse is where the engineering is: a small local model emits free text, so
`parse_route` scans the reply for the earliest of `vector` / `keyword` / `hybrid`
as a substring, rather than testing the whole reply for equality. That absorbs
`**keyword**`, `Route: keyword.`, and trailing whitespace, and reads
`"vector, not keyword"` the way it is written. O(n) in reply length over a fixed
three-item alphabet; the reply is 2–3 tokens, so this is free.

**Unparseable output falls back to `hybrid`, not to a specialist.** The three routes
are not peers. Hybrid is the superset, so choosing it can only cost the extra
retrieval — never recall. Falling back to a specialist means betting on the exact
question the router just failed to answer, and losing that bet drops the right
chunk entirely. (This is corpus- and deployment-dependent, not a law: retrieval here
is local and free. Behind a billed retrieval API the safe default is worth
re-deriving.) The fallback is also *loud* — `termination_reason` becomes
`routed_hybrid_fallback` and the step detail says `FALLBACK` — because a silent
fallback is indistinguishable in the trace from a confident correct decision, and
those are the two cases most worth telling apart.

Dispatch itself costs what the chosen path costs:

| route | work | complexity |
|---|---|---|
| `vector` | one embedding + HNSW walk | ~O(log N) over N chunks |
| `keyword` | BM25 scoring over the corpus | O(N · q) for q query terms, then O(N log N) to sort |
| `hybrid`  | both in parallel, then RRF | max of the two, plus O(n) + O(n log n) to merge |

RRF is unchanged from Phase 4 (`core/fusion.py`): `score(d) = Σ 1/(k + rank(d))`,
k = 60, merging by rank because scores from a cosine metric and from BM25 are not
measurements of the same thing.

`core/retrieval.py` is new. The rule of three fired: Standard RAG and Multi-Pass
both open with the same dense lookup, and Fusion RAG's scatter-gather is now
duplicated by Auto RAG's hybrid route. Extracting it also renames the idea usefully
— retrieval *strategies* become a vocabulary the router can select from, instead of
a detail buried inside whichever pipeline happens to use it.

## Failure mode: the router decides on wording, but divergence lives in the corpus

The honest result of measuring this: **on well-formed questions, the three routes
mostly return the same chunks, so the router usually has no decision to get wrong.**

Retrieval-only comparison, top-4 sources per route:

```
"How does commit wait give Spanner external consistency?"
  vector : spanner, spanner, spanner, chubby
  keyword: spanner, spanner, spanner, spanner
  hybrid : spanner, spanner, spanner, spanner      -> vector loses 1 chunk of 4

"Explain how the ISR mechanism keeps Kafka replicas consistent."
  vector / keyword / hybrid : all four chunks from kafka.md   -> no decision to make

"commit wait"            <- the bare term
  vector : chubby, kafka, spanner, raft
  keyword: spanner, spanner, kafka, spanner
  hybrid : kafka, spanner, spanner, spanner        -> vector loses 2 chunks of 4
```

The divergence Phase 4 was built to exploit lives in **bare-term queries**, where
dense retrieval has no surrounding sentence to anchor on and drifts to the wrong
document. Wrap the same rare term in a full question and the extra words give the
embedding enough context to land correctly — so the route stops mattering. But a
full question is exactly what a real user types, and it is what the router sees.

The router picked `vector` for the Spanner sentence, deterministically, six times
out of six. By the compare-view preset that query is a Fusion win, so this is a
genuine misroute — and it cost one chunk out of four. That is the shape of the
failure: not dramatic, just a slow leak of recall in exchange for skipping a
retrieval that was free anyway.

Which sharpens the lesson. The router pattern's expected value is

    P(the workers actually differ) × (cost saved by picking the cheap one)

On a local corpus with free retrieval, the right-hand factor is near zero, so
almost any routing error is affordable and almost any routing success is worthless.
Auto RAG earns its keep when the workers are genuinely expensive and genuinely
different — a paid reranker, a slow graph traversal, a large model. Ship the pattern
for what it will be worth there, and read its numbers here as a demonstration
rather than as a win.

## Edge case: the router burns a call before it can see the index is empty

Standard and Fusion discover an unbuilt index for free — they retrieve first, get
nothing, and return "run `make index`" without ever calling a model. Auto RAG routes
first, so on an empty index it has already spent an LLM call before it learns there
was nothing to route to. Guarding with `vectorstore.count()` up front would avoid
that, and it was deliberately not done: it would give this pipeline a different
shape from every other one to optimise a state that only occurs during first-time
setup. The cost is one cheap local call; the metadata reports `llm_calls=1` and
`termination_reason="empty_index"` honestly, and the backend is taken from the
router's actual response rather than derived, because unlike the other techniques a
real call genuinely happened.

## What is asserted, and why those things

- The router is invoked `helper=True, reason=False`. Load-bearing and invisible if
  wrong — a thinking router still *works*, it is just 100× too slow, and nothing
  errors.
- Each route dispatches to the right retriever, enforced by monkeypatching the
  *other* retriever to raise. Asserting on the returned chunks alone would pass if
  both ran.
- Garbage router output falls back to hybrid **and says so** in both the trace and
  `termination_reason`.
- The router counts as an LLM call (`llm_calls == 2`). Reporting one would flatter
  the technique in precisely the view built to compare costs.
