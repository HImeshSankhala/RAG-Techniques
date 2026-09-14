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
| `helper=True, reason=True`  | **11.1 – 34.2 s** | up to the full 1024 budget |

Between **55× and 190×** for the same one-word verdict, reproduced twice on separate
days (0.07–0.23 s and 0.16–0.20 s for the fast path). The work being dispatched is a
sub-second retrieval. A thinking router would cost twenty times the retrieval it is
choosing between, and Auto RAG would become a strictly slower Fusion RAG — the
technique would demonstrate the opposite of its own lesson. That is why the flags
were split in the commit before this one, and why `test_auto_rag.py` and
`test_llm_backends.py` both assert `reason is False`.

**A thinking router is not merely slow — it can return nothing at all.** One
`reason=True` run spent its entire 1024-token budget inside the thinking block and
came back with an **empty** reply, which `parse_route` cannot parse, so the run took
tens of seconds to arrive at the hybrid fallback. This is the same failure already
documented for the Multi-Pass critique in `core/config.py`: Ollama draws thinking
tokens and reply tokens from one `num_predict` budget, so "let it think" and "cap the
output" are in direct competition. Slowness is the symptom you notice; silent
truncation of the actual answer is the one that bites.

Measured across end-to-end runs on the local backend, routing was **2.7 %–5.1 % of
total latency** — tens of milliseconds against multi-second generation. That ratio is
the whole justification for the technique.

**What routing does NOT buy: accuracy.** Its ceiling is whatever the best single path
would have returned. It is tempting to go one step further and say hybrid already sits
at that ceiling by construction, because its result set is a superset of either
specialist's — **that is false, and the corpus falsifies it immediately.** RRF does not
union the two lists; it re-ranks them *and then truncates to `top_k`*. Measured on
`reversed hostnames`:

```
keyword: bigtable.md#0, #1, #2, #3        <- 4/4 gold, the only document with the phrase
vector : dynamo.md#3, raft.md#2, raft.md#3, chubby.md#2
hybrid : cassandra.md#0, chubby.md#2, cassandra.md#3, dynamo.md#3   <- gold absent
```

Hybrid returns *worse* evidence than the specialist it contains, because
`cassandra.md#0` placed 5th in both lists and RRF's consensus bonus outranks BM25's
single decisive first place (Phase 4's "Where RRF itself fails", same query). The right
framing is that **hybrid is the best-hedged default, not a superset** — it is the route
least likely to be badly wrong across query types, and it is beatable on any given one.

Auto RAG is still a cost optimisation rather than an accuracy improvement. But the
reason is not "hybrid already wins"; it is that a router can only ever pick among paths
that already exist.

## The algorithm and its cost

The router is not an algorithm so much as a classifier call plus a tolerant parse.
The parse is where the engineering is: a small local model emits free text, so
`parse_route` scans the reply for the earliest of `vector` / `keyword` / `hybrid`
as a substring, rather than testing the whole reply for equality. That absorbs
`**keyword**`, `Route: keyword.`, and trailing whitespace, and reads
`"vector, not keyword"` the way it is written. O(n) in reply length over a fixed
three-item alphabet; the reply is 2–3 tokens, so this is free.

**Unparseable output falls back to `hybrid`, not to a specialist.** The three routes
are not peers — but the reason is weaker than "hybrid is the superset", which is not
true (see above). The real argument is variance: falling back to a specialist bets the
whole answer on the exact question the router just failed to answer, and half the time
that bet is on the retriever with the wrong profile for the query. Hybrid is the route
with the smallest worst case, not the route that dominates. It still loses sometimes —
`reversed hostnames` is fallback-into-a-miss in one query — and choosing it is a
hedge, not a guarantee. (This is corpus- and deployment-dependent either way: retrieval
here is local and free. Behind a billed retrieval API the safe default is worth
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

## Correction: measuring at the wrong granularity got this backwards

This section previously concluded: *"on well-formed questions, the three routes
mostly return the same chunks, so the router usually has no decision to get wrong."*

**That is wrong, and the error was in the measurement, not the reasoning.** I compared
**source filenames**. Four chunks all from `kafka.md` is not the same evidence as four
other chunks all from `kafka.md`, and the model is shown chunks, not filenames.

Re-measured at **chunk-id** granularity — set equality over the top-4, 23 queries
(15 full questions + 8 bare terms), against the current 43-chunk / 9-document index:

| | agree | |
|---|---|---|
| all three routes identical | **0 / 23** | (2 / 23 by source filename — the old method) |
| `vector` == `keyword` | 0 / 23 | |
| `vector` == `hybrid` | 2 / 23 | |
| `keyword` == `hybrid` | 4 / 23 | |

(The set: one full question per document plus six more spanning two documents, and the
eight rarest literal phrases in the corpus — `reversed hostnames`, `hinted handoff`,
`commit wait`, `SSTable`, `KRaft`, `chunkserver lease`, `TrueTime`, `log compaction`.
Retrieval only, no LLM in the loop, so the numbers are deterministic and re-runnable.
A different set moves the counts a little; it does not move `0/23`, because two
retrievers with genuinely different failure modes agreeing on all four of four is rare
by construction.)

The old file's own showcase example is the cleanest illustration of the bug:

```
"Explain how the ISR mechanism keeps Kafka replicas consistent."
  by source : vector / keyword / hybrid — all four chunks from kafka.md  ("identical")
  by chunk  : vector  kafka#1 kafka#4 kafka#0 kafka#2
              keyword kafka#3 kafka#2 kafka#1 kafka#0
              hybrid  kafka#1 kafka#3 kafka#2 kafka#0     <- three different prompts
```

**The routes diverge on essentially every query. The router has a real decision to make
almost every time, and it gets the important ones wrong.**

## The actual failure mode: the classifier, not the pattern

Two deterministic misroutes, both on queries this project ships in its own UI:

**`reversed hostnames`** — routed `vector`, 5 runs out of 5. Dense returns
`dynamo.md#3`, `raft.md#2`, `raft.md#3`, `chubby.md#2`: **zero** chunks of
`bigtable.md`, which is the only document in the corpus containing the phrase.
Groundedness **0.0**. BM25 returns 4/4 gold. This is the query the compare page ships
*as its retrieval-divergence demo* — the one chosen because the difference between
retrievers is maximally obvious.

**`hinted handoff`** — routed `hybrid`. Fusion drops BM25's #1 hit (`cassandra.md#4`)
out of the merged top 4 for the same RRF-consensus reason as above. `keyword` was the
better route and hybrid was not a safe superset of it.

The router's own prompt describes both of these almost verbatim under `keyword`:
*"the question names a rare, literal, technical term… finding the documents that
contain that exact token is what matters."* Two bare noun phrases, no verbs, no
paraphrase to interpret. qwen3:8b routes them elsewhere anyway, and it does so
deterministically — this is not sampling noise, it is the classifier's actual opinion.

**So the honest lesson is: the router pattern is sound and the plumbing is correct, but
an 8B classifier is not good enough to be the cheap front end** — on a corpus
deliberately rigged so the right answer is obvious. That is a more useful result than
"it works". It says the thing to measure before shipping a router is not "does routing
save money" (it does, trivially) but **routing accuracy against a labelled set**, and
it says the cheap front end has a floor below which the pattern stops being cheap and
starts being a recall tax. `auto-rag.mdx`'s "you can measure routing accuracy;
otherwise you are adding a component you cannot debug" was written in Phase 3 as
advice. It turned out to be the finding.

The obvious next moves, in increasing cost: hand-written rules for the unambiguous
cases (`query is 1–3 tokens with no verb → keyword` would fix both misroutes above,
free and fully debuggable); few-shot examples in the router prompt; a larger router
model, at which point re-derive the inequality at the top of this file because it may
no longer hold.

## The economics, corrected

The router pattern's expected value is

    P(the workers actually differ) × (cost saved by picking the cheap one)

I previously wrote that the left factor was near zero here. It is **near 1**: 0/23
queries had all three routes agree. What is near zero is the *right* factor — retrieval
on this corpus is local and free, so a correct route saves microseconds and a wrong one
costs real recall. On this deployment the product is therefore small and the sign is
arguably negative: always-hybrid would be the better engineering choice, and Auto RAG
is here to teach the pattern rather than to win the compare view.

Auto RAG earns its keep when the workers are genuinely expensive *and* the classifier is
genuinely accurate — a paid reranker, a slow graph traversal, a large model, in front of
a router good enough that its error rate costs less than the work it skips. Both
conditions, not one. This phase demonstrates the pattern and measures the condition it
fails.

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
  wrong — a thinking router still *works*, it is just 55×–190× too slow (and can
  return an empty reply outright), and nothing errors.
- Each route dispatches to the right retriever, enforced by monkeypatching the
  *other* retriever to raise. Asserting on the returned chunks alone would pass if
  both ran.
- Garbage router output falls back to hybrid **and says so** in both the trace and
  `termination_reason`.
- The router counts as an LLM call (`llm_calls == 2`). Reporting one would flatter
  the technique in precisely the view built to compare costs.
