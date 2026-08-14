# Phase 5 — Compare page

The headline feature, and the payoff for a decision made in Phase 0.

## The endpoint is ~20 lines, and that is the point

```python
a, b = run_side(query, request.a), run_side(query, request.b)
return CompareResponse(a=a, b=b, diff=_diff(request, a, b))
```

That is the whole comparison. It works because every technique returns the same
`RAGResult`, so there is no per-pair adapter code — and adding Multi-Pass in Phase 6
makes it comparable against both existing techniques for free.

Phase 0's write-up predicted this would be the payoff and estimated the cost of the
alternative as O(n²) adapters against O(1). That held. The interesting part is that the
constraint also bound the *other* direction: `Metadata` had to become a fixed dataclass
rather than a free dict precisely so a diff row could line two runs up field by field.
A technique that invented its own metadata keys would be uncomparable.

## The diff is computed server-side

It would have been easy to compute overlap and deltas in the React component. Putting it
in the API instead means the numbers are part of the contract: any client — the compare
page, a script, a future CLI — gets identical arithmetic. The diff is the thing a reader
is meant to learn from, so it belongs where the meaning is defined, not in one renderer.

## What the comparison actually shows

Two answers side by side mostly look alike. Both are fluent paragraphs on the same topic,
and a reader skimming them concludes the techniques are equivalent. The differences that
matter are in the *evidence*, and they are structural rather than stylistic.

**Standard vs Fusion on `What is hinted handoff?`** (same model):

| | Standard RAG | Fusion RAG |
|---|---|---|
| lead chunk | `raft.md#0` — wrong document | `dynamo.md#2` — correct |
| overlap | 75% (3 of 4 shared) | |
| only A | `dynamo.md#3` | |
| only B | `mapreduce.md#0` | |

Fusion's trace states the mechanism outright:
`dense 12 (top raft.md#0), BM25 12 (top dynamo.md#2)`. Dense had the topic right and the
document wrong; BM25 matched the literal phrase.

**Standard on local vs Haiku** (same technique):

| | qwen3:8b | claude-haiku-4-5 |
|---|---|---|
| overlap | **100%** | |
| latency | 12.1s | 7.7s |
| cost | free | $0.0015 |

Identical evidence, different prose. That is the cleanest possible demonstration of the
distinction the whole site is built around: **retrieval decides what the model can know;
the model decides how it says it.** Varying one axis at a time is what makes that legible,
which is why compare takes two independent `(technique × model)` pairs rather than just
two techniques.

## The bug: `lru_cache` is not a lock

The first compare request crashed:

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

Two threads called `embeddings._model()` simultaneously. `functools.lru_cache` guarantees a
*completed* result is reused — it does **not** prevent two threads from both missing and
both executing the body. For most factories that is merely wasteful. For
`SentenceTransformer` it is fatal: torch loads weights lazily onto a `meta` device and
materialises them on first use, and two concurrent loads race that transition.

The fix is double-checked locking, so the warm path stays a single `is None` test:

```python
def _model():
    global _instance
    if _instance is None:
        with _load_lock:
            if _instance is None:
                _instance = SentenceTransformer(settings.embedding_model)
    return _instance
```

**Why it appeared only now:** every earlier phase embedded from one thread. Fusion RAG uses
a thread pool, but only its dense branch embeds — BM25 does not. Compare is the first
caller to run two full pipelines at once.

The same latent race exists in `keyword._index()` and `llm._anthropic_client()`, both also
`lru_cache`d. Neither is *broken* by it — a duplicated `BM25Okapi` or Anthropic client is
wasted work, not a crash — so they are deliberately left alone. Worth knowing they are
there, and worth fixing if BM25 ever indexes a corpus large enough that building it twice
matters.

## Parallelism that made things worse

Running both sides concurrently is the obvious design, and for two local models it is
actively harmful.

Measured: two concurrent `qwen3:8b` generations **did not complete within 600 seconds**.
Sequentially, the same pair takes about 12. `ollama ps` showed no model resident afterwards
and the machine had logged 1.9M pageouts — two 5.2GB model instances do not fit in 16GB
alongside everything else, so the system swapped instead of computing.

The fix is to parallelise only when it can help:

```python
if _both_local(request):
    a, b = run_side(a), run_side(b)      # sequential
else:
    ...ThreadPoolExecutor...            # one local + one network-bound
```

With one local and one hosted side, the two genuinely overlap and the comparison finishes
in `max(a, b)` — 12.1s measured, against roughly 16s if serialised.

The general lesson: **concurrency helps when the resources being contended are different.**
Two network calls overlap. A network call and a CPU-bound call overlap. Two calls that both
want the same 5GB of RAM do not overlap — they queue, and if the system cannot queue them
it thrashes. Ollama already serialises same-model requests internally, so there was never
throughput to win here; the only thing concurrency added was memory pressure.

The UI states this rather than hiding it — when both sides are local it shows "they run one
after the other" and sets the elapsed-timer hint to ~30s instead of ~15s.

## Edge case: `steps_delta` is a weak signal

Standard RAG and Fusion RAG both record three steps, so `steps_delta` is 0 even though the
steps are entirely different work — `Embed query / Retrieve chunks / Generate` versus
`Retrieve (dense + BM25) / Fuse by reciprocal rank / Generate`.

The count is only meaningful once a technique with a genuinely different shape exists.
Multi-Pass (Phase 6) will record nine, and *that* is when the number starts carrying
information. Until then the step *names* in the two traces say more than the delta does,
which is why both traces render in full rather than being collapsed into a number.

## What "done" means here

The done-when was "Standard vs Fusion on the same query shows different chunks and you can
explain why." Verified in a browser: the two techniques retrieved 75%-overlapping evidence,
Fusion led with the correct document where Standard did not, and Fusion's own trace names
the cause — BM25 matched the literal phrase that the embedding blurred.

Also verified: the drift axis (same technique, local vs Haiku) at 100% overlap, error
propagation from either side (404/409), and that the paid call registered on `/api/usage`
at $0.0059 of the $5 ceiling.
