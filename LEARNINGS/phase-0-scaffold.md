# Phase 0 — Scaffold

## The problem this phase solves

Nine RAG techniques share almost all of their machinery: the same corpus, the same
embeddings, the same LLM client, the same "here is an answer plus the chunks it came
from" shape. Only the *retrieval strategy* differs. If you build them one at a time as
nine standalone scripts, you write that machinery nine times, and comparing two of them
means reconciling nine slightly different output formats after the fact.

Phase 0 builds nothing that retrieves anything. It exists to fix the **layer boundaries
and the data contract** before any technique is written, so that the ninth technique
costs roughly what the second one did.

## Why three layers, and why this boundary

```
Frontend (Next.js)  →  API (FastAPI)  →  Engine (core/ + implementations/)
```

The rule is one-directional: the engine never imports FastAPI, and the API never touches
Chroma or the LLM directly. Each layer only knows the one below it.

The payoff is concrete and shows up in Phase 5. Because every technique returns the same
`RAGResult`, the compare endpoint is:

```python
return {"a": registry["standard-rag"].run(q), "b": registry["fusion-rag"].run(q)}
```

That's the whole feature. In the naive alternative — HTTP handling inside each technique,
or each technique returning its own ad-hoc dict — compare would need per-pair adapter code
that grows as O(n²) in the number of technique pairs you want to support. Behind a uniform
contract it's O(1): one function that works for all 36 pairs of 9 techniques.

**The trade-off is real, not free.** The uniform contract constrains what a technique can
express. Graph RAG (Phase 8) genuinely wants to return a subgraph, which `RAGResult` has
no field for. Interactive RAG (Phase 10) genuinely wants two round-trips, which
`run(query) -> RAGResult` cannot express. Both will push on this design. The bet is that
paying a small extension cost twice beats paying an integration cost nine times — and
`RAGResult.metadata` is the escape hatch that absorbs the cheap cases without a schema
change.

## The one design decision that isn't in the plan

The plan says `GET /api/techniques` returns a hardcoded list. The obvious implementation
is a literal list of nine dicts inside the route handler, each with `"implemented": false`.

I put the catalog in `implementations/registry.py` instead, and **derived** `implemented`:

```python
PIPELINES: dict[str, Any] = {}          # empty in Phase 0
implemented = name in PIPELINES          # derived, never hand-written
```

Why: a hand-maintained boolean is a second source of truth. The moment Phase 1 lands
`StandardRAG`, someone has to remember to also flip `"implemented": true` in a different
file. When they forget, the API lies — the card says "Docs only" while `/api/run` happily
serves answers, and the bug is invisible until a user reports it. Deriving the flag makes
that class of bug unrepresentable: registering the pipeline *is* marking it implemented.

This is the same instinct as normalizing a database. Store the fact once; compute
everything else from it.

## Algorithms

None yet — that's the point of a scaffold. The relevant complexity here is
**dictionary lookup**: `registry.get_info(name)` is a hash-map lookup, O(1) average,
O(n) worst case under adversarial collisions, with O(n) space for n techniques. The
alternative — scanning the catalog tuple linearly, O(n) per lookup — is genuinely fine
at n=9, and I built `_BY_NAME` anyway only because `/api/run` will do this lookup on
every request from Phase 1 onward.

Worth being honest about: at n=9, this choice is unmeasurable. It's the habit that
matters, not the microseconds.

## Failure mode: the silent fallback

The home page is a Server Component that fetches `/api/techniques` at request time. The
tempting way to make it robust is a fallback — if the fetch fails, render a hardcoded
list of nine cards so the page never looks broken.

**That's the wrong robustness.** The page would look perfect while the backend is dead,
and it would keep looking perfect after you introduce a tenth technique that the fallback
doesn't know about. You'd be debugging "why doesn't my new technique show up" against a
page that is not talking to your API at all.

So the failure is surfaced instead: the page renders an amber panel naming the URL it
couldn't reach and telling you to run `make dev`. Verified by killing the backend and
reloading — no crash, no blank page, an actionable message.

The general rule: **a fallback that hides which layer failed converts a five-second
diagnosis into a thirty-minute one.** Degrade visibly, not silently.

### A second, quieter failure mode

`fetch(..., { cache: "no-store" })` is doing load-bearing work. Next.js 14's App Router
caches `fetch` in Server Components by default, and the default is aggressive enough that
without `no-store` the home page would serve a snapshot of the catalog taken at build
time. You'd land Phase 1, flip `standard-rag` to runnable, reload — and see "Docs only",
with nothing in the logs to explain it. The framework's default is tuned for content that
rarely changes; this catalog changes on exactly the cadence that the project makes
progress.

## What "done" meant here

Not "the files exist" but: `make dev` starts both servers, the browser at
`localhost:3000` shows nine cards, and those cards came over HTTP from
`localhost:8000/api/techniques` — verified by watching the page break when the backend
was killed, then recover when it came back.
