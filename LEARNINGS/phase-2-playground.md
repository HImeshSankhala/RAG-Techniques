# Phase 2 — Playground

Phase 1 made Standard RAG runnable from `curl`. This phase makes it runnable from a
browser, which sounds like packaging and isn't: the thing being built is not a form
that submits a query, it's a surface that shows *why* an answer came out the way it
did.

## The problem this phase solves

A RAG answer with no provenance is just a chatbot response. The value of retrieval
is that the reasoning is inspectable — you can look at what the model was handed and
decide for yourself whether the answer follows from it. A UI that renders only the
answer throws that away and keeps the expensive part.

So the panel shows four things, in descending order of how often you'll look at them:
the answer, the numbers (latency, tokens, cost, citation rate), the passages the model
was actually given, and the stage-by-stage trace.

The passages are the load-bearing one. Phase 1's write-up made the point that grounding
is *an instruction, not an enforcement mechanism* — nothing stops the model answering
from its own knowledge of Dynamo, which it certainly has. Rendering `retrieved_chunks`
is what converts that from a caveat into something a reader can check in five seconds.

## Server Component or Client Component

Both, split on a specific line: **data that exists before interaction is fetched on the
server; anything that responds to a click happens in the browser.**

- `app/playground/page.tsx` is a Server Component. It fetches the technique and model
  lists so the two dropdowns render populated on first paint. Fetching them client-side
  would mean an empty-then-filled flicker on every load, for lists that never change
  during a session.
- `components/Playground.tsx` is a Client Component. A run is a response to a click, so
  it cannot be anything else.

The alternative — making the whole page client-side and fetching everything in
`useEffect` — is one fewer file and strictly worse: two round trips before the form is
usable, and the layout shifts under the user as the lists arrive.

## Latency is a design constraint, not a detail

The local model takes **~15 seconds** to answer on this laptop. That number drove more
of this phase's UI than anything else.

Fifteen seconds of an unchanged screen reads as *broken*, not *working*. Users abandon
and re-click, which on a paid backend means paying twice. So the run button disables
itself, relabels to "Running…", and sits next to a live elapsed counter that also states
the expectation outright: `13.7s — local models take ~15s`.

That last clause is doing the real work. A spinner says "something is happening." A
counter next to a stated expectation says "this is normal, and you are 13.7 seconds into
a ~15 second wait." The first invites abandonment at second 10; the second doesn't.

Measured, same query, same retrieved chunks:

| model | latency | cost |
|---|---|---|
| `qwen3:8b` (local) | 14.9s | free |
| `claude-haiku-4-5` | 4.9s | $0.0021 |

Local is ~3x slower and free. That trade is the whole reason the model selector exists,
and putting both numbers on screen makes it something you feel rather than read about.

## Where the time actually goes

The trace bars are scaled to the slowest step *in that run*, so they answer "where did
the time go here" rather than "how does this compare to other runs." On a Standard RAG
run the answer is stark:

```
1  Embed query        821ms   ▁
2  Retrieve chunks    164ms   ▏
3  Generate answer    13.9s   ████████████████████
```

Retrieval — the part this project is *about* — is 1% of the wall clock. Generation is
93%. This is worth internalizing early, because it predicts which optimizations matter
later: Fusion RAG (Phase 4) adds a second retriever and will barely move the total,
while Multi-Pass (Phase 6) adds LLM calls and will multiply it.

## Error taxonomy: who owns which sentence

The backend already returns an actionable message per status. The temptation is to
re-word those in the UI, which duplicates the logic and lets the two drift.

The split used here: **the backend says what went wrong, the UI says what to do about it
in this interface.**

| status | backend's sentence | UI adds |
|---|---|---|
| 0 (unreachable) | "Could not reach the API at …" | "Start the backend with `make dev`" |
| 400 | "Unknown model 'x'. Available: …" | — (already actionable) |
| 429 | "Session cap of 50 paid calls reached." | "Restart the backend to reset, or use the local model." |
| 503 | "ANTHROPIC_API_KEY is not set." | "…or switch the model back to the local one." |

The 503 case shows why this matters. The backend cannot know there is a dropdown one
line up that makes the problem disappear without a key at all — that's UI knowledge. But
it *can* say precisely what is missing, and it shouldn't have to say it twice.

## Edge case: the linter caught a React purity bug

The elapsed timer originally read the clock during render:

```tsx
const startedAt = useRef(Date.now());   // ✗ Cannot call impure function
```

`react-hooks/purity` rejected it. This is not stylistic. React may call a component
function more than once per commit and may discard the result; a `useRef` initializer
evaluated during render can therefore be computed on a render that never commits, so
the "start time" can silently belong to a render that was thrown away. Under Strict
Mode's double-invocation in development, that is not hypothetical.

The fix moved the clock read into `useEffect` — which also deleted the ref:

```tsx
useEffect(() => {
  const startedAt = Date.now();
  const id = setInterval(() => setSeconds((Date.now() - startedAt) / 1000), 100);
  return () => clearInterval(id);
}, []);
```

Correct *and* smaller. Worth noting that the ESLint 10 upgrade — the one that cost a Node
version and a hand-composed flat config — is what surfaced this. `react-hooks/purity` is
a newer rule; the Phase 0 toolchain would have shipped the bug silently.

## Design decision worth defending

Unbuilt techniques stay in the dropdown, `disabled`, labelled "— not built yet":

```
Standard RAG
Fusion RAG — not built yet
Multi-Pass RAG — not built yet
...
```

Filtering them out would be cleaner and is wrong for a teaching tool. The list of nine
*is* the curriculum; seeing eight greyed entries tells you where you are in it. An
absent option reads as a bug, a disabled one reads as a roadmap.

## What "done" means here

Verified in a real browser, not asserted: selected a technique and a model, ran a query,
watched the elapsed counter, read the answer, expanded the four passages, read the trace.
Then switched the model to Haiku and ran the same query — different answer, 3x faster,
cost badge amber instead of green, spend estimate rising from $0.0022 to $0.0043. Then
killed the backend mid-session and confirmed the error panel renders the reason and the
fix rather than a blank screen.
