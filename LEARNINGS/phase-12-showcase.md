# Phase 12 — REALM page + Showcase

You know how the eight runnable techniques work and how the compare view measures the
difference between two of them. This phase adds no retrieval. It is about the gap between
a system that works on your laptop and a system someone else can run, plus the ninth
technique — the one that marks where the other eight stop.

## The problem

Three problems that turn out to be the same problem.

**A project nobody can run is a project nobody can evaluate.** The quickstart in the
README said `make setup && make dev`. It never said `make index`. Follow it exactly on a
clean machine and every technique answers "Nothing is indexed yet" — the app is running,
the backend is healthy, and nothing works. There is no error to search for, because
nothing errored.

**REALM is not "not built yet".** The playground and the compare view listed it disabled,
labelled `— not built yet`, which is a promise that a later phase will deliver it. REALM
trains the retriever and the language model jointly; the training *is* the technique.
There is no version of this repo that runs it. The learn page had said so since Phase 3,
and the selector two clicks away contradicted it.

**Nine techniques with no way to see them side by side.** The home page had nine cards
with taglines. What a reader actually wants before picking one is the cost: how many model
calls, how many retrieval passes, does it need a person.

## Why this design, against the naive alternatives

### `docs_only` is declared, not derived

The tempting version: REALM is the only catalog entry outside `PIPELINES`, so
`implemented === false` already identifies it. Derive the label, add no field.

That is true today and wrong as a rule. `implemented` is derived from `PIPELINES`
membership (`registry.py`), and it answers "has anyone built this?". The selector needs
the answer to a different question: "*can* anyone build this here?". The two coincide
right now only because the catalog happens to be complete — the moment a tenth technique
is added to `CATALOG` before its pipeline lands, a derived flag would tell readers that a
half-finished technique is permanently impossible. That is the same class of error as the
one being fixed, pointing the other way.

The other tempting version is a slug check in the frontend: `t.name === "realm"`. That is
a second source of truth for a fact the backend already owns, sitting in a file that has
no way to notice when the catalog changes.

So the catalog declares it. Cost: one boolean. What it buys is that `/api/run` can now
distinguish three different mistakes — unknown slug (404), built later (409, "not yet
runnable"), never (409, "cannot run here") — instead of two.

### The range columns are prose, and they live beside `tagline`

The comparison table wants "how many model calls does this cost?". The instinct is to read
it off `Metadata.llm_calls`, which is a real measured integer the API already returns.

It is the wrong number. `Metadata.llm_calls` describes *one run*. Auto RAG's count depends
on which route its router picked for that query; Agentic's depends on how early its planner
said ANSWER. Publishing one run's integer as the technique's cost would be false for most
other runs of the same technique.

So the columns are honest ranges — `2–5`, `1–3` — and ranges are prose. The repo already
has a home for per-technique prose: `tagline`, which lives in `registry.CATALOG`, is
exposed through `api/schemas.py` and mirrored in `frontend/lib/api.ts`. Following that
precedent costs one line per technique in a file that already exists. The alternative considered
and rejected was a hand-written map in the frontend, which would have created a **fourth**
list of technique slugs (the registry, the learn route's `SLUGS`, the MDX filenames, and
the map) to avoid a change the repo had already made three times.

Every range is read off the pipeline's own code, never estimated:

| Technique | LLM calls | Where the number comes from |
|---|---|---|
| Standard, Fusion, Feedback | 1 | one `llm.generate` in `run()` |
| Auto | 2 | router call + answer call |
| Graph | 1 (+1/chunk at index time) | one at query time; extraction is `make graph` |
| Interactive | 2 | Standard's draft, then one more in `finalize()` |
| Multi-Pass | 2–5 | draft + per pass (critique + re-answer), `max_passes = 3` |
| Agentic | 2–4 | one planner call per iteration + one answer, `max_iterations = 3` |

Multi-Pass and Agentic are worth doing by hand, because they are why a single number would
not do. Multi-Pass loops `while passes < 3`, so at most two iterations run after the
initial draft, and each spends a critique plus a re-answer: `1 + 2×2 = 5`. Best case the
first critique finds no gaps and it stops at 2. Agentic plans before it drafts, so an
iteration is one call rather than two, and the answer is generated once at the end:
three iterations cost `3 + 1 = 4`, one iteration costs 2. **Agentic's worst case is
cheaper than Multi-Pass's** — 4 calls against 5 — which is a real architectural result
that the table now shows and nine cards never could.

## In place of an algorithm: the claim that expires

This phase has no algorithm. It has something better, which is a lesson that cost me a
round of review to learn.

I read PLAN.md's section on the demo corpus, found two paragraphs explaining that both of
the project's canonical "these techniques diverge" examples had been destroyed by a corpus
expansion, and concluded the headline demo had no subject. I wrote that up as the biggest
risk in the phase: possibly unshippable, someone should go hunting for a query where two
techniques visibly differ.

I had stopped reading one line early. The next block is headed *"Live replacements,
measured against the 43-chunk index"*. Further, five divergence presets already ship in
the compare view, and four `compare.preset-*` claims in `evals/retrieval.py` pin their
captions to the live index. Running `make eval` answers the question in 1.7 seconds:
25 claims, 25 ok. Measuring it directly confirms 1-of-4 overlap on
*"What are reversed hostnames used for?"* and 4-of-4 on the Raft control.

Two things worth keeping from that:

**A sentence about retrieval is a test, not prose.** "Dense misses `bigtable.md` entirely"
is a claim about a specific index. Re-chunk the corpus and it can become false without a
single line of code changing, without any test going red, and without anyone noticing —
which is exactly what happened to PLAN.md's two original examples for two entire phases.
The fix was not to write the claim more carefully. It was to make the claim executable.
That is why `make eval` exists, why every preset caption has a claim id, and why
`assets/RECORDING.md` opens by telling the owner to run it before pressing record.

**Prose in a planning document decays faster than code, and reads as authoritative
anyway.** The stale paragraph was confidently written, correctly reasoned, and wrong,
while the harness three files away was green. Given a conflict between a document and a
thing you can run, run the thing. I did not, and the review caught it.

## Failure mode: the single-user assumption that only surfaces at deploy

Feedback RAG (Phase 11) stores votes in SQLite at `backend/rag_lab.db`, keyed to the
passage, applying to every later question. Interactive RAG keeps its drafts in the same
file. Locally this is correct and is the whole point — the lesson is that persisted
feedback reshapes future rankings, and you are supposed to feel it.

Deploy it and the same design becomes a bug. There is no per-visitor isolation and no
session scoping: one visitor's thumbs-down demotes a passage for **every** subsequent
visitor, permanently, and the only undo is `make reset-feedback` on the host. A visitor who
downvotes the best chunk in the corpus has quietly degraded the demo for everyone who
arrives afterwards, and the second lesson of Phase 11 — that a demoted passage stops being
shown and therefore stops being votable, so the dead end is unreachable — now applies to
strangers who never cast a vote.

Nothing detects this. Every test passes, because every test runs single-user. It surfaces
only when the deployment story is written down, which is the actual reason a "deploy notes"
section earns its place in a project that may never deploy: it forces you to say who else
is touching your state.

**The edge case inside the edge case:** option 1 in the deploy notes (flip
`LLM_BACKEND=anthropic`, supply a key through host secrets) has the same shape.
`ANTHROPIC_MAX_SESSION_CALLS` reads like a spend guard, and locally it is one. It is
per-process, so on a host it caps the *server*, not the visitor — one visitor cannot be
stopped from consuming the budget of every other. The real cap remains the Anthropic
Console spend limit, which is why the README states it as the real cap in both places.
