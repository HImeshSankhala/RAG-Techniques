# The retrieval eval harness

`backend/evals/retrieval.py`. Run it with `cd backend && .venv/bin/python -m evals.retrieval`,
or let `make test` run the same checks through `backend/tests/test_evals.py`.

## The problem

Every learn page and every LEARNINGS file in this repo states measured facts about
retrieval — *dense misses this term*, *KRaft occurs in exactly one chunk of 43*, *the
fused top 4 loses `bigtable.md#0` at k = 7*. Each is a claim about **one index**, and
this project has invalidated at least one of them in every phase so far:

- a multi-hop example stopped being multi-hop when `cassandra.md` started stating both
  halves in a single sentence, and the page went on claiming it
- a test pinned `dynamo.md` as BM25's top hit for `hinted handoff`; `cassandra.md` moved
  it, though the claim was about retrievers, not filenames
- an RRF `k`-sweep threshold documented as `k <= 3` measured as `k <= 6` once the corpus
  grew
- a route-divergence claim was measured on filenames instead of chunk ids, and was wrong

All four were caught two phases late, by a human re-reading prose. **Prose cannot notice
that the index moved underneath it. A test can.** That is the whole of what this is.

## Why this shape and not the obvious alternatives

**Why not a data file of expectations.** The tempting design is `expectations.yaml` and a
generic runner. It does not survive contact with the actual claims. "Exactly one chunk of
43 contains KRaft", "P@1 is 3/8 over these eight queries", "membership in the fused top 4
flips between k = 6 and k = 7", and "BM25's rank 2 is `raft.md#1`" have nothing in common
except a boolean at the end. A data format expressive enough to hold all four is a
programming language, and writing an interpreter for it would be more code than the
claims. So a claim is a small record — id, where it is published, what it says, whether it
is corpus-specific — carrying a Python predicate. Data and code are not separated because
here they are not separable.

**Why retrieval only, no LLM.** The expensive claims (answer quality, groundedness, which
route the classifier picks) are also the non-deterministic ones. A harness that made them
would be slow, flaky, and unrunnable in CI, so it would not run — and a check that does not
run is worse than none, because it looks like coverage. Local embeddings + Chroma + BM25
finish 25 claims in ~2.3 s of measurement.

**Why exact comparisons and no tolerances.** Every assertion is on chunk ids, ranks,
counts, or substring presence. None is on a score. Cosine scores are floats that move with
a `sentence-transformers` upgrade; ranks do not. A tolerance on a rank is just a vaguer
claim, and the claims in the docs are not vague — the page says "rank 2", not "roughly
rank 2". The one exception is a threshold rather than a tolerance: the traversal-ball claim
asserts the 2-hop ball still covers most of the corpus, not that it covers exactly 35
chunks, because the page's argument is "the graph is not acting as a filter" and 34 would
not refute it.

## Corpus-invariant vs corpus-specific, and why both are pinned

The two kinds fail for different reasons and need different advice, so `Claim` carries the
distinction and the failure message changes with it.

**Corpus-invariant** claims are properties of the retrievers: *for a rare literal term,
dense's top hit does not contain the phrase and BM25's does*. Adding a document should not
move that. If it fails, retrieval regressed or the claim was never true — go look at
`core/`.

**Corpus-specific** claims are examples chosen for this 43-chunk index: *dense's top 4 for
the multi-hop question is `kafka.md#4`, `chubby.md#4`, `kafka.md#1`, `kafka.md#0`*. These
expire, and expiring is correct behaviour, not a bug. A multi-hop example **is** a claim
about a corpus — it says "no single passage answers this here", which is a fact about the
documents and nothing else. So the failure message does not just print a diff; it says the
example was measured on one index, names the file and section, and tells the reader to fix
the prose rather than the harness.

Where a documented table exists, both forms are pinned: the invariant one so a real
regression is caught even after a re-index, and the specific one so the page's literal
example is caught the moment it stops being true. `fusion.exact-terms-invariant` and
`fusion.exact-terms-table` are the same three queries asserted two ways for exactly that
reason.

## What a failure prints, and who it is for

The reader is someone who just added a document and has no idea which page they broke. So
the report leads with the file and section, not with the diff:

```
FAIL  compare.preset-commit-wait
      claims : dense leads with chubby.md and only the literal term finds spanner.md
      stated : frontend/components/CompareView.tsx — PRESETS[1].note
      found  : spanner.md chunks in dense's top 4 is True, documented as False
      fix    : This is an example measured on one index, not a property of the technique.
               ...
```

A test that pins `graph.traversal-ball` to a diff of chunk ids would be technically correct
and practically useless. `test_every_claim_names_a_file_to_go_fix` enforces the one thing
that makes the output actionable: every claim's `where` must name a real file.

If `corpus.shape` itself fails, the summary says so and warns that the corpus-specific
failures below it are probably downstream — 43 chunks and 9 documents is the ground the
other claims stand on.

## Complexity

Nothing interesting: the harness is O(claims × retrievals), each retrieval an HNSW walk
(~O(log N)) or a BM25 pass (O(N·q) then O(N log N)). The only engineering is caching —
`reversed hostnames` is measured by five different claims, so `dense`/`sparse`/`fused` are
`lru_cache`d and return tuples so a later claim cannot mutate an earlier claim's evidence.
The same instinct fixed the test suite: `vectorstore.count()` opens a fresh Chroma client
per call, and an autouse per-test index guard cost more than every retrieval it guarded.
Session-scoped, the eval tests went from 16 s to 7 s.

## What is deliberately NOT pinned

- **Anything with an LLM in it.** Answer text, groundedness, latency, and the router's
  actual route. Auto RAG's headline failure — qwen3:8b routes `reversed hostnames` to
  `vector` five times out of five — is a claim about a model, not an index. What *is*
  pinned is the cost of that misroute, which is pure retrieval: keyword returns 4/4
  `bigtable.md`, vector and hybrid return none of it.
- **Graph RAG's end-to-end top 4.** It needs `backend/.graph.json`, which needs Ollama.
  The one graph claim here, the 81% traversal ball, skips cleanly when the file is absent
  or stale rather than failing — a missing optional index is a setup state, not a broken
  claim.
- **Phase 7's `vector == hybrid` 2/23 and `keyword == hybrid` 4/23.** Not because they are
  unimportant, but because **the 23 queries were never written down.** The file records
  their shape ("one full question per document plus six more...") and the eight bare terms,
  not the fifteen questions. Nobody can re-run that measurement, including its author. What
  is pinned instead are the two claims that file says are robust to the choice of set —
  nobody agrees three ways, and `vector` never equals `keyword` — over a query set that
  **is** recorded, in `ROUTE_SET`, drawn entirely from queries already published elsewhere
  in the repo.

That last one is the general rule, and it is the point of the whole file:

> **A claim that cannot be expressed as a retrieval assertion is a claim this project
> should stop making.** Not because it is false — because it cannot be re-checked, so
> nobody will ever find out when it becomes false. A measurement whose inputs were not
> recorded is a number, not a result. If a page wants to assert something, the assertion
> belongs in `CLAIMS`; if it cannot go in `CLAIMS`, the page should say less.

## Failure mode: the harness that rewrites the claims it checks

The tempting move, the first time a claim fails, is to edit the doc until it passes — or
worse, to loosen the assertion. Both convert the harness into an elaborate way of
confirming whatever is currently true, which is precisely the state the project was already
in with prose.

This ships with `compare.preset-commit-wait` **failing**, and that is on purpose. The
compare view's second preset promises *"dense leads with chubby.md; only the literal term
finds spanner.md"*. Half of it holds: dense does lead with `chubby.md#2`. The other half
does not — dense's top 4 is `chubby.md#2, spanner.md#3, chubby.md#0, spanner.md#4`, and
`spanner.md#4` is one of the two chunks that literally contain "commit wait". Dense finds
the exact chunk the note says only BM25 can reach; it just ranks it fourth instead of
first. The note is true about rank 1 and false about the evidence window, which is what the
model actually sees.

The claim is pinned as written, red, until a human decides whether to weaken the note or
change the preset. A red check is information. A green check bought by editing the claim is
a lie with a CI badge on it.
