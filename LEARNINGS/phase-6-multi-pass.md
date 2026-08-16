# Phase 6 — Multi-Pass RAG

The first technique with a loop, which moves the design problem from retrieval to
termination.

## The problem

Phases 1–5 all retrieve once. That is a bet that the question's wording is close
enough to the answer's wording for a single lookup to find it. Fusion widened *how*
we search — dense and BM25 have opposite blind spots — but it still searched once,
so it cannot help when the missing evidence is only identifiable *after* reading
what came back.

Multi-Pass closes that by letting the first answer diagnose the first search:

    retrieve -> draft -> critique for gaps -> retrieve for those gaps -> redraft

The insight worth keeping: **a draft is a better query than a question**, because it
contains the vocabulary of the answer. You cannot know to search for "quorum" until
you have started answering and noticed nothing explains the consistency guarantee.

## Termination is the whole design

A loop that decides for itself whether to continue can decide "yes" forever, and on
a paid backend every turn is money. Three independent conditions stop it:

1. **A hard pass cap** (`settings.multi_pass_max_passes`, 3). Not a fallback — the
   guarantee. Everything else is best-effort; only this cannot be talked out of
   stopping by a confused model.
2. **The critique reporting no gaps** — the loop's own judgement.
3. **Re-retrieval surfacing nothing new.** A critique asking for evidence the corpus
   does not contain would otherwise burn every remaining pass fetching the same
   chunks. Redrafting on identical context yields an identical answer, so the pass
   is pure cost.

(2) and (3) make it usually stop early; (1) makes it *always* stop. The distinction
matters — the first two depend on a model behaving sensibly, and the whole point of
this phase is that it sometimes does not.

`termination_reason` reports which fired: `single_pass`, `gaps_closed`,
`no_new_evidence`, `max_iterations`. That field was defined back in Phase 1 as a
fixed `Metadata` slot with only one possible value; this is the phase it starts
carrying information.

## The cap moved to config, and the learn page was right

`MAX_PASSES` began as a module constant in the pipeline. The Phase 3 learn page —
written before any of this existed — asserted it "lives in config next to the spend
guardrails, not as a tuning parameter buried in the pipeline."

The doc was right and the code was wrong. An iteration cap on a self-terminating
loop is not a tuning knob, it is a budget guarantee: at 3 passes the worst case is
5 LLM calls, and that bound is the only reason the worst case is a *number* rather
than a hope. It belongs beside `anthropic_max_session_calls`, and it now has a test
in `test_cost_guardrails.py` rather than in the pipeline's own suite.

Worth noticing that the content-first ordering of Phase 3 paid off in a direction
nobody planned: writing the explanation before the code meant the explanation could
catch the code.

## The failure mode: `think: False` made self-critique a no-op

The headline bug, and it passed every unit test.

Phase 1 disabled qwen3's reasoning mode globally — grounded answering from supplied
passages is extraction, and thinking only slowed the trace and leaked a stray
`/think` token into answers. Correct for that call. Silently catastrophic for this
one.

Live, the critique answered `COMPLETE` on **every** query, including
`"How does Spanner use TrueTime?"` against a corpus containing no mention of Spanner
at all. Five queries, then 3-for-3 deterministic on a repeat. The loop never ran once.

Two hypotheses were wrong before the right one:

- *The prompt asks the wrong question.* Partly true and worth fixing — it asked
  whether "the draft answers the question from the passages", and a draft saying
  "the passages do not mention Spanner" satisfies that perfectly. Rewording it to
  judge the **evidence** rather than the writing changed nothing on its own.
- *The system prompt is under-weighted.* Moving the instruction into the user turn
  changed nothing.

The controlled test:

| variant | reply |
|---|---|
| system prompt, `think: False` | `COMPLETE` |
| system prompt, `think: True` | correctly names the gap, emits search queries |
| user turn, `think: False` | `COMPLETE` |

So the fix is that **thinking follows the call type, not the process**: extraction
does not need it, judgement does. `helper=True` — which already existed to mark
cheap internal calls for the tighter token cap — now also enables reasoning on the
Ollama path.

The transferable lesson is about the shape of the bug, not the flag. A global
inference setting chosen for the *dominant* call type silently degrades every
different kind of call added later, and it degrades them into plausible output
rather than errors. `COMPLETE` is a perfectly well-formed answer. Nothing raised,
nothing logged, every test green — the technique simply did not do its one job.

**This is also the phase's argument for the `steps` trace.** The only visible symptom
was `reason=single_pass` on a question about a system the corpus has never heard of.

### The same bug again, one layer down

Enabling thinking did not fix it, which was the genuinely instructive part.

The critique still reported no gaps, but now *intermittently* — same code, same query,
different answer between runs. The cause: **Ollama draws thinking tokens from the same
`num_predict` budget as the reply.** Measured on one call, 233 tokens of a 256 budget
went to reasoning; when it ran a little longer there was nothing left, and the reply
came back **empty**.

An empty reply parses as "no gaps". Identical control flow to agreement.

The 256 came from `anthropic_max_tokens_helper` — a spend guardrail for *paid* calls
that the Ollama path had been borrowing since Phase 1. Local generation is free, so
capping it there bought nothing and cost the loop. Local helper calls now have their
own budget (`ollama_helper_num_predict`, 1024) while the Anthropic cap stays at 256
where it belongs, still asserted in `test_cost_guardrails.py`.

Two guardrails that look interchangeable were doing different jobs, and reusing one
constant for both is what let a cost control silently become a correctness bug.

### Making silence audible

The deeper problem is that "the critique said nothing" and "the critique said no gaps"
were indistinguishable by the time the value reached `gaps`. The pipeline now
distinguishes them in the trace:

    critique returned nothing — treating as no gaps

Same behaviour, but the trace stops reporting a verdict that was never given. A loop
whose termination depends on a model's answer has to be able to say "there was no
answer" — otherwise the failure mode is a technique that looks healthy while doing
nothing, which is precisely what happened here for an afternoon.

## Second-order: a thinking model narrates before it searches

Turning thinking on introduced a new parsing failure. The critique now replies:

```
The provided context passages do not mention Spanner or TrueTime.
Therefore, the answer is incomplete.
spanner TrueTime
external consistency
```

Line 1 is reasoning, not a query. Retrieving for it embeds the critique's own prose
instead of the missing topic — a wasted pass that also pollutes the context for
every later pass. `_parse_gaps` now keeps only lines that look like queries:
8 words or fewer, not ending in sentence punctuation. Crude, and correct for the
thing it guards against, which is prose.

Both parsing directions cost exactly one pass: misreading `COMPLETE` as a gap spends
one on nothing, misreading prose as a query spends one on the wrong thing.

## Algorithms and cost

**The loop** is O(P) LLM calls for P passes, with P ≤ 3 — at most 5 calls (draft,
critique, redraft, critique, redraft). There is deliberately no critique after the
final redraft: the cap is already reached, so nothing could act on its verdict, and
a call whose answer must be discarded is not worth buying.

**Per-gap retrieval, not concatenated.** Each gap gets its own embedding and its own
query. Averaging two unrelated questions into one vector lands between them, which
is where neither answer is. This is what "targeted" re-retrieval means — aimed at a
specific hole, not at the topic again. Cost is one extra embedding per gap, which is
milliseconds on the local model.

**Merging** is O(n) with a seen-set on `chunk_id`, preserving order. Not re-ranked by
score, for the same reason Fusion merges by rank: scores from different queries are
not comparable. Order also makes the trace readable — pass 1's evidence, then what
each pass added.

**Context growth is bounded by arithmetic, not by luck:**

    top_k + (passes - 1) x MAX_GAPS x GAP_TOP_K  =  4 + 2 x 2 x 2  =  12 chunks

At `max_chunk_chars` 1500 that is ~18k chars, ~4.5k tokens — inside the 8192 `num_ctx`
from Phase 1, with room for a 512-token answer. `GAP_TOP_K` is deliberately half of
`top_k` to make that sum work. **Raising any of those three constants without redoing
this arithmetic silently reintroduces the Phase 1 truncation bug** — the model stops
seeing some retrieved context and answers confidently anyway.

## Measured, once the critique actually worked

`"How does Dynamo achieve high write availability, and how does Raft handle log
compaction?"`, local qwen3:8b:

| | Standard RAG | Multi-Pass RAG |
|---|---|---|
| steps | 3 | **8** |
| LLM calls | 1 | 4 |
| retrieval passes | 1 | 3 |
| latency | 7.6s | **43.5s** |
| chunks in final context | 4 | 6 |
| termination | `single_pass` | `no_new_evidence` |

Nearly 6x the latency for two extra chunks of evidence. That ratio *is* the lesson,
and it is why the learn page lists "interactive UIs" under when NOT to use this.

The trace of that run also contains the failure mode the Phase 3 learn page predicted
before the code existed:

```
Critique (pass 1): gaps: Raft log compaction; Dynamo write availability
Retrieve for gaps (pass 2): 2 new          <- worked
Redraft with 6 chunks (pass 2)
Critique (pass 2): gaps: Raft log compaction; Dynamo write availability   <- identical
Retrieve for gaps (pass 3): 0 new          <- criterion (3) fires
```

The critique asked for the *same two things* after the evidence had arrived — it does
not reliably notice that a gap was filled. Termination criterion (3) is what turns
that from an infinite-ish loop into one wasted retrieval, and it is the reason that
criterion exists rather than trusting the critique to converge.

**Not yet observed live: `gaps_closed`.** Every run either had no gaps or ended on
`no_new_evidence`. That is a property of an 18-chunk corpus — `top_k` of 4 is 22% of
everything there is, so pass 1 usually already holds the relevant documents and a gap
query rarely ranks anything new. The path is unit-tested, but it has not fired against
the real index, and that is worth knowing before treating this technique as proven.

## Refactor: `core/prompting.py`

`SYSTEM_PROMPT`, `build_prompt`, and `groundedness` moved out of `standard_rag.py`.
They started there because it was the only consumer; Fusion then imported them across
module boundaries, reaching into another pipeline's privates. Multi-Pass is the third
consumer, which is where the rule of three actually fires.

They are not Standard RAG's logic — they are the project's answer to "how do you show
passages to a model and check it used them." All three techniques now feed evidence to
the model in an identical shape, which is what lets the compare view attribute a
difference to retrieval rather than to framing.

One consequence worth knowing: `groundedness`'s denominator is every source retrieved
across all passes, so Multi-Pass is scored against more sources than Standard RAG.
That is intended — citing 3 of 8 is genuinely weaker grounding than citing 3 of 3 —
but it means the metric is not directly comparable between the two techniques, which
is a trap in the compare view.
