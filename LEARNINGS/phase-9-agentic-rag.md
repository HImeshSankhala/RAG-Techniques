# Phase 9 — Agentic RAG

The second technique with a loop, which makes "what does an agent add over Multi-Pass?"
the only question worth answering. The honest answer is one thing, and it is not the loop.

## The problem

Phase 6 let the pipeline retrieve more than once. It still retrieved the same *way* every
time: `vectorstore.query(embed(text), k)`, with the critique choosing only the text. Phase 7
let a router choose the retrieval strategy, but exactly once, before it had seen a single
passage — so a wrong route is unrecoverable, and the router is guessing from the question's
wording alone.

Agentic RAG makes the strategy a decision taken *repeatedly, with evidence in hand*:

    plan -> search(<strategy the model chose>, <query the model wrote>)
         -> assess & re-plan -> ... -> answer

That is the whole of what "agentic" means in this project. Not "it loops" — Multi-Pass
loops. **The action space is larger than one.**

## Two structural choices, both of which pay

**Assess and plan are one LLM call, not two.** PLAN.md describes plan → retrieve → assess. An
assessment's only consumer is the plan that follows it, so producing a verdict and then
re-supplying that verdict as input to a second call buys nothing but a call. Iteration 1 is
labelled `Plan` in the trace and later ones `Assess & re-plan`, which is what they are.

**There is no draft.** Multi-Pass generates a full answer every pass and buys a second call to
critique it: 2 calls per pass. An iteration here is one planner call, with a single answer
generated at the end. Three iterations cost 4 calls against Multi-Pass's 5 for three passes.

The second choice has a correctness consequence that is easy to miss. Phase 6's first critique
prompt asked whether the *draft* answered the question — and a draft saying "the passages do
not mention Spanner" satisfies that reading perfectly, so the critique approved everything and
the loop never ran. **Planning before drafting makes that bug unrepresentable**: with no draft
in existence there is nothing to grade except the passages.

## Termination

Four conditions, and only the first is a guarantee:

1. **`settings.agentic_max_iterations`** (3). In config beside the spend caps, for the reason
   Phase 6 established: an iteration cap on a self-terminating loop is a budget guarantee, not
   a tuning knob.
2. **`agent_stopped`** — the planner replied ANSWER.
3. **`repeated_action`** — the planner asked for a search it already ran.
4. **`no_new_evidence`** — a genuinely new search returned only chunks already held.

(3) is new to this phase and it exists because the action space is larger than one. Multi-Pass
can only notice a repeat *after* paying for the retrieval it produced; here the planned action
is a comparable value, so the repeat is caught before the retrieval runs. `Action.key` includes
the strategy, so the same words through BM25 after a dense miss is not a repeat — that is the
one repetition worth paying for.

(3) and (4) look alike and diagnose opposite problems: a bad planner versus a thin corpus.
Collapsing them into one reason would hide which.

`agent_stopped` is deliberately not spelled `no_gaps_found` or `gaps_closed`. Those report a
critique's verdict on a draft that already exists; this is a prediction about evidence made
before anything was written. How many searches it took is already in `retrieval_passes`, so it
does not need a second termination value to say so.

## The extraction decision: three bounded loops, and I did not extract

This is the third bounded loop in the project — Multi-Pass caps passes, `core.graph.traverse`
bounds a BFS, and this caps iterations. CLAUDE.md says abstract on the third repetition. I read
all three and decided not to, which needs justifying rather than asserting.

**What is actually common** is three lines: a counter bounded by a config value, a `reason`
variable pre-set to the cap's name, and `break`-with-reason at each early exit. Everything
around them differs:

| | Multi-Pass | `graph.traverse` | Agentic |
|---|---|---|---|
| body | 2 LLM calls + retrieve | frontier expansion, no LLM | 1 LLM call + retrieve |
| second bound | — | `MAX_NODES`, checked in an inner loop | — |
| carries | tokens, calls, chunks, asked | depth map, frontier, hops | tokens, calls, evidence, history |
| exits | 4 | 4 | 4 |
| exit vocabulary | shared with nothing | shared with nothing | shared with nothing |

A `BoundedLoop` helper could own the counter and the reason. Each pipeline would still write
its own body, its own step recording, its own token accounting, its own merge. And the graph
walk does not fit at all: it has no tokens and no cost, it returns from the middle of an inner
loop on a second budget only it has, and forcing it in would mean growing the abstraction an
"extra budget" concept with one user. The termination vocabularies are not shared either —
`node_budget` and `repeated_action` mean something only to the loop that emits them, so a
shared class either takes them as opaque strings (adding nothing) or tries to own them (wrong).

Net: three lines saved per call site, one new object's semantics to learn, and a forced
migration of a working, measured file. CLAUDE.md's rule is "shorter is better *when it is also
clearer*," and this would be shorter and less clear.

**The extraction the project actually wants is a different one.** Look at what repeats:

```python
llm_calls += 1
tokens_in += response.input_tokens
tokens_out += response.output_tokens
...
cost = llm.estimate_cost_usd(tokens_in, tokens_out) if response.backend == "anthropic" else 0.0
```

The accumulating form is in Auto RAG, Multi-Pass and Agentic — three sites, rule of three
satisfied — and the cost ternary is in five. That is a real duplication with a real name
(a per-run LLM ledger), and it is not loop machinery. **Not done here** because `auto_rag.py`
is outside this phase's file ownership, and a migration that converts two of three call sites
leaves the project worse than either state. Flagged for whichever phase owns those files.

## Algorithms and cost

**The loop** is O(I) LLM calls for I iterations, I ≤ 3: at most 4 calls (3 plans + 1 answer).
There is deliberately no assess call after the final search — the cap is already reached, so
nothing could act on its verdict, and a call whose answer must be discarded is not worth buying.
This is the same reasoning that leaves Multi-Pass without a critique after its final redraft.

**Repeat detection** is O(1) amortised against a set of at most 3 normalised keys. The
normalisation is the load-bearing part: the parser strips **quotes**, and the key lowercases and
collapses whitespace. Measured, this model writes `SEARCH keyword "Kafka replaced ZooKeeper"`
about half the time, and two spellings of one search would otherwise slip past the guard.

**Merging** is O(n) with a `seen` set on `chunk_id`, order preserved. Not re-ranked by score,
for the reason Phase 4 established: scores from dense, BM25 and RRF are three different rulers,
and this pipeline mixes all three within a single run. Concatenating them in retrieval order is
the only honest arrangement.

**Context growth is bounded by arithmetic, not luck:**

    max_iterations x ACTION_TOP_K  =  3 x 3  =  9 chunks

At `max_chunk_chars` 1500 that is ~13.5k chars, ~3.4k tokens — inside the 8192 `num_ctx` from
Phase 1, with room for the planner's instructions and the answer. `ACTION_TOP_K` is 3 rather
than `top_k`'s 4 to make that sum work, because the planner is shown *all* accumulated evidence
on every iteration. Raising either constant without redoing this arithmetic silently
reintroduces the Phase 1 truncation bug.

## Measured: the assess step is the project's weakest component, and now there are numbers

Fourteen live runs on local `qwen3:8b`, four question shapes:

| | observed |
|---|---|
| stopped after exactly one search | **11 of 14** |
| assess step *chose* to search again | **1 of 14** |
| continued only because the plan was unreadable | 2 of 14 |
| hit the iteration cap | 1 of 14 |
| planner replies that came back empty | **7 of 51 calls (14%)** |

The defining feature — a loop whose length the model decides — fired once in fourteen runs, and
twice as often the loop continued by accident rather than by decision.

Three of the four termination reasons have been seen against the real index: `agent_stopped`
(11), `max_iterations` (1), `no_new_evidence` (1). `repeated_action` and `empty_index` are
unit-tested and have not fired live — the same caveat Phase 6 recorded about `gaps_closed`, and
worth keeping visible rather than treating a tested path as a proven one.

### Thinking is load-bearing, again

Held at the hardest state on this corpus (a two-part question; the passages answer part one and
contain nothing about part two):

| planner setting | verdict |
|---|---|
| `reason=False` | `ANSWER` **3/3**, 3 output tokens, 0.1s |
| `reason=True` | never named the real gap in 6 samples — 1 `ANSWER`, 2 searches for material it already had, 3 empty replies |
| `reason=True` + "check each part separately" | named the real gap **1 of 3** |

`reason=False` makes the assessor a no-op that agrees with everything. That is Phase 6's
`COMPLETE`-to-everything result, reproduced on a different prompt, a different task and a
different question shape — which is what makes it a property of the model rather than of one
badly worded prompt. So `reason=True`, at 8–27 seconds per iteration.

On evidence that genuinely *does* cover the question, both settings correctly reply `ANSWER`
3/3. **The assessor is not indiscriminate; it is specifically bad at noticing an absence.**

### Why it stops early, and why that is not stupidity

The question: *"What replaced ZooKeeper in newer Kafka, and what was that protocol designed to
be easier than?"* The retrieved `kafka.md#4` says KRaft's motivation was "removing a second
distributed system from every deployment." That reads like an answer to "easier than what." The
real answer — Raft was published as an understandable alternative to **Paxos** — is in
`raft.md#1`, which nothing in this project retrieves for that question, Graph RAG included.

**The assess step cannot detect a gap that the evidence plausibly appears to fill**, because
detecting it requires already knowing the fact that is missing. That is the general shape, and
an agent loop compounds it across iterations rather than washing it out.

The "check each part separately" paragraph shipped because it is the only variant in 12 samples
that ever named the actual missing hop, and it does not regress the sufficient-evidence case. It
is a small improvement to a step that is still wrong most of the time, and the code says so.

### Against the baseline, it loses

Same question, same model, same index:

| | Standard RAG | Multi-Pass RAG | Agentic RAG |
|---|---|---|---|
| latency | **7.3s** | 30.4s | 32–73s |
| LLM calls | **1** | 2 | 3–4 |
| retrieval passes | 1 | 1 | 1 (3 once) |
| second hop | wrong | wrong | **wrong, identically** |

All three produce the same wrong sentence. Agentic RAG pays 4–10x the latency to be wrong in
the same way, including on the one run where the loop did fire and gathered five chunks.

What planning *does* buy: in **8 of 14 runs** at least one retrieved chunk was outside dense
top-4 on the raw question, and `Compare against plain dense retrieval` names them in the trace
on every run. The retrieval genuinely changes. It never changed for the better here.

The diagnosis is corpus size and model size, not agents. Nine documents are small enough that
the first search usually holds whatever exists, so there is nothing for iteration 2 to find —
and the assessor is right to stop more often than it is wrong. An 8B local model is not a
reliable judge of absence. Both conditions have to fail before the loop earns its latency.

## The failure mode: an empty reply is not a decision

14% of planner calls returned nothing. Ollama draws thinking tokens from the same
`num_predict` budget as the reply, so a planner that thinks hard about a hard question exhausts
the budget before writing its one line — and the harder the question, the likelier the silence.
Measured cleanly: on one evidence set the planner hit exactly 1024 output tokens and returned
empty **3 times out of 3**, at 37 seconds each.

This is Phase 6's bug at a different scale. There it was 256 tokens and the fix was a separate
local budget of 1024. Here 1024 is itself the ceiling, on a prompt that carries more evidence.

`parse_plan` returns `(action, understood)` rather than just an action precisely so that silence
cannot be read as a verdict. An unreadable or empty plan becomes a dense search on the user's
original question — **degrading to the Standard RAG baseline rather than to nothing** — and the
trace says `UNREADABLE PLAN` so it can never be mistaken for a real decision. That fallback has
no termination path of its own on purpose: on the next iteration it either recovers or produces
the identical fallback, and `repeated_action` stops it. One rule, no second exit to get wrong.

**Not fixed here, and flagged rather than decided:** raising `ollama_helper_num_predict` above
1024 would reduce the empty rate, but that constant is shared with Multi-Pass's critique, and
changing it would silently alter a technique this phase does not own. The measured empty rate is
the argument for someone making that call deliberately.

## The transferable lesson

Phase 6's was about a global inference setting silently degrading a later call type. This one is
narrower and sharper:

**An LLM asked "is this enough?" is being asked to detect an absence, and absence is the thing a
language model is worst at detecting.** It will confirm what is present and fail to miss what is
not — and every self-terminating agent loop rests its stopping decision on exactly that
question. The caps are not a safety net around a system that mostly works. On this corpus the
caps and the structural guards are doing *all* of the stopping that is worth anything, and the
model's own judgement contributed one correct continuation in fourteen runs.
