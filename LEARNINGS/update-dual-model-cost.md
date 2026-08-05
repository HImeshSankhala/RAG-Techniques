# Update — dual-model backend and the $5 ceiling

Not a phase. A changeset applied to the already-built Phase 0/1 code after two facts
changed: a local model became available (Ollama, `qwen3:8b`), and the Anthropic budget
became a hard **$5 of prepaid credit**.

## The problem

Phase 1 had one backend and it was the paid one. Every playground query cost money.
That is survivable while a human types queries one at a time from curl — and it stops
being survivable the moment Phase 2 puts a text box in a browser and Phase 6 adds a
loop that calls the model three times per query.

The constraint is not "spend carefully." It's that **an accident must be impossible**,
because the failure mode is discovering it after the money is gone. Two accidents
dominate: calling the wrong model, and calling the right model too many times.

## The design: local by default, paid by explicit request

One `generate()`, two backends, routed by model id:

```
generate(system, user, model=None)
  model is None            -> settings.default_model  (local)
  model == ollama_model    -> Ollama    free, uncapped
  "haiku" in model         -> Anthropic paid, capped
  anything else            -> raise
```

The important property is that **routing is closed**. There is no `try local, else
paid` fallback and no default that lands on the paid backend. An unrecognized model id
is an error, because the cost of guessing wrong is money and the cost of raising is a
clear message.

Why route on model id rather than a `backend` parameter: the compare view needs to run
*the same technique on two different models* to show drift. If the caller had to
specify a backend separately from a model, those two could disagree, and reconciling
them would be another branch that could resolve toward "paid."

### Why the allowlist is a substring check

`ANTHROPIC_MODEL` is validated in `core/config.py` at import time:

```python
if HAIKU_MARKER not in value.lower():
    raise ValueError(...)
```

A substring rather than a fixed list, because a fixed list has to be maintained: when
`claude-haiku-5` ships, a list silently rejects it and someone "fixes" it by adding an
entry — and the same edit could add `claude-opus-5`. The substring encodes the actual
rule (*only the cheap tier*) rather than a snapshot of it.

Validating at **import** rather than at call time matters more than it looks. A
misconfigured model that raises on first use fails after the server is up, possibly in
front of a user, possibly after other calls have already gone out. Raising at import
means a misconfigured project cannot start.

## Defence in depth — five layers, only one of which is software I wrote

| Layer | Stops | Where |
|---|---|---|
| Console prepaid credit + auto-reload OFF | Everything. The real cap. | Anthropic Console — not code |
| Haiku-only allowlist | One catastrophic call | `config.py`, at import |
| `max_tokens` ≤ 512 / ≤ 256 | Long generations | `llm.py`, per call |
| `top_k` ≤ 5, chunk truncation | Bloated prompts | `config.py` + prompt builder |
| Session call cap (50) | Runaway loops | `llm.py`, in-process counter |
| `.usage.json` spend meter | Nothing — it *observes* | `llm.py` → `GET /api/usage` |

Two of those deserve a note.

**The spend meter prevents nothing.** It is an estimate from list prices and token
counts, written to a gitignored file with a read-modify-write and no locking. It is not
a ledger and it is not the bill. It earns its place by making burn *visible* while you
work, which is the difference between noticing a mistake in one query and noticing it
in fifty. The Console limit is what actually enforces the ceiling.

**Output is capped 2× tighter than input for a reason.** On Haiku, output tokens cost
$5/MTok against input's $1 — 5× as much. A verbose answer costs more than a large
retrieved context, so the cap goes where the money is.

## The bug worth the whole update: `num_ctx`

Ollama's default context window is **2048 tokens**. A RAG prompt is four retrieved
chunks plus a system prompt plus a question — at this project's current settings that
measures ~1082 tokens, comfortably under, so the bug does not bite today.

That is worth stating plainly rather than glossing: at `top_k=4` and `chunk_size=1200`,
`num_ctx=8192` changes nothing observable. It is insurance against the settings this
project will actually reach — `top_k=8` puts the prompt at ~1770 tokens, and Fusion RAG
(Phase 4) merges two retrievers' results into one prompt, which roughly doubles it. The
value of fixing it now is that the failure is invisible when it arrives.

What makes this dangerous is not that it truncates. It's *how*:

```
prompt: 18902 chars (~4725 tokens), canary "SECRET_CODE: PLATYPUS-42" at the very front

num_ctx=2048   model actually saw 1026 of 4162 tokens   -> "NOT FOUND"
num_ctx=8192   model actually saw 4162 of 4162 tokens   -> "PLATYPUS-42"
```

That is a real measurement from this codebase, at Ollama's **actual default** of 2048 —
not a contrived low value. All 18 indexed chunks were sent with a canary planted at the
front, then the model was asked to read it back.

Two details make this worse than ordinary truncation. First, at `num_ctx=2048` the model
saw only **1026** tokens, not 2048 — Ollama reserves the rest of the window for
generation, so the usable input is roughly half the number you configured. Second, and
the real problem: **both runs returned HTTP 200 with a fluent, well-formed answer.** No
exception, no warning, no `truncated: true` field. `NOT FOUND` is a perfectly reasonable
sentence; nothing about the response says "you did not receive what you sent."

The only external evidence is `prompt_eval_count` coming back lower than the prompt you
believe you sent — a field nobody reads unless they already suspect the problem.

This is the worst shape a bug can have in a RAG system, because RAG's whole proposition
is *the model answers from the retrieved context*. Silent truncation breaks that
proposition while leaving every visible symptom intact: retrieval succeeded, chunks
came back, an answer was produced, it cited a source. The pipeline reports success at
every stage and the answer is fabricated.

So `num_ctx=8192` is a **correctness requirement**, not tuning. It is also why
`prompt_eval_count` is surfaced into `Metadata.tokens_in`: comparing it against the
prompt you believe you sent is the only way to detect this from the outside.

**The generalizable lesson:** a default that silently discards input is worse than one
that fails. When wiring any new model runtime, find out what it does when the input
exceeds its window — and if the answer is "truncates quietly," treat configuring that
limit as part of correctness, not performance.

## `Metadata` as a dataclass instead of a dict

`RAGResult.metadata` was `dict`. It's now a fixed dataclass: `model`, `backend`,
`latency_ms`, `llm_calls`, `retrieval_passes`, `tokens_in`, `tokens_out`,
`termination_reason`, `groundedness`, `cost_estimate_usd`.

The reason is Phase 5. The compare view puts two runs side by side and diffs them field
by field. It cannot do that if each technique invents its own keys — the diff row would
need per-technique knowledge, which is exactly the O(n²) coupling that Phase 0's
uniform contract existed to avoid. A dict is more flexible and the flexibility is the
defect.

The fields with no variance yet are the point. Standard RAG always reports
`retrieval_passes=1` and `termination_reason="single_pass"`. That looks like dead
weight until Multi-Pass reports `3` and `"gaps_closed"` in the next column, and the
difference is the entire lesson of that phase rendered as two numbers.

**`groundedness` is deliberately weak and labelled as such.** It's the fraction of
retrieved sources the answer cites — a *compliance* proxy, not an accuracy measure. It
cannot tell whether a cited claim is actually supported by the passage. It catches one
specific failure well: a 0.0 means the model ignored the supplied context entirely,
which is precisely what silent truncation produces.

## Failure mode: the secret in the wrong file

While applying this update, the API key was pasted into `backend/.env.example` — a
**tracked** file — instead of `backend/.env`, which is gitignored. It was caught before
any commit, in a public repo.

The trap is that the two filenames differ by one suffix and sit in the same directory,
and the file that looks like documentation is the one that ships. Worse, the normal
workflow here runs `git add -A`, so the window between "pasted" and "public forever" is
one routine command.

The structural fix is in the template itself. `.env.example` now opens with:

```
# Template only — copy to backend/.env and fill in there.
# This file IS committed. NEVER put a real key in it.
```

A comment is weak protection, but it is at the point of use, which is where the
mistake happens. The strong protection is the one that was already working: `.env` was
in `.gitignore` from Phase 0, so the *correct* action was always safe. And a key that
has been pasted into a tracked file or a chat transcript should be rotated regardless
of whether it was committed — rotation is free, and "probably fine" is not a security
posture.

## What is verified, and what isn't

**Verified:** Standard RAG answers correctly on `qwen3:8b` with no API key set —
grounded, cited, `groundedness=1.0`, `cost_estimate_usd=0.0`, ~15s end to end. Opus and
Sonnet are refused at the config layer, the router, and the HTTP boundary. `/api/models`
reports the paid model as unavailable with the reason. `/api/usage` returns zeros.
41 tests pass.

**Not verified:** a real Haiku call. No key is configured, so the paid path — output
capping, the spend meter incrementing, `/api/usage` rising — has been exercised only
through tests and stubs, never against the live API.
