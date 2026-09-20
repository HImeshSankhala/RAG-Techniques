# RAG Lab

A learning website for 9 retrieval-augmented generation techniques: **read** about each
one, **run** it in a playground, and **compare** any two side-by-side on the same query.

Monorepo — `backend/` (FastAPI + RAG engine), `frontend/` (Next.js App Router + TS +
Tailwind). See [PLAN.md](PLAN.md) for the phase-by-phase build, and `LEARNINGS/` for
write-ups of what each phase taught.

## Status

Phases 0–11 complete. **Eight of the nine techniques run end-to-end**; the ninth (REALM)
is documentation only, on purpose — it is a pre-training method, and its learn page
explains why no amount of building makes it runnable on a laptop.

- **Read** — all 9 have a learn page at `/learn/<slug>`: what it is, a diagram, a
  trade-off table, and when *not* to use it.
- **Run** — `/playground`: pick a technique and a model, ask a question, see the answer
  with the passages it came from and a timing trace. Standard, Fusion, Multi-Pass, Auto,
  Graph, Agentic, Interactive and Feedback-Based RAG all work.
- **Compare** — `/compare` runs two `(technique × model)` sides on one query and measures
  what differed: evidence overlap, latency, cost, and which chunks only one side saw. It
  ships with five preset queries chosen because they diverge — including one 100%-overlap
  control, because a demo that only shows wins teaches the wrong lesson.
- **Home** — a comparison table of all nine: model calls, retrieval passes, whether a
  human is needed, and whether it runs.

**No API key required.** The default backend is a local model via Ollama, so everything
above runs free. A hosted Haiku model is selectable per query if you add a key — see
[Models](#models).

## The demos

Three recordings, made locally against the corpus in `backend/data/sample_docs`. They live
at `assets/playground.gif`, `assets/compare-divergence.gif` and `assets/local-vs-haiku.gif`;
if they are not there yet they have not been recorded, and
[`assets/RECORDING.md`](assets/RECORDING.md) is the script for making them. The
descriptions below are accurate either way.

1. **Playground** — Standard RAG answering from four retrieved passages, with the steps
   trace showing where the ~15 seconds went.
   <!-- Once assets/playground.gif exists, uncomment:
   ![Standard RAG answering in the playground, with its passages and timing trace](assets/playground.gif)
   -->
2. **Compare divergence** — Standard RAG vs Fusion RAG on
   *"What are reversed hostnames used for?"*. Dense retrieval returns no `bigtable.md`
   chunk at all; BM25 leads with one, and the fused evidence pulls it back. Measured
   overlap: **1 of 4**. Then the same pair on *"How does Raft elect a leader?"*, where
   overlap is **4 of 4** — fusion changes nothing, which is the honest other half.
   <!-- Once assets/compare-divergence.gif exists, uncomment:
   ![Standard RAG and Fusion RAG compared on two queries: one where their evidence barely overlaps, one where it is identical](assets/compare-divergence.gif)
   -->
3. **Local vs Haiku drift** — the same technique and query on `qwen3:8b` and
   `claude-haiku-4-5`, showing the cost and latency columns diverge.
   <!-- Once assets/local-vs-haiku.gif exists, uncomment:
   ![The same query answered by the local model and by Haiku, with differing cost and latency](assets/local-vs-haiku.gif)
   -->

The four retrieval presets have their captions pinned by `make eval` (`compare.preset-*`),
so re-indexing cannot quietly turn one of them into a lie. The fifth — the two-part Dynamo
and Raft question that makes Multi-Pass loop — is not pinned: what it claims is about how
many passes a pipeline runs, which a retrieval-only harness cannot check. The latency and
cost figures below are not pinned by anything either.

## Quickstart

Developed on macOS. Steps 1–5 are derived from the Makefile and from what CI installs,
not from a clean-machine run — if you are the first to follow them end to end and something
is missing, that is a bug worth reporting.

**1. Prerequisites**

| | Version | Checked against |
|---|---|---|
| Python | **3.11+** | `backend/pyproject.toml` (`requires-python = ">=3.11"`) |
| Node | **22+** | `frontend/package.json` (`engines.node`); ESLint 10 needs it |
| [Ollama](https://ollama.com) | any current | must be running before `make dev` |

**2. Pull the local model.** The tag must match `OLLAMA_MODEL` (default `qwen3:8b`, ~5GB):

```bash
ollama pull qwen3:8b
ollama run qwen3:8b "hi"     # verify it answers
```

**3. Install both stacks.**

```bash
make setup
```

If your Python 3.11+ interpreter is not at the default path:

```bash
make setup PYTHON=/path/to/python3.12
```

**4. Build the index — do this before the first run.** Nothing is retrievable until you
do, and every technique will answer "Nothing is indexed yet".

```bash
make index
```

**5. Start both servers.**

```bash
make dev
```

Open http://localhost:3000. API docs at http://localhost:8000/docs.

**Optional: build the knowledge graph.** Graph RAG — and only Graph RAG — needs it.
Without it that one technique answers "the knowledge graph has not been built yet"; the
other seven are unaffected. It costs **no money** (it is local Ollama), but it does cost
about **3 minutes**: one extraction call per chunk, ~43 calls on the current corpus.

```bash
make graph
```

## Bring your own documents

The playground and compare pages can run against your own files instead of the demo corpus.
Upload up to **5** files (`.txt`, `.md`, `.pdf`) — 2 MB each, 5 MB per upload, 50 pages per
PDF. They are indexed into a Chroma collection of their own, so the demo corpus is untouched,
and the corpus stops working an hour after upload.

Uploading needs two dependencies added in this phase — `pypdf` and `python-multipart` — so
**re-run `make setup`** after pulling.

Three techniques are refused on an uploaded corpus, and say so in the selector:

| Technique | Why |
|---|---|
| Graph RAG | Its graph is built once, offline, at a cost of one LLM call per chunk; it has no graph for your files. |
| Interactive RAG | Its drafts are held server-side against a corpus that expires. |
| Feedback RAG | Its votes are stored per passage to rerank future runs, which an expiring corpus cannot support. |

**Your own documents may show no difference between techniques, and that is a real result.**
The demo corpus is deliberately built so techniques diverge. An arbitrary document often has
none of those properties: every technique retrieves the same passages and answers the same
way. When that happens the compare view says the evidence was identical rather than hiding it.

## Commands

| Command | What it does |
|---|---|
| `make setup` | Create the backend venv, install both stacks |
| `make index` | Ingest `backend/data/sample_docs` into Chroma — **run before first use** |
| `make dev` | Backend on :8000 + frontend on :3000 |
| `make graph` | Build Graph RAG's knowledge graph (~43 local Ollama calls, ~3 min, free) |
| `make test` | pytest (backend) **and** Vitest (frontend) |
| `make lint` | ruff + eslint + tsc |
| `make eval` | Score retrieval against the corpus and re-check every published claim |
| `make reset-feedback` | Delete every stored Feedback RAG vote |

Uploaded corpora need no command: they expire on their own, and the backend sweeps expired
ones at startup and on the next upload.

`make eval` **does** fail loudly — it exits non-zero and names the claim that broke — but
it is deliberately not part of `make test` and not in CI. It needs a built index and the
real corpus, neither of which CI has by default. Run it yourself after changing the corpus:
it is what catches a README sentence or a preset caption that has quietly stopped being
true, which is a thing that has already happened to this repo twice.

## Models

Two backends, selectable per query in the playground:

| Model | Backend | Cost | Typical latency |
|---|---|---|---|
| `qwen3:8b` | Ollama (local) | free | ~15s |
| `claude-haiku-4-5` | Anthropic (hosted) | ~$0.002/query | ~5s |

Latencies are for a single-pass technique. Multi-Pass RAG makes up to 5 model calls per
query and takes ~45s locally — that cost *is* the lesson it teaches.

Local is the default everywhere and needs no key. The hosted path is **opt-in twice**: you
must add a key, and then select the Haiku model on the specific query you want it for.
Nothing falls through to it.

```bash
cp backend/.env.example backend/.env   # then add ANTHROPIC_API_KEY
```

### The $5 ceiling, and why Haiku only

This project is built to a **$5 Anthropic budget**, and the guardrails are correctness
requirements rather than good intentions (the full list is in
[PLAN.md](PLAN.md#cost-guardrails--protect-the-5-anthropic-ceiling)):

- **Haiku only.** `backend/core/config.py` rejects any model id without `haiku` in it, at
  startup. Opus and Sonnet are not callable from this project at all — one expensive call
  could eat a large share of $5.
- **Output capped** at 512 tokens for answers, 256 for router and critique calls.
- **Iteration caps** — Multi-Pass ≤ 3 passes, Agentic ≤ 3 iterations. On a paid model
  these are a spend guarantee, not a latency tweak.
- **A per-session call cap** (`ANTHROPIC_MAX_SESSION_CALLS`, default 50) so a stuck loop
  during development cannot quietly rack up calls.
- **A running spend estimate** at `GET /api/usage`, shown as a badge in the playground.

That badge is a local estimate, not your bill. **Set a spend limit in the Anthropic
Console and keep auto-reload off** — with prepaid credit and auto-reload off, the API
physically cannot exceed your limit. That is the real cap; everything above just makes
the $5 last.

Secrets come only from `backend/.env`, which is gitignored. `backend/.env.example` is the
committed template and its key field is blank.

## Deploying it

**The local model does not deploy.** Ollama runs on your machine; a host has no `qwen3:8b`.
So there are three honest options, and no fourth:

1. **Flip `LLM_BACKEND=anthropic` and supply a key through host secrets.** Everything
   works, including compare. But every visitor then spends *your* money against your $5
   Console cap, and `ANTHROPIC_MAX_SESSION_CALLS` is a per-process valve, not a
   per-visitor one. Only do this behind a Console spend limit you are happy to lose.
2. **Ship the frontend and the GIFs only.** The nine learn pages are prerendered from MDX
   and deploy fine on their own. Everything else needs the API: the home page fetches the
   catalog with `cache: "no-store"`, so with no backend the technique grid **and** the
   comparison table are replaced by an error notice, and the playground and compare pages
   have nothing to call. Worth doing for the writing; expect to link the GIFs for the rest,
   or give the home page a static fallback first.
3. **Bring your own key** — point readers at the Quickstart above and let them run it
   locally with Ollama, free. This is what the project is actually for.

**One caveat that matters for any hosted option.** Feedback RAG's votes and Interactive
RAG's drafts live in a single shared SQLite file (`backend/rag_lab.db`). There is no
per-visitor isolation: **one visitor's thumbs reshape every later visitor's Feedback RAG
ranking**, for everyone, permanently. That is fine — arguably instructive — for a
single-user local demo, and it is a real problem for a public one. `make reset-feedback`
on the host is the only undo.

## Dependency notes

- **ESLint config is composed by hand** in `frontend/eslint.config.mjs` rather than using
  `eslint-config-next`. That preset bundles `eslint-plugin-react` and
  `eslint-plugin-jsx-a11y`, and neither supports ESLint 10 yet (both cap their peer range
  at ESLint 9). Switching back to `eslint-config-next/core-web-vitals` once they ship
  support would restore the jsx-a11y rules we currently give up.
- **`overrides` in `frontend/package.json`** force `postcss` and `sharp` to patched
  versions. Next 16.2.12 pins `postcss@8.4.31` and `sharp@^0.34.5` internally; both carry
  open advisories and no Next release fixes them yet. Drop the overrides once one does.
