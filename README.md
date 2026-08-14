# RAG Lab

A learning website for 9 retrieval-augmented generation techniques: **read** about each
one, **run** it in a playground, and **compare** any two side-by-side on the same query.

Monorepo — `backend/` (FastAPI + RAG engine), `frontend/` (Next.js App Router + TS +
Tailwind). See [PLAN.md](PLAN.md) for the phase-by-phase build, and `LEARNINGS/` for
write-ups of what each phase taught.

## Status

Phase 3 complete.

- **Read** — all 9 techniques have a learn page at `/learn/<slug>`: what it is, a diagram,
  a trade-off table, and when *not* to use it. Readable now, even though 8 aren't built.
- **Run** — Standard RAG and Fusion RAG work end-to-end at `/playground`: pick a technique
  and a model, ask a question, see the answer with the passages it came from and a
  timing trace.
- **Compare** — `/compare` runs two `(technique × model)` sides on one query and measures
  what differed: evidence overlap, latency, cost, and which chunks only one side saw.

**No API key required.** The default backend is a local model via Ollama, so everything
above runs free. A hosted Haiku model is selectable per query if you add a key — see
Models below.

## Quickstart

Requires Python 3.11+, **Node 22+** (ESLint 10 needs `>=22`), and
[Ollama](https://ollama.com) with an ~8B model pulled:

```bash
ollama pull qwen3:8b
```

```bash
make setup
make dev
```

Then open http://localhost:3000. API docs at http://localhost:8000/docs.

If your Python 3.11+ interpreter isn't at the default path:

```bash
make setup PYTHON=/path/to/python3.12
```

## Commands

| Command | What it does |
|---|---|
| `make setup` | Create the backend venv, install both stacks |
| `make dev` | Backend on :8000 + frontend on :3000 |
| `make test` | pytest |
| `make lint` | ruff + eslint + tsc |
| `make index` | Ingest `backend/data/sample_docs` into Chroma (run before first use) |

## Models

Two backends, selectable per query in the playground:

| Model | Backend | Cost | Typical latency |
|---|---|---|---|
| `qwen3:8b` | Ollama (local) | free | ~15s |
| `claude-haiku-4-5` | Anthropic (hosted) | ~$0.002/query | ~5s |

Local is the default everywhere and needs no key. To enable the hosted option:

```bash
cp backend/.env.example backend/.env   # then add ANTHROPIC_API_KEY
```

Guardrails on the paid path, because it spends real money: Haiku only (config refuses
any other model at startup), output capped at 512 tokens, a per-session call cap, and a
running spend estimate at `GET /api/usage` shown as a badge in the playground. Set a
spend limit in the Anthropic Console and keep auto-reload off — that is the real cap.

## Dependency notes

- **ESLint config is composed by hand** in `frontend/eslint.config.mjs` rather than using
  `eslint-config-next`. That preset bundles `eslint-plugin-react` and
  `eslint-plugin-jsx-a11y`, and neither supports ESLint 10 yet (both cap their peer range
  at ESLint 9). Switching back to `eslint-config-next/core-web-vitals` once they ship
  support would restore the jsx-a11y rules we currently give up.
- **`overrides` in `frontend/package.json`** force `postcss` and `sharp` to patched
  versions. Next 16.2.12 pins `postcss@8.4.31` and `sharp@^0.34.5` internally; both carry
  open advisories and no Next release fixes them yet. Drop the overrides once one does.
