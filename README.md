# RAG Lab

A learning website for 9 retrieval-augmented generation techniques: **read** about each
one, **run** it in a playground, and **compare** any two side-by-side on the same query.

Monorepo — `backend/` (FastAPI + RAG engine), `frontend/` (Next.js App Router + TS +
Tailwind). See [PLAN.md](PLAN.md) for the phase-by-phase build, and `LEARNINGS/` for
write-ups of what each phase taught.

## Status

Phase 1 complete. Standard RAG runs end-to-end; the other 8 techniques are listed
but not yet implemented.

Answer generation needs an Anthropic API key:

```bash
cp backend/.env.example backend/.env   # then add your key
```

Without one, retrieval still works and `POST /api/run` returns a 503 explaining
the setup step.

## Quickstart

Requires Python 3.11+ and **Node 22+** (ESLint 10 needs `>=22`; Node 21 is EOL and its
missing `util.styleText` breaks ESLint's output formatter).

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

## Dependency notes

- **ESLint config is composed by hand** in `frontend/eslint.config.mjs` rather than using
  `eslint-config-next`. That preset bundles `eslint-plugin-react` and
  `eslint-plugin-jsx-a11y`, and neither supports ESLint 10 yet (both cap their peer range
  at ESLint 9). Switching back to `eslint-config-next/core-web-vitals` once they ship
  support would restore the jsx-a11y rules we currently give up.
- **`overrides` in `frontend/package.json`** force `postcss` and `sharp` to patched
  versions. Next 16.2.12 pins `postcss@8.4.31` and `sharp@^0.34.5` internally; both carry
  open advisories and no Next release fixes them yet. Drop the overrides once one does.
