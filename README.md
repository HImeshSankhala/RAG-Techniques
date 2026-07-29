# RAG Lab

A learning website for 9 retrieval-augmented generation techniques: **read** about each
one, **run** it in a playground, and **compare** any two side-by-side on the same query.

Monorepo — `backend/` (FastAPI + RAG engine), `frontend/` (Next.js App Router + TS +
Tailwind). See [PLAN.md](PLAN.md) for the phase-by-phase build, and `LEARNINGS/` for
write-ups of what each phase taught.

## Status

Phase 0 (scaffold) complete. All 9 techniques are listed; none are runnable yet.

## Quickstart

Requires Python 3.11+ and Node 18.17+.

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
| `make index` | Ingest `backend/data/sample_docs` into Chroma (Phase 1) |
