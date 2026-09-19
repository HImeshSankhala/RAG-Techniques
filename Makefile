.PHONY: setup dev dev-backend dev-frontend index graph test lint eval reset-feedback

# Python 3.11+ required. Full path, not bare `python3.12`, because an Anaconda
# install earlier on PATH would otherwise shadow the Homebrew one. Override with:
#   make setup PYTHON=/usr/local/bin/python3.12
PYTHON ?= /opt/homebrew/bin/python3.12
VENV := backend/.venv
PY := $(VENV)/bin/python

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e "backend[dev]"
	cd frontend && npm install

# Both servers in one terminal. `trap` kills the backend when you Ctrl-C the frontend,
# otherwise uvicorn keeps holding :8000 after the shell returns.
dev:
	@trap 'kill 0' EXIT INT TERM; \
	$(MAKE) dev-backend & \
	$(MAKE) dev-frontend & \
	wait

dev-backend:
	cd backend && .venv/bin/uvicorn api.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

index:
	cd backend && .venv/bin/python -m core.index

# Graph RAG only. Separate from `index` and NOT a prerequisite of it: indexing is
# five seconds and everyone needs it, extraction is one Ollama call per chunk and
# only this one technique needs it. Measured on the current corpus: 43 calls,
# ~3 minutes. Without it Graph RAG answers "the knowledge graph has not been
# built yet" — so it is optional, not broken, and you should know the price
# before you type it. Requires the index and a running Ollama.
graph:
	@echo "Extracting the knowledge graph: one Ollama call per chunk (~43 calls, ~3 min)."
	cd backend && .venv/bin/python -m core.graph

# Backend first: it is the slower and the more informative of the two, and a
# frontend suite that runs in 250ms is not worth reordering for.
test:
	cd backend && .venv/bin/python -m pytest
	cd frontend && npm test

# Retrieval evaluation over the sample corpus. Deliberately NOT part of `make
# test` and NOT in CI: it scores retrieval quality rather than asserting
# correctness, so it is a number you read and argue with, not a gate that goes
# red. Requires the index (`make index`).
eval:
	cd backend && .venv/bin/python -m evals.retrieval

# Feedback RAG only. Deletes every stored vote, so the next run ranks like Standard
# RAG again. A maintenance command rather than a button: a demoted passage stops
# being shown and therefore stops having thumbs, and that dead end IS the lesson —
# putting an undo in the UI would quietly cancel it.
reset-feedback:
	cd backend && .venv/bin/python -c "from implementations.feedback_rag import clear_feedback; print(f'Deleted {clear_feedback()} stored votes.')"

lint:
	cd backend && .venv/bin/ruff check .
	cd frontend && npm run lint
	cd frontend && npm run typecheck
