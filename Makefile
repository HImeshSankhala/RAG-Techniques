.PHONY: setup dev dev-backend dev-frontend index test lint eval

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

lint:
	cd backend && .venv/bin/ruff check .
	cd frontend && npm run lint
	cd frontend && npm run typecheck
