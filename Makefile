.PHONY: help setup dev dev-backend dev-frontend index test lint clean

# Python 3.11+ required. Override if your interpreter lives elsewhere:
#   make setup PYTHON=/usr/local/bin/python3.12
PYTHON ?= /opt/anaconda3/bin/python3.12
VENV := backend/.venv
PY := $(VENV)/bin/python

help:
	@echo "setup  - create backend venv + install both stacks"
	@echo "dev    - backend :8000 + frontend :3000"
	@echo "index  - ingest data/sample_docs into Chroma (Phase 1)"
	@echo "test   - pytest"
	@echo "lint   - ruff (backend) + next lint (frontend)"

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
	@echo "make index lands in Phase 1 (ingest data/sample_docs into Chroma)." && exit 1

test:
	cd backend && .venv/bin/python -m pytest

lint:
	cd backend && .venv/bin/ruff check .
	cd frontend && npm run lint
	cd frontend && npm run typecheck

clean:
	rm -rf $(VENV) frontend/node_modules frontend/.next
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
