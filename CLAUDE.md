# RAG Lab — instructions for Claude Code

## Project
Learning website for 9 RAG techniques. Monorepo: backend/ (FastAPI + RAG engine),
frontend/ (Next.js App Router + TS + Tailwind). See PLAN.md for phases — implement
ONE phase at a time, in order, and stop for review after each.

## Architecture rules
- Engine never imports FastAPI; API never touches Chroma/LLM directly — always via pipelines.
- New technique = new class implementing core.pipeline.RAGPipeline + one line in
  implementations/registry.py. NEVER duplicate core/ logic; extend core/ if shared.
- Every pipeline populates RAGResult.steps with meaningful stage names + durations —
  the UI trace and compare view depend on it.
- Pydantic schemas in api/schemas.py are the source of truth; mirror them in
  frontend/lib/api.ts types.

## Conventions
- Python 3.11+, full type hints; TS strict mode.
- Conventional commits (feat:, fix:, docs:, refactor:, test:).
- Every pipeline: pytest smoke test against data/sample_docs.
- Keep costs low: local embeddings, local LLM by default, small sample docs.
- Commands: `make index` before first run; Ollama must be running for the default backend.

## Commands
- make dev      # backend :8000 + frontend :3000
- make index    # ingest data/sample_docs into Chroma
- make test     # pytest
- API docs at http://localhost:8000/docs

## Teaching mode (important)
The repo owner is learning system design and DSA. When implementing, explain:
- WHY this design (trade-offs: latency, cost, complexity, failure modes)
- Any algorithm used (e.g., RRF, BM25, top-k) — what it optimizes and its complexity
Keep explanations concise but real.

## Cost safety (NON-NEGOTIABLE — $5 Anthropic ceiling)
- Default backend is ollama. Never make an Anthropic call unless the request explicitly
  selects it.
- Anthropic model is HAIKU ONLY, enforced by an allowlist in config. Never wire Opus/Sonnet.
- Every paid call caps max_tokens (≤512 answers, ≤256 helper calls) and honors iteration caps.
- core/llm.py tracks estimated spend to backend/.usage.json and exposes GET /api/usage.
- Secrets only from gitignored .env; commit .env.example with blank placeholders. Never print
  or commit a key.
- Ollama backend must set num_ctx (default 8192) — the 2048 default silently truncates RAG prompts.

## Guardrails (apply to EVERY phase — no reminder needed)
1. ONE PHASE AT A TIME. When told "do Phase N", implement only Phase N.
   When its "done when" check passes, STOP and summarize. Do NOT start Phase N+1.
2. VERIFY, DON'T ASSUME. Run the code. If a "done when" check can't be met,
   report what failed rather than stubbing or faking it.
3. CHECK CURRENT DOCS WHEN UNSURE. If unsure whether a library's API, function
   signature, or parameter is current, SAY SO and look it up — never guess an API.
   Prefer pinning known-good versions in pyproject.toml / package.json.
4. WRITE A LEARNINGS FILE each phase: LEARNINGS/phase-<N>-<name>.md containing:
   - The problem this phase's technique/feature solves
   - Why the chosen approach is optimal (trade-offs vs. the naive alternative)
   - Any algorithm used + its time/space complexity
   - One failure mode or edge case
   Write it for someone who knows the previous phase but not this one.
5. FLAG DON'T SILENTLY DECIDE. If the plan is ambiguous or you'd deviate from it,
   pause and ask rather than guessing.
6. ASK ABOUT PHASE 13 (document upload) WHEN PHASE 12 COMPLETES. It is marked OPTIONAL
   and is deliberately undecided. On finishing Phase 12, raise it explicitly: restate
   the upside (a demo people use, not just watch) AND the risk (an arbitrary uploaded
   document usually shows no divergence between techniques, which makes the project
   look pointless), then let the repo owner decide. Never build it unasked, and never
   quietly drop it either.

## Minimalism — write the least code that fully works (ponytail-style)
Default to the simplest solution that satisfies the requirement. Before writing code, ask
"what is the smallest thing that works?" and build that.
- YAGNI: build only what THIS phase needs. No speculative features, options, or config
  for hypothetical futures.
- Stdlib / built-ins first. Reach for a library or a new dependency only when the stdlib
  or the framework's native feature genuinely can't do it. Prefer a native HTML input over
  a custom component; prefer a plain function over a class hierarchy.
- No unrequested abstractions. Don't add layers, wrappers, factories, or generic
  "frameworks" unless the plan asks for them or duplication actually demands it
  (rule of three: abstract on the third repetition, not the first).
- Shorter is better WHEN it's also clearer. Delete dead code, collapse needless
  indirection. But never sacrifice correctness or readability just to cut lines.

## Reconciling minimalism with the abstractions this plan DOES want
The core/ layer and the RAGPipeline interface are deliberate, plan-mandated abstractions —
they exist because 8+ techniques genuinely share infra (rule of three is satisfied many
times over). Implement them. Minimalism means "no abstractions BEYOND what's justified,"
not "no abstractions." If you think a plan-specified abstraction isn't earning its keep,
FLAG it (Guardrail 5) with your reasoning rather than silently skipping it — that trade-off
discussion is a learning goal, not an obstacle.
