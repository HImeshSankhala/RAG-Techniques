# RAG Lab — Learning Website Project Plan

A full-stack learning website for 9 RAG techniques: **read** about each one, **test** it live in a playground, and **compare** any two side-by-side on the same query.

**Goals:**
1. Learn RAG architectures by implementing them.
2. Learn system design by building a real frontend + API + engine stack.
3. End with a portfolio-grade project.

---

## Architecture Overview

```
┌─────────────────────────────┐
│  Frontend (Next.js + TS)    │  Learn pages · Playground · Compare view
└──────────────┬──────────────┘
               │ REST (JSON)
┌──────────────▼──────────────┐
│  Backend (FastAPI)          │  /api/techniques · /api/run · /api/compare
│  - Pipeline registry        │  Thin layer: validates input, calls engine
└──────────────┬──────────────┘
               │ Python calls
┌──────────────▼──────────────┐
│  RAG Engine (core/ + impls) │  RAGPipeline ABC · 9 technique classes
│  ChromaDB · embeddings · LLM│  Shared infra written ONCE
└─────────────────────────────┘
```

**Key design decision:** the engine knows nothing about HTTP, and the frontend knows nothing about vector stores. Each layer talks only to the one below through a small contract. This is why the compare feature costs ~20 lines: `POST /api/compare` just calls `registry[a].run(q)` and `registry[b].run(q)` and returns both `RAGResult`s.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 14 (App Router) + TypeScript + Tailwind | Industry standard; file-based routing maps cleanly to Learn/Playground/Compare pages |
| Diagrams | Mermaid (rendered client-side) | Diagrams live as text in markdown — versionable, no image files |
| Content | MDX files (one per technique) | Write docs in markdown, embed interactive components |
| Backend | FastAPI + Pydantic | Auto-generated OpenAPI docs; Pydantic models = typed contract with frontend |
| RAG engine | Python: `core/` module + technique classes | Strategy pattern |
| LLM | Anthropic API via thin wrapper in `core/llm.py` | Provider-swappable |
| Embeddings | sentence-transformers (local) | Free while learning |
| Vector store | ChromaDB (embedded) | Zero infra; swappable later |
| Graph (Graph RAG) | NetworkX in-memory | Start simple |
| Feedback store | SQLite | Enough for feedback RAG |
| Tests | pytest (engine) + basic API tests | Smoke tests per pipeline |
| Dev orchestration | Two dev servers + one `Makefile` | `make dev` runs both |

---

## Repo Structure

```
rag-lab/
├── README.md                    # Project intro, screenshots, quickstart
├── CLAUDE.md                    # Claude Code instructions (below)
├── PLAN.md                      # This file
├── Makefile                     # dev, test, index, lint targets
├── backend/
│   ├── pyproject.toml
│   ├── .env.example             # ANTHROPIC_API_KEY=...  (never commit .env)
│   ├── core/
│   │   ├── config.py            # pydantic-settings
│   │   ├── ingest.py            # pdf/md/txt loaders → Document
│   │   ├── chunking.py          # fixed + recursive strategies
│   │   ├── embeddings.py        # sentence-transformers wrapper
│   │   ├── vectorstore.py       # Chroma add/query wrapper
│   │   ├── llm.py               # LLM client wrapper
│   │   └── pipeline.py          # RAGPipeline ABC + RAGResult
│   ├── implementations/
│   │   ├── registry.py          # name → pipeline instance (single source of truth)
│   │   ├── standard_rag.py
│   │   ├── fusion_rag.py
│   │   ├── multi_pass_rag.py
│   │   ├── auto_rag.py
│   │   ├── graph_rag.py
│   │   ├── agentic_rag.py
│   │   ├── interactive_rag.py
│   │   └── feedback_rag.py      # (REALM is docs-only, no class)
│   ├── api/
│   │   ├── main.py              # FastAPI app, CORS for localhost:3000
│   │   ├── schemas.py           # Pydantic: RunRequest, RunResponse, CompareResponse
│   │   └── routes/
│   │       ├── techniques.py    # GET /api/techniques (list + metadata)
│   │       ├── run.py           # POST /api/run {technique, query}
│   │       ├── compare.py       # POST /api/compare {technique_a, technique_b, query}
│   │       └── feedback.py      # POST /api/feedback (for feedback RAG)
│   ├── data/sample_docs/        # 3–5 docs users query against
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── app/
│   │   ├── page.tsx             # Home: what is RAG, grid of 9 technique cards
│   │   ├── learn/[slug]/page.tsx    # Renders MDX for one technique
│   │   ├── playground/page.tsx      # Pick technique → run query → see result
│   │   └── compare/page.tsx         # Pick TWO → same query → side-by-side
│   ├── content/                 # MDX: one file per technique
│   │   ├── standard-rag.mdx
│   │   ├── ...                  # what it is, Mermaid diagram, trade-offs,
│   │   └── realm.mdx            # when to use / NOT use, real-world example
│   ├── components/
│   │   ├── TechniqueCard.tsx
│   │   ├── ResultPanel.tsx      # answer + chunks + steps trace + latency badge
│   │   ├── CompareView.tsx      # two ResultPanels + diff summary row
│   │   ├── StepsTrace.tsx       # visualizes RAGResult.steps
│   │   └── MermaidDiagram.tsx
│   └── lib/api.ts               # typed fetch client mirroring backend schemas
└── assets/                      # screenshots for README
```

---

## Core Contracts (everything hangs off these)

### Engine contract (Python)

```python
# backend/core/pipeline.py
@dataclass
class RAGResult:
    answer: str
    retrieved_chunks: list[Chunk]      # text + source + score
    steps: list[Step]                  # [{name, detail, duration_ms}] — powers UI trace
    metadata: dict                     # latency_ms, llm_calls, tokens, passes

class RAGPipeline(ABC):
    name: str          # "fusion-rag" (slug, matches MDX filename)
    display_name: str  # "Fusion RAG"
    tagline: str       # one-liner for cards

    @abstractmethod
    def run(self, query: str) -> RAGResult: ...
```

### API contract (mirrored in frontend/lib/api.ts)

- `GET  /api/techniques` → `[{name, display_name, tagline, implemented: bool}]`
- `POST /api/run`        → `{technique, query}` → `RunResponse` (RAGResult + technique name)
- `POST /api/compare`    → `{technique_a, technique_b, query}` → `{a: RunResponse, b: RunResponse}`
- `POST /api/feedback`   → `{technique, query, chunk_ids, rating}` → `{ok: true}`

**Why `steps` matters:** every pipeline logs its stages ("Embedded query — 12ms", "Retrieved 5 chunks", "Pass 2: found gaps: [dates]"). The frontend renders this as a trace timeline. In compare mode, seeing Standard RAG's 3 steps next to Multi-Pass's 9 steps IS the lesson.

---

## Phases

Each phase ends demo-able. Do not start N+1 until N runs.

### Phase 0 — Scaffold
- Monorepo layout above; backend installs; frontend boots; `make dev` runs both
- CORS configured; `GET /api/techniques` returns hardcoded list; home page renders 9 cards from it
- **Done when:** browser shows 9 cards fetched from the API.

### Phase 1 — Engine: core + Standard RAG
- `core/` modules; ingest sample docs into Chroma (`make index`)
- `StandardRAG` implementing the ABC, registered in `registry.py`
- pytest smoke test
- **Done when:** `POST /api/run {"technique":"standard-rag", ...}` returns grounded answer with chunks + steps.

### Phase 2 — Playground page
- Technique selector (only `implemented: true` enabled), query box, ResultPanel
- Show answer, expandable chunks with sources, StepsTrace, latency badge
- **Done when:** you can test Standard RAG fully from the browser.

### Phase 3 — Learn pages (all 9, content first!)
- MDX per technique: what/why, Mermaid diagram, trade-off table, when NOT to use
- `/learn/[slug]` renders MDX; cards link to it; "Try it" button → playground pre-selected
- **Done when:** all 9 learn pages readable even though only 1 is runnable. The site is already a useful learning resource.

### Phase 4 — Fusion RAG
- Dense (Chroma) + BM25 (`rank_bm25`) retrievers, merged with Reciprocal Rank Fusion
- **Learning focus:** scatter-gather; why rank-based fusion beats naive score merging (scores from different retrievers live on different scales — ranks are comparable).

### Phase 5 — Compare page (the headline feature)
- Two selectors + one query → `POST /api/compare` → side-by-side ResultPanels
- Diff summary row: latency A vs B, LLM calls, chunks overlap %, steps count
- Preset "interesting queries" that highlight differences
- **Done when:** Standard vs Fusion on the same query shows different chunks and you can explain why.

### Phase 6 — Multi-Pass RAG
- Draft → self-critique for gaps → targeted re-retrieval → final (max 3 passes)
- **Learning focus:** loop termination; paying latency for accuracy. Compare vs Standard is now dramatic.

### Phase 7 — Auto RAG
- Cheap LLM router call → vector / keyword / hybrid path
- **Learning focus:** router pattern — cheap classifier in front of expensive workers.

### Phase 8 — Graph RAG
- LLM entity/relation extraction at index time → NetworkX graph → subgraph retrieval
- Bonus: render retrieved subgraph in ResultPanel
- **Learning focus:** multi-hop questions where similarity search fails.

### Phase 9 — Agentic RAG
- Plan → retrieve → assess loop with max iterations; steps trace shows agent reasoning
- **Learning focus:** stopping criteria; cost control in agent loops.

### Phase 10 — Interactive RAG
- Playground gains a feedback round-trip: draft → user marks helpful chunks / adds hint → final
- **Learning focus:** human-in-the-loop; API becomes two-step (needs a session/draft id).

### Phase 11 — Feedback-Based RAG
- Thumbs up/down on chunks → SQLite → future rankings boost/demote (simple weight)
- **Learning focus:** online feedback loops; why naive boosting can create filter bubbles.

### Phase 12 — REALM page + polish
- REALM learn page (why it can't run locally; paper link)
- README with screenshots; comparison table on home page; deploy notes (Vercel + Fly.io/Railway) — optional

---

## CLAUDE.md (copy to repo root)

```markdown
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
- Keep costs low: local embeddings, small sample docs, small model for router calls.

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
```

---

## How to run this with Claude Code

The Guardrails in CLAUDE.md auto-apply every session (Claude Code reads CLAUDE.md
automatically), so per-phase prompts stay SHORT — you don't retype the rules.

**First session only:**
> "Read PLAN.md and CLAUDE.md. Then do Phase 0."

**Every session after:**
> "Do Phase N from PLAN.md."   (e.g. "Do Phase 4 from PLAN.md")

Because of the Guardrails, each of those automatically: implements only that phase,
verifies against the "done when" check, checks docs when unsure, writes the
LEARNINGS file, and stops for your review. You only supply the phase number —
which is your checkpoint that you understood the previous phase.

## Working Rules (for you)

- One phase per branch, merged via PR — even solo.
- Don't say "Phase N+1" until Phase N runs AND you've read its LEARNINGS file and
  can explain it out loud. Explaining it is the test.
- Read every diff. Ask Claude Code "why?" on anything unclear before merging.
- Content (Phase 3) before more engines — the site should be useful early.
