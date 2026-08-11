# RAG Lab — Learning Website Project Plan

A full-stack learning website for 9 RAG techniques: **read** about each one, **test** it live in a playground, and **compare** any two side-by-side on the same query.

**Goals:**
1. Learn RAG architectures by implementing them.
2. Learn system design by building a real frontend + API + engine stack.
3. End with a portfolio-grade project.

---

## Prerequisites (one-time, on your machine — not built by Claude Code)

- **Ollama** installed and running.
- **Pull an ~8B local model** (fits both an M2 Pro 16GB and an 8GB RTX 3070, so results are
  reproducible across machines). Recommended exact tag: `ollama pull qwen3:8b` (~5GB Q4).
  **Do NOT use `qwen3.6` or `gemma4:12b`** — those are 12B–35B and won't fit the 8GB laptop.
  Whatever you pull, set `OLLAMA_MODEL` in `.env` to match.
- **Verify:** `ollama list` shows the model; `ollama run qwen3:8b "hi"` replies.
- **Anthropic API key (optional, for the compare view only):** create a key at the Claude
  Console, put it in `backend/.env` as `ANTHROPIC_API_KEY`. **Set a Console spend limit ≤ $5
  and keep auto-reload OFF** — with prepaid credit and auto-reload off, the API physically
  cannot exceed your $5. This is the real hard cap; the software guardrails below just make
  the $5 last.

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
| LLM (default) | **Ollama, local — tag `qwen3:8b`** | Free and uncapped; the default for every technique. *qwen3.6 / gemma4:12b are 12B–35B and won't fit an 8GB GPU — use an 8B tag.* |
| LLM (opt-in) | **Anthropic Haiku only**, via the same `core/llm.py` | Paid "strong model" for the compare/drift demo. Allowlisted — Opus/Sonnet are never callable. |
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
class Metadata:                        # fixed fields — the compare diff row lines these up
    model: str; backend: str
    latency_ms: float
    llm_calls: int
    retrieval_passes: int
    tokens_in: int; tokens_out: int
    termination_reason: str            # "single_pass" | "gaps_closed" | "max_iterations" | ...
    groundedness: float                # fraction of retrieved sources cited (compliance proxy)
    cost_estimate_usd: float           # 0.0 on the local backend

@dataclass
class RAGResult:
    answer: str
    retrieved_chunks: list[Chunk]      # text + source + score
    steps: list[Step]                  # [{name, detail, duration_ms}] — powers UI trace
    metadata: Metadata

class RAGPipeline(ABC):
    name: str          # "fusion-rag" (slug, matches MDX filename)
    display_name: str  # "Fusion RAG"
    tagline: str       # one-liner for cards

    @abstractmethod
    def run(self, query: str, model: str | None = None) -> RAGResult: ...
    # `model` overrides the default for one call — this is what powers
    # same-technique/different-model (local vs Haiku) drift in the compare view.
```

### API contract (mirrored in frontend/lib/api.ts)

- `GET  /api/techniques` → `[{name, display_name, tagline, implemented: bool}]`
- `GET  /api/models`     → `[{id, display_name, backend, is_paid, is_default, available, note}]`
- `GET  /api/usage`      → `{spend_estimate_usd, calls, session_calls, session_call_limit}`
- `POST /api/run`        → `{technique, query, model?}` → `RunResponse` (RAGResult + technique name)
- `POST /api/compare`    → two `(technique × model)` sides → `{a: RunResponse, b: RunResponse}`
  Supports same-model/different-technique AND same-technique/different-model (drift).
  The diff row is built from the `Metadata` fields.
- `POST /api/feedback`   → `{technique, query, chunk_ids, rating}` → `{ok: true}`

**Why `steps` matters:** every pipeline logs its stages ("Embedded query — 12ms", "Retrieved 5 chunks", "Pass 2: found gaps: [dates]"). The frontend renders this as a trace timeline. In compare mode, seeing Standard RAG's 3 steps next to Multi-Pass's 9 steps IS the lesson.

---

## Cost Guardrails — protect the $5 Anthropic ceiling

These are correctness requirements, not suggestions. The goal: it is *impossible* to
accidentally burn money.

1. **Local is the default backend everywhere.** `LLM_BACKEND=ollama` by default. Haiku is
   used only when a request explicitly asks for it (playground/compare model selector).
2. **Haiku-only allowlist.** `core/config.py` defines `ANTHROPIC_MODEL` and MUST reject any
   value that isn't a Haiku model. Opus/Sonnet are never callable from this project — a
   single expensive call could eat a large chunk of $5. Fail loudly at startup if misconfigured.
3. **Output cap.** Every Anthropic call sets `max_tokens` ≤ 512 for answers (≤ 256 for
   router/critique calls). Output tokens are the expensive side; cap them hard.
4. **Input cap.** Keep retrieval `top_k` small (≤ 5) and truncate each chunk; never stuff
   huge context into a paid call.
5. **Iteration caps on paid calls.** Multi-Pass ≤ 3 passes, Agentic ≤ 4 iterations — ALREADY
   required, but on the Anthropic backend these caps are a spend guarantee, not just latency.
   An uncapped agent loop on a paid model is the #1 way to drain credit.
6. **Spend meter.** `core/llm.py` estimates $ per Anthropic call from token counts (Haiku
   input/output rates in config) and accumulates a running total to `backend/.usage.json`
   (gitignored). Expose `GET /api/usage` → `{spend_estimate_usd, calls}`; show it as a small
   badge in the UI. This is a rough local estimate, NOT your bill — the Console spend limit
   is the real cap — but it lets you watch burn in real time.
7. **Dev safety valve.** `ANTHROPIC_MAX_SESSION_CALLS` (default 50) makes the Anthropic
   backend refuse further paid calls once exceeded in a run, so a stuck loop during
   development can't quietly rack up calls. Local (Ollama) is never capped — it's free.

Budget reality check: Haiku is ~$0.004 per single-call query, so $5 is roughly 400–1000+
queries even with Multi-Pass. The danger isn't normal use — it's an accident (a loop, or a
wrong model). The guardrails above remove those accidents.

---

## Phases

Each phase ends demo-able. Do not start N+1 until N runs.

### Phase 0 — Scaffold
- Monorepo layout above; backend installs; frontend boots; `make dev` runs both
- CORS configured; `GET /api/techniques` returns hardcoded list; home page renders 9 cards from it
- **Done when:** browser shows 9 cards fetched from the API.

### Phase 1 — Engine: core + Standard RAG
- `core/` modules; ingest sample docs into Chroma (`make index`)
- `llm.py` has TWO backends behind one `generate()`: Ollama (default, free) and
  Anthropic Haiku (opt-in, capped). **REQUIRED:** the Ollama backend must pass
  `num_ctx` = 8192 (configurable `OLLAMA_NUM_CTX`). Ollama's default context is 2048
  tokens, which SILENTLY TRUNCATES RAG prompts — the model never sees some retrieved
  context and answers confidently anyway. This is a silent-failure bug, not a nicety:
  treat it as a correctness requirement and note it in LEARNINGS.
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

### Phase 12 — REALM page + Showcase (GIFs first)
- REALM learn page (why it can't run locally; paper link)
- `.env.example` with blank placeholders; stranger-followable README quickstart
- **Demo GIFs as the README headline:** playground, compare divergence, local-vs-Haiku drift
- Comparison table on home page
- Deploy notes: the local model doesn't deploy — flip `LLM_BACKEND=anthropic` via host
  secrets, or ship frontend + GIFs only, or bring-your-own-key

### Phase 13 — Bring your own documents (OPTIONAL — decide after Phase 12)

**Not committed to. Claude must ask before building this** — see the Guardrail in
CLAUDE.md. It is scoped here so the decision is informed, not so it happens by default.

**What:** upload a PDF/MD/TXT and run the techniques against it instead of the demo corpus.

**Why it might be worth it:** "try it on your own document" is the difference between a
demo someone watches and one they use, and RAG's actual proposition is *your* documents.
Building it also demonstrates engineering a fixed corpus never does — multipart handling,
per-session collection isolation, async re-indexing, TTL cleanup.

**Why it is NOT the default, and the risk that makes this a real decision:** the demo
corpus is deliberately rigged so techniques visibly diverge — `memtable` is a term dense
retrieval ranks badly and BM25 nails (Fusion wins), and Cassandra's lineage is stated
across two files (Graph RAG multi-hop wins). An arbitrary uploaded PDF usually has none
of those properties. Someone uploads a 3-page résumé, all nine techniques return the same
chunk and the same answer, and the reviewer concludes the techniques don't matter. **A
badly-scoped version of this feature actively undermines the project's thesis.**

**If built, it is additive, never a replacement.** The curated corpus stays the default
and the demo path. Upload is a labelled second mode: "compare on the demo corpus to see
how the techniques differ, then bring your own."

**Scope (roughly one phase):**
- `pypdf` extraction (the PDF loader deliberately skipped in Phase 1 — no PDF existed yet)
- `POST /api/documents` — multipart, validate type and size
- Per-session Chroma collections, or uploads collide between users
- Re-index on upload (~20s): blocking with a progress state, or a job queue
- TTL cleanup — stored embeddings are not free
- Compare view pins both sides to one corpus snapshot, or you are comparing corpora
  rather than techniques
- Paid backend: an uploaded document is unbounded input, so the `top_k` and chunk
  truncation caps become load-bearing rather than precautionary

**Done when:** upload a PDF, run two techniques on it, and the compare view is explicit
about which corpus each side used.

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
