# Audit — Phases 0–8, plan compliance and documentation-vs-code truth

Audited against `main` @ `2412299` (Merge PR #20, phase-8-graph-rag). Worktree tree is
byte-identical to `main` (`git diff HEAD main` empty). Corpus: 9 documents, 43 chunks.
Ollama was **not** called this round — see DEFERRED.

**No security finding.** No key is committed (`git log --all -S'sk-ant'` finds only
`sk-ant-not-a-real-key` in `test_cost_guardrails.py`); `backend/.env` is gitignored and
untracked; `.env.example` ships a blank `ANTHROPIC_API_KEY=`. No uncapped paid path: the
Haiku allowlist rejects Opus/Sonnet at config import, at `resolve_backend`, at
`_generate_anthropic`, and at the HTTP boundary (verified by running each).

Phases 0–3 are clean apart from four stale sentences in their LEARNINGS files (findings
6–9, 15). Everything else below is Phases 4–8.

---

## Findings

| # | Sev | File:line | Claim | Measured reality | Proof |
|---|---|---|---|---|---|
| 1 | **BUG** | `frontend/content/fusion-rag.mdx:108` + `:113`, `LEARNINGS/phase-4-fusion-rag.md:90` | RRF `k` sweep on `reversed hostnames`: "k = 0…3 `bigtable.md#0` … **ranked above** `cassandra.md#0`"; "At `k = 4` `cassandra.md#0` overtakes the correct chunk exactly as the inequality predicts" | Off by one. At **k = 3** the two tie exactly (`1/(3+1) = 2/(3+5) = 0.25`) and the stable sort puts `cassandra.md#0` **above**. True band is `k = 0…2` above, `k = 3…6` below. `k ≥ 7` eviction and the `cassandra.md#3` (dense #9 / BM25 #8) detail are both correct. | sweep below |
| 2 | **BUG** | `LEARNINGS/phase-6-multi-pass.md:216` | Multi-Pass run table: `steps 8` alongside `LLM calls 4`, `retrieval passes 3`, `termination no_new_evidence` | That combination produces **7** steps. 8 steps only occurs on the `max_iterations` path, which costs **5** LLM calls. The table is internally inconsistent with the current code. | stubbed-LLM run below |
| 3 | **BUG** | `LEARNINGS/phase-5-compare.md:144` | "Multi-Pass (Phase 6) will record **nine** [steps]" | Structural maximum is **8** (`2 + 2×3` at `multi_pass_max_passes=3`). Nine is unreachable. | stubbed-LLM run below |
| 4 | **BUG** | `frontend/components/CompareView.tsx:38` | Compare preset note, `"What is commit wait?"`: "dense leads with chubby.md; **only the literal term finds spanner.md**" | Dense leads with `chubby.md#2` ✓, but dense's own top-4 is `chubby.md#2, spanner.md#3, chubby.md#0, spanner.md#4` — **two of the four are `spanner.md`**. Dense finds it; it just doesn't lead with it. User-visible on the compare page. | preset repro below |
| 5 | DRIFT | `frontend/content/standard-rag.mdx:69-70` | "Questions spanning documents — *which system combined X's data model with Y's replication?* needs facts from two places. Use Graph RAG." | `frontend/content/graph-rag.mdx:40-46` says that exact example stopped being multi-hop when the corpus grew, and the corpus agrees: `dynamo.md#4` and `bigtable.md#3` each state it whole, and both are in dense's top-4 for it. Two learn pages contradict each other. | corpus scan below |
| 6 | DRIFT | `LEARNINGS/phase-1-engine-standard-rag.md:127-130` | "`What is a memtable?` returns as its top hit `bigtable.md#2` — a chunk about the *tablet location hierarchy*. The chunk that actually defines a memtable ranks third" | Top hit is `bigtable.md#2` and **that chunk now contains the memtable definition** (`## Storage: SSTables and the memtable`). Nothing mis-ranks. Corrected in phase-3 and phase-4 files; phase-1's own copy still states it flat. | memtable repro below |
| 7 | DRIFT | `LEARNINGS/phase-1-engine-standard-rag.md:178-182` | "The generation step is not verified. It needs an `ANTHROPIC_API_KEY`, which is not set in this environment… Phase 1's *done when* is met only once that call runs." Also: "The request shape was checked against the installed SDK (`output_config.effort` accepts `"low"`)" | Three ways stale: the default backend is Ollama (no key needed), a key **is** configured (`GET /api/models` → `available: true`), and `.usage.json` records 3 real paid calls. Separately, **`output_config` and `effort` do not exist anywhere in the backend** — `_generate_anthropic` calls `messages.create(model, max_tokens, system, messages)`. A doc describing an API shape the code never uses. | grep + curl below |
| 8 | DRIFT | `LEARNINGS/update-dual-model-cost.md:186-191` | "*Verified:* … `/api/models` reports the paid model as **unavailable** with the reason. `/api/usage` returns **zeros**. **41 tests** pass." / "*Not verified:* a real Haiku call." | All four wrong today: models reports `available: true, "Paid. Output capped at 512 tokens."`; usage returns `{calls: 3, spend_estimate_usd: 0.005881}`; **138 passed, 6 skipped**; and `phase-5-compare.md:56-62` tabulates a live Haiku run. | curl + pytest below |
| 9 | DRIFT | `LEARNINGS/phase-5-compare.md:151` | "*What done means here* … the two techniques retrieved **75%-overlapping** evidence" | The table 100 lines above was re-measured to **25% (1 of 4)** and explicitly flags 75% as the expired figure. The done-when paragraph was not updated with it. Measured today: 25.0%. | compare repro below |
| 10 | DRIFT | `LEARNINGS/phase-7-auto-rag.md:138-154` | Route agreement "over the top-4, **23 queries** (15 full questions + 8 bare terms)": all three identical 0/23, `vector==hybrid` 2/23, `keyword==hybrid` 4/23 | Not reproducible as written. Only the **8 bare terms** are named; the 15 questions are recorded nowhere in the repo — no script, no test, no fixture. Re-running the 8 named terms gives `all3 0/8, v==k 0/8, v==h 0/8, k==h 1/8` — consistent with the table but not a reproduction of it. This is the same measurement that was already wrong once (by filename); its input set should be checked in. | 8-term repro below |
| 11 | DRIFT | `PLAN.md:158-159` vs `backend/core/pipeline.py:142` | Engine contract: `class RAGPipeline` carries `name`, `display_name`, `tagline` | No pipeline carries `display_name` or `tagline` — `StandardRAG().display_name` raises `AttributeError`. They live only in `registry.CATALOG`. The deviation is deliberate and argued in the ABC docstring (`pipeline.py:135-139`), but PLAN.md — the document this audit is measured against — was never updated, and no LEARNINGS file flags it under Guardrail 5. | `python -c` below |
| 12 | DRIFT | `PLAN.md:301-303` | Phase 13 rationale: "`memtable` is a term dense retrieval ranks badly and BM25 nails (Fusion wins), and Cassandra's lineage is stated across two files (Graph RAG multi-hop wins)" | Both premises expired with the corpus expansion, and both expiries are recorded elsewhere in the repo (phase-3 LEARNINGS, phase-4 LEARNINGS, graph-rag.mdx) — just not here. Dense ranks the memtable chunk **first**; the Cassandra lineage is stated whole in **four** chunks. This is the reasoning the owner is meant to decide Phase 13 on. | corpus scan below |
| 13 | NIT | `backend/implementations/auto_rag.py:21-22` | Thinking router "takes ~10s against ~0.5s off" | `LEARNINGS/phase-7-auto-rag.md:40-41` and `auto-rag.mdx:39-41` both say **0.07–0.23 s** off and **11.1–34.2 s** on. Three numbers for one measurement; the docstring's fast-path figure is 2–7× the documented one. (Live re-measure is DEFERRED.) | — |
| 14 | NIT | `frontend/content/multi-pass-rag.mdx:46` vs `PLAN.md:208` | "At Haiku rates that is roughly $0.01 per query rather than **$0.002**" | PLAN.md says "Haiku is ~**$0.004** per single-call query". Same quantity, 2× apart, in two documents a reader sees. | — |
| 15 | NIT | `LEARNINGS/phase-3-learn-pages.md:147` | "the *Runnable* badge appears only on `standard-rag`" | Five techniques are runnable now; the live `/learn/graph-rag` shows the badge. Historically scoped ("what done meant here") but reads as present tense. | curl below |
| 16 | NIT | `backend/api/routes/models.py:10`, `backend/api/routes/compare.py:14` | CLAUDE.md: "API never touches Chroma/LLM directly — always via pipelines" | The API imports `core.llm` in two places: `models.py` for `read_usage()`/`session_calls()` (which `GET /api/usage` in PLAN.md:171 requires) and `compare.py` for `resolve_backend()` in `_both_local`. Chroma is never touched. Defensible on both counts, but it is the one place the stated boundary bends and nothing records that. | grep below |
| 17 | NIT | `LEARNINGS/update-dual-model-cost.md:86,91` | Prompt sizing: "~1082 tokens" at `top_k=4`, "~1770 tokens" at `top_k=8` | Measured on the 18-chunk corpus. On 43 chunks a chars/4 proxy gives ~1174 and ~2152. The conclusion **strengthens** (at `top_k=8` the prompt now exceeds Ollama's 2048 default outright), and my proxy is not their tokenizer — recorded for completeness, not as a defect. | prompt repro below |

### Finding 1, in full

```
$ cd backend && .venv/bin/python -c "
from core import retrieval, keyword
from core.fusion import reciprocal_rank_fusion
d, s = retrieval.dense('reversed hostnames', 12), keyword.query('reversed hostnames', 12)
for k in range(8):
    ids = [c.chunk_id for c in reciprocal_rank_fusion([d, s], 4, k=k)]
    print(k, ids)"

k=0 ['dynamo.md#3', 'bigtable.md#0', 'raft.md#2', 'bigtable.md#1']
k=1 ['dynamo.md#3', 'bigtable.md#0', 'raft.md#2', 'cassandra.md#0']
k=2 ['dynamo.md#3', 'bigtable.md#0', 'cassandra.md#0', 'raft.md#2']
k=3 ['dynamo.md#3', 'cassandra.md#0', 'bigtable.md#0', 'chubby.md#2']   <- already below
k=4 ['cassandra.md#0', 'dynamo.md#3', 'bigtable.md#0', 'chubby.md#2']
k=5 ['cassandra.md#0', 'chubby.md#2', 'dynamo.md#3', 'bigtable.md#0']
k=6 ['cassandra.md#0', 'chubby.md#2', 'dynamo.md#3', 'bigtable.md#0']
k=7 ['cassandra.md#0', 'chubby.md#2', 'cassandra.md#3', 'dynamo.md#3']  <- evicted, as documented
```

Exact arithmetic: `bigtable.md#0 = 1/(k+1)`, `cassandra.md#0 = 2/(k+5)`. At k=3 both are
`0.25`. The order is decided by `sorted()`'s stability over `scores.items()`, i.e. by which
retriever's list was iterated first in `reciprocal_rank_fusion` — deterministic for this
code, but a tie, not a win. The prose derives `k > 3` from the strict inequality and then
tabulates k=3 on the wrong side of it.

The irony is load-bearing: this is the section whose stated lesson is *"I derived a
threshold and did not check it against a sweep; the sweep disagreed by three."* The sweep
disagrees with the corrected version too, by one. `tests/test_fusion.py:57-67` pins the
pairwise-winner behaviour on synthetic ranks, so nothing tests the band.

---

## What I verified and found CORRECT

Everything here was **run**, not read.

**The running system matches the committed code.** The reported `graph-rag:
implemented=false` staleness is gone. `uvicorn api.main:app --reload --port 8000` is the
live process (`ps aux`), `/api/health` → `{"status":"ok"}`, and `GET /api/techniques` from
the live server is **element-for-element equal** to the committed `registry.CATALOG`
including taglines, order, and `implemented` flags. The Next dev server on :3000 serves the
same commit.

**Slug drift — all four copies agree.** `registry.CATALOG` names, `PIPELINES` keys, each
pipeline's `.name` attribute, the nine `frontend/content/*.mdx` filenames, and the live
`/api/techniques` response are one identical set of nine. Every `PIPELINES[k].name == k`.
`implemented` is derived from `PIPELINES` membership, never hand-written.

**Schema mirroring — checked mechanically against the live OpenAPI, not by eye.** All ten
shared models (`Technique`, `ModelInfo`, `Chunk`, `Step`, `Metadata`, `RunResponse`,
`ComparisonSide`, `ComparisonDiff`, `CompareResponse`, `UsageResponse`) have field sets
identical to their `frontend/lib/api.ts` interfaces. Field counts 4/7/4/3/10/6/2/13/4/7.

**Architecture import rules.** `grep -rE '^\s*(from|import)\s+(fastapi|starlette)' core/
implementations/` → nothing. No `chromadb`, `core.vectorstore`, `core.embeddings`,
`core.retrieval`, `core.graph`, `core.keyword`, `core.fusion`, `core.chunking`,
`core.prompting` import anywhere under `api/`. (Two `core.llm` imports — finding 16.)

**Every pipeline populates `RAGResult.steps`.** Ran all five with the LLM stubbed: every
step has a non-empty `name`, a non-empty `detail` that says what happened rather than
restating the name, and a real `duration_ms`. Standard 3, Fusion 3, Multi-Pass 3 (early
stop) / 7 (`no_new_evidence`) / 8 (`max_iterations`), Auto 3, Graph 1 (no-graph path).

**Cost guardrails — each one exercised.**
- `Settings(anthropic_model=...)` raises `ValidationError` for `claude-opus-4-1`,
  `claude-sonnet-4-5`, `gpt-4o`; accepts `claude-haiku-4-5`. Also rejects via the
  `ANTHROPIC_MODEL` env var, i.e. at import, not at first call.
- `resolve_backend` raises `LLMError` for `claude-opus-4-1`, `gpt-4o`, `llama3`; routing is
  closed with no paid fallback.
- Live HTTP: `POST /api/run {"model":"claude-opus-4-1"}` → **400**, *"Unknown model…"*,
  before any client is constructed.
- Caps in `config.py`: answers 512, helpers 256, session cap 50, `multi_pass_max_passes` 3,
  `top_k` 4 (PLAN requires ≤5), `max_chunk_chars` 1500, `ollama_num_ctx` 8192.
  `build_prompt` truncates each passage at `max_chunk_chars`.
- `GET /api/usage` live: `{calls: 3, input_tokens: 3476, output_tokens: 481,
  spend_estimate_usd: 0.005881, session_calls: 0, session_call_limit: 50}`. Matches
  phase-5's "$0.0059 of the $5 ceiling".
- `.env`, `.usage.json`, `.graph.json` all confirmed gitignored via `git check-ignore -v`.

**Test suite.** `138 passed, 6 skipped in 5.96s`, `ruff check .` clean, every dependency
pinned to an exact version in `pyproject.toml`. Every implemented pipeline has a test file.

**Phase 4 — Fusion.** Reproduced *every* other measured claim exactly:
- dense/BM25 top-hit table: `hinted handoff` → `raft.md#0` / `cassandra.md#4`;
  `commit wait` → `chubby.md#2` / `spanner.md#4`; `reversed hostnames` → `dynamo.md#3` /
  `bigtable.md#0`. All three rows correct.
- Scale claim: BM25 tops `[6.0455, 5.7155, 6.105]` (doc says 5.7–6.1), cosine tops
  `[0.2081, 0.2116, 0.0607]` (doc says 0.06–0.21). Exact.
- `reversed hostnames` structure: `bigtable.md#0` is BM25 #1 and absent from dense's 12
  (rank 27/43 overall, cosine **−0.0395** — "does not return it at all" is fair);
  `cassandra.md#0` is #5 in **both**; `cassandra.md#3` is dense #9 / BM25 #8 and is the
  chunk that evicts at k=7. All correct.
- The 8-query benchmark, run in full: dense **P@1 3/8, R@4 5/8**; BM25 **7/8, 8/8**; fused
  **3/8, 7/8**. Every cell matches, including "fusion does not improve precision@1 at all".
- `hinted handoff` re-measurement: dense top-4 `raft.md#0, raft.md#1, chubby.md#2,
  chubby.md#0` with **no Dynamo**; `dynamo.md#2` at dense **#7**; fused top-4 `dynamo.md#2,
  chubby.md#0, chubby.md#1, raft.md#0`. Exact.
- `How is a large file split up for storage?` → dense `gfs.md#1`, BM25 `bigtable.md#1`. ✓
- `top_k=4` over 43 chunks = **9.3%**. ✓

**Phase 5 — Compare.** The headline table reproduces through the **real `/api/compare`
handler** (LLM stubbed): A = `raft.md#0, chubby.md#2, spanner.md#3, chubby.md#0`;
B = `dynamo.md#2, chubby.md#3, raft.md#0, cassandra.md#4`; overlap **1 / 25.0%**; shared
`[raft.md#0]`; only-A `[chubby.md#0, chubby.md#2, spanner.md#3]`; only-B `[cassandra.md#4,
chubby.md#3, dynamo.md#2]` — every cell as documented. Fusion's trace string is verbatim:
`dense 12 (top raft.md#0), BM25 12 (top cassandra.md#4)`. `steps_delta` is 0 between
Standard and Fusion, exactly as the "weak signal" section says. Error propagation live:
404 unknown slug, 409 documented-but-unbuilt, 422 whitespace-only query, 400 bad model.
CORS returns `access-control-allow-origin: http://localhost:3000`.

**Phase 6 — Multi-Pass.** The structural claims all hold (only the step count in one table
does not — finding 2). Forced worst case with a stubbed LLM: **5 LLM calls, 3 retrieval
passes, `max_iterations`, exactly 12 chunks** — the documented bound
`top_k + (passes−1)·MAX_GAPS·GAP_TOP_K = 4 + 2·2·2 = 12` is reached exactly, 18k chars,
inside `num_ctx=8192`. Critique calls go out `helper=True, reason=True` and answer calls
`helper=False, reason=False`, verified on the request path. `_parse_gaps` on the exact
narration example from the doc returns `['spanner TrueTime', 'external consistency']` —
the two prose lines are dropped as documented. `COMPLETE`, `COMPLETE.` and an empty reply
all return `[]`. `no_gaps_found` / `gaps_closed` / `no_new_evidence` / `max_iterations` all
reachable and distinct from `single_pass`.

**Phase 7 — Auto RAG.** Every retrieval-level claim reproduces:
- `reversed hostnames`: keyword → `bigtable.md#0,#1,#2,#3` (4/4 gold); vector →
  `dynamo.md#3, raft.md#2, raft.md#3, chubby.md#2` (zero bigtable); hybrid →
  `cassandra.md#0, chubby.md#2, cassandra.md#3, dynamo.md#3` (gold absent). The
  "hybrid is not a superset" correction is right.
- `hinted handoff`: BM25 #1 is `cassandra.md#4` and hybrid's top-4 does **not** contain it.
- The Kafka ISR illustration of the filename-vs-chunk-id bug reproduces **character for
  character**: vector `kafka#1,#4,#0,#2`; keyword `kafka#3,#2,#1,#0`; hybrid
  `kafka#1,#3,#2,#0`; all three identical by source filename, three different prompts by
  chunk id.
- `llm_calls=2` (the router counted), `termination_reason="single_pass"` (a loop outcome,
  not a route), `FALLBACK` surfaced in both the trace and the step detail, and
  `parse_route` earliest-match parsing — all as described.

**Phase 8 — Graph RAG.** Every corpus-level claim, checked against the live index:
- `KRaft` occurs in **exactly one** chunk of 43 — `kafka.md#4`. ✓
- `kafka.md#4` contains neither `paxos` nor `understandab`. ✓
- **No chunk contains both `Paxos` and `Kafka`** (paxos ∈ 12 chunks across chubby/raft/
  spanner; kafka ∈ 4 chunks; intersection empty). ✓
- `raft.md#1` and `chubby.md#3` both contain `paxos` + `understandab` and neither mentions
  Kafka. ✓
- Dense top-4 on the demo query is **exactly** `kafka.md#4, chubby.md#4, kafka.md#1,
  kafka.md#0`, hybrid fails the same way, and the honest caveat holds: BM25 alone lands
  `raft.md#1` at **rank 2**. ✓
- Cassandra lineage stated whole in **four** chunks: `cassandra.md#0`, `cassandra.md#4`,
  `bigtable.md#3`, `dynamo.md#4`. ✓
- `MAX_HOPS=2`, `MAX_NODES=80`, `MAX_TRIPLES_PER_CHUNK=8`. ✓
- `normalize('Apache ZooKeeper')` → `zookeeper`; `normalize('Google File System')` →
  `google file system` (the documented near-miss is guarded). ✓
- `drop_ungrounded` rejects **all four** documented `MapReduce | was designed to be easier
  than | …` fabrications against `mapreduce.md#0`, and `Bigtable | is replicated by |
  KRaft` against `bigtable.md#1`. The documented abbreviation cost is real: `Google File
  System` is rejected against a passage that writes only `GFS`. ✓
- `match_entities` reaches `tablets` from "What is **a tablet** in Bigtable?", and `raft`
  does not match inside `kraft`. ✓
- Missing-graph path is honest end to end: live `POST /api/run {"technique":"graph-rag"}`
  returns the build instructions with `llm_calls: 0`, `termination_reason: "no_graph"`,
  and no LLM call.

**Phases 0–3 done-when.**
- P0: live home page returns 200 and renders nine cards linking to nine `/learn/<slug>`
  pages, fetched over HTTP from `:8000`. ✓
- P2: playground renders 11 `<option>`s — nine techniques with **4 disabled and labelled
  "— not built yet"**, plus two models. The design decision phase-2 defends is actually
  shipped. ✓
- P3: all nine `/learn/<slug>` return 200, `/learn/bogus-slug` 404s, the Runnable badge is
  derived live (present on `standard-rag` and `graph-rag`, absent on `realm`), and
  "Try it in the playground" links to `/playground?technique=standard-rag`. ✓
- The phase-3 correction block is the best-maintained doc in the repo: its re-measured
  memtable ordering (`bigtable.md#2, cassandra.md#2, bigtable.md#1, spanner.md#1`) matches
  what I measured **exactly**, and so do its KRaft and Cassandra-lineage corrections.
- `standard-rag.mdx:29-30`'s vector-clocks claim: dense #1 for *"how does Dynamo handle
  conflicting writes?"* is `dynamo.md#2`, which does contain "vector clock". ✓
- Chunking: overlap/chunk_size = 200/1200 = **16.7%** ("about 17%"). ✓ Embedding dim 384. ✓
- The four docs-only pages (`agentic`, `interactive`, `feedback`, `realm`) make no
  project-specific measured claim; nothing there can expire.

---

## DEFERRED — needs Ollama, not run

Run these when the local model is free. Do not assume outcomes.

```bash
cd backend

# 0. PREREQUISITE for every graph check: .graph.json does not exist, so 6 tests skip
#    and all of Phase 8's graph-shaped numbers are unverified. ~184s, 43 LLM calls.
.venv/bin/python -m core.graph
.venv/bin/python -m pytest tests/test_graph_rag.py -q   # expect the 6 skips to run

# 1. Phase 1 / 2 done-when: a real grounded generation end to end.
curl -s -X POST localhost:8000/api/run -H 'Content-Type: application/json' \
  -d '{"technique":"standard-rag","query":"How does Dynamo handle conflicting concurrent writes?"}'

# 2. Phase 5 done-when, end to end (BOTH LOCAL = SEQUENTIAL, expect ~25-30s).
curl -s -X POST localhost:8000/api/compare -H 'Content-Type: application/json' \
  -d '{"query":"What is hinted handoff?","a":{"technique":"standard-rag"},"b":{"technique":"fusion-rag"}}'

# 3. Phase 6 measured table (LEARNINGS/phase-6:214-221) — latency 40.2s, 4 calls,
#    3 passes, no_new_evidence, and the step count from finding 2.
curl -s -X POST localhost:8000/api/run -H 'Content-Type: application/json' \
  -d '{"technique":"multi-pass-rag","query":"How does Dynamo achieve high write availability, and how does Raft handle log compaction?"}'

# 4. Phase 7 router claims: 'reversed hostnames' routed `vector` 5/5, 'hinted handoff'
#    routed `hybrid`, and routing = 2.7-5.1% of total latency.
for i in 1 2 3 4 5; do
  curl -s -X POST localhost:8000/api/run -H 'Content-Type: application/json' \
    -d '{"technique":"auto-rag","query":"reversed hostnames"}' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["steps"][0]["detail"], r["steps"][0]["duration_ms"], r["metadata"]["latency_ms"])'
done

# 5. The 55x-190x router figure AND finding 13 (0.07-0.23s vs the docstring's ~0.5s).
.venv/bin/python -c "
import time
from core import llm
from implementations.auto_rag import ROUTER_SYSTEM
for reason in (False, True):
    t=time.perf_counter(); r=llm.generate(ROUTER_SYSTEM,'Query: reversed hostnames',helper=True,reason=reason)
    print(reason, round(time.perf_counter()-t,2),'s', r.output_tokens,'out', repr(r.text[:60]))"

# 6. Phase 8 build numbers: 308 triples / 293 nodes / 308 edges / 184s
#    (printed by step 0), then the graph-vs-standard comparison the page tabulates
#    (Graph top-4 kafka.md#4,bigtable.md#0,gfs.md#1,kafka.md#0; latency 4.4s vs 2.6s;
#    groundedness 0.33 vs 0.5; the 53-node / 35-of-43-chunk / 81% ball).
curl -s -X POST localhost:8000/api/compare -H 'Content-Type: application/json' \
  -d '{"query":"What replaced ZooKeeper in newer Kafka, and what was that protocol designed to be easier than?","a":{"technique":"standard-rag"},"b":{"technique":"graph-rag"}}'
```

Also deferred and **not** Ollama-related: `npm run typecheck` / `npm run build`.
`frontend/node_modules` is not present in this worktree, so TS strict mode and
phase-3's "13 static pages at the time; 14 now" were not exercised. `tsconfig.json`
has `"strict": true` and the `lint` target does run `typecheck`.

**Graph RAG claims that remain unverified** because `.graph.json` does not exist: the
308 triples / 293 nodes / 308 edges / 184s build figures; the 53-node two-hop ball and
its 35-of-43-chunks (81%) coverage; the `reason=False` vs `reason=True` extraction table;
the Graph-vs-Standard retrieval/latency/groundedness table on both
`LEARNINGS/phase-8-graph-rag.md:114-120` and `graph-rag.mdx:96-108`; and the claim that
`raft.md#1` is "never in the top 4". Everything about Graph RAG that could be checked
without the graph — the corpus facts, the bounds, the entity-resolution guards, the
ungrounded-triple filter, the no-graph path — is verified above and correct.

---

## The one thing to fix first

**Finding 1** — the `k` sweep on the Fusion RAG learn page. It is the only wrong number
that a reader is shown and invited to reason from, it sits inside the paragraph whose
lesson is "check the derivation against a sweep", and the correct value (k = 3 is a tie,
resolved by sort order) is a *better* teaching point than the one printed. Fix
`fusion-rag.mdx:108` and `:113` together with `phase-4-fusion-rag.md:90`, and consider a
test over the real index that pins the band, since `test_fusion.py` deliberately uses
synthetic ranks and therefore cannot catch this class of error.

Second: **finding 4**, the compare preset note, because it is the other user-visible
false sentence and it is one line.
