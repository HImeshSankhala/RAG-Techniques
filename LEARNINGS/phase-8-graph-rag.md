# Phase 8 — Graph RAG

Retrieve by following links instead of by measuring similarity. The first technique
here that builds a second index at ingest time, and the first one that **loses to
the Phase 1 baseline on the query it was built for**. That negative result is the
main thing worth taking from this phase, so it is stated up front rather than
buried.

## The problem

Every technique so far scores each passage against the question *independently*.
Standard RAG uses cosine distance, Fusion RAG adds BM25 and merges ranks, Auto RAG
picks which of those to run. All three share one assumption: the passage that
answers the question resembles the question.

A two-hop question breaks that assumption structurally, not by degree:

> What replaced ZooKeeper in newer Kafka, and what was that protocol designed to be
> easier than?

Verified against the 43-chunk corpus:

- `kafka.md#4` is the **only** chunk containing "KRaft". It says KRaft replaced
  ZooKeeper and is replicated by Raft. It contains neither "Paxos" nor
  "understandable".
- The second hop lives in `raft.md#1` and `chubby.md#3`: Raft was designed as an
  understandable alternative to Paxos. Neither chunk mentions Kafka.
- **No chunk in the corpus contains both "Paxos" and "Kafka".**

So no passage resembles the whole question, because no passage holds the whole
answer. A better embedding model does not fix this. The second passage is not
*about* the question — it is about a thing the question's answer happens to name.

## The approach

    index time:  one LLM call per chunk -> (subject, relation, object) triples
                 -> NetworkX MultiDiGraph, nodes remember their chunk ids
                 -> cached to backend/.graph.json (gitignored)

    query time:  match entities in the query -> BFS <= 2 hops
                 -> collect the chunks the reached nodes live in -> answer

The cost asymmetry is the whole design. Extraction is O(chunks) LLM calls — 43
calls, 184 seconds, measured — and is paid **once, offline**. Query time makes
exactly one LLM call (the answer) and the walk itself costs under a millisecond.
Doing the extraction per query would be an unusable technique; caching it makes the
graph a second index alongside Chroma, built by `python -m core.graph`, invalidated
by a SHA-256 fingerprint over every chunk id *and* its text.

Why hash the text and not just the ids? Chunk ids are positional (`kafka.md#4`), so
editing a document keeps every id and changes every passage. A graph validated on
ids alone would cite passages that no longer say what its edges claim.

### Why `reason=False` on the extraction call

Measured on three chunks, both ways:

| chunk | `reason=False` | `reason=True` |
|---|---|---|
| `kafka.md#4` | 8.3 s, 8 usable triples | 34.3 s, **empty reply** |
| `raft.md#1` | 4.0 s, 8 usable triples | 35.9 s, **empty reply** |
| `dynamo.md#1` | 3.2 s, 8 usable triples | 35.0 s, **empty reply** |

The empty replies are the failure already documented in `core/config.py`: Ollama
draws thinking tokens from the same `num_predict` budget as the reply, and a chunk
this size fills all 1024 with reasoning before emitting a single triple. Even if it
did not, 35 s × 43 chunks is 25 minutes against 3.5.

The deeper reason it needs no thinking: extraction is not a judgement call. Phase
6's critique has to decide whether something is *missing*, which is an inference
about absent evidence. This call only copies relations the passage already states —
the same shape as answering from context, which already runs with thinking off.

## The algorithms and their complexity

**Construction.** One LLM call per chunk, then a linear pass inserting triples.
Time O(C) model calls + O(T) insertions for C chunks and T triples; space O(V + E).
Measured: C = 43, T = 308, V = 293, E = 308, 184 s wall clock. The LLM calls
dominate by five orders of magnitude — graph building itself is noise.

**Traversal.** Multi-source breadth-first search from every matched entity at once,
over the undirected view of a directed graph (a chain of facts does not care which
way the arrows point). Multi-source rather than one walk per seed because the seeds
are all in the same question: what matters is the shortest distance to *any* of
them, which a shared frontier computes for free. Time and space O(V' + E') in the
visited subgraph.

**The bounds are the interesting part.** Two independent caps, for the same reason
an agent loop caps both iterations and tokens:

- `MAX_HOPS = 2` — every extra hop is a branching factor more nodes.
- `MAX_NODES = 80` — the hop limit alone is useless against a hub. One node with
  200 neighbours blows any chunk budget at depth 1.

An unbounded walk on a connected graph returns the whole corpus, which is the
retrieval equivalent of returning nothing. `termination_reason` reports which bound
fired — `max_hops`, `node_budget`, `traversal_exhausted`, or `no_entities_matched`
— and those are loop outcomes, which is what that field is for.

**Selection**, which turned out to matter more than traversal. Chunks are bucketed
by the shallowest walked node they contain and drawn **round-robin across buckets**,
not sorted by depth. Sorting was the first version and on the demo query it spent
all four slots on hop-0 chunks — passages that merely contain the words in the
question, i.e. keyword retrieval with extra steps. A multi-hop question needs one
passage *per hop*, so depth is a bucket to sample from, not a score to sort on.
Within a bucket: seed coverage first (how many of the question's own entities the
chunk holds — BM25's principle applied to resolved entities), then traversed edges
asserted, then raw membership.

## The measured result: it does not work on this corpus

Run live, `qwen3:8b`, same query, same model:

| | Standard RAG | Graph RAG |
|---|---|---|
| chunks | `kafka.md#4`, `chubby.md#4`, `kafka.md#1`, `kafka.md#0` | `kafka.md#4`, `bigtable.md#0`, `gfs.md#1`, `kafka.md#0` |
| latency | 2.6 s | 4.4 s |
| groundedness | 0.5 | 0.33 |
| hop 1 (KRaft) | correct | correct |
| hop 2 (Paxos) | **wrong** — says "easier than ZooKeeper" | **wrong** — says "easier than ZooKeeper" |

Graph RAG replaced two on-topic chunks with two irrelevant ones, took 1.7× the
latency, and got the same answer wrong in the same way. On single-hop queries it is
roughly level (Dynamo, Spanner/TrueTime answered correctly) but never better, and
its stray chunks cost it a mis-citation on the GFS query — it attributed a GFS fact
to `dynamo.md`, because `cassandra.md` was sitting in the context for no reason.

### Why, precisely

The diagnosis is not "the walk missed the evidence". Instrumented, on every
configuration tried (2 or 3 hops, with and without the bad edges below):

    targets in the 2-hop ball : chubby.md#3, kafka.md#4, raft.md#0, raft.md#1, raft.md#2
    targets in the top 4      : none
    ball size                 : 53 nodes -> 35 of 43 chunks

**The 2-hop neighbourhood covers 81% of the corpus.** Recall is fine; the graph
simply is not a *filter*. Nine documents produce 293 entities that are nearly
connected, so "within two hops of Kafka" excludes almost nothing, and selection
inside that ball has to do all the work — with no similarity signal, which is
exactly the signal cosine distance has and this does not.

Graph RAG's premise requires a graph where two hops is a *small* fraction of the
corpus. That is true of a 100k-document corpus with a sparse entity graph. It is
false here, and no amount of ranking fixes a filter that filters nothing. Four
different ranking rules were tried and measured (depth-sorted, round-robin,
edge-depth-weighted, seed-coverage); the best of them is what shipped, and it still
loses.

**Flagged, not implemented:** the standard fix is to invert the composition — seed
the walk from *dense retrieval's* top-k rather than from query entities, and use
the graph only to add linked evidence dense missed. That keeps cosine's ranking
(which works) and adds multi-hop recall (which cosine lacks). On this query it
would reach `raft.md#1` in one hop from the entities in `kafka.md#4`. It is a
different technique from the one PLAN.md specifies, so it is a decision for the
repo owner rather than a silent substitution.

## Failure mode: the prompt's own examples became facts

The first extraction prompt illustrated what an entity is with a list:

> Examples: Kafka, KRaft, Raft, Paxos, ZooKeeper, Bigtable, ISR.

The model copied those names into chunks that never mention them. From
`mapreduce.md#0` — a passage about Jeffrey Dean, Sanjay Ghemawat and the 2004 OSDI
paper — it returned:

    MapReduce | was designed to be easier than | Paxos
    MapReduce | was designed to be easier than | KRaft
    MapReduce | was designed to be easier than | Raft
    MapReduce | was designed to be easier than | ZooKeeper

and from `bigtable.md#1`, `Bigtable | is replicated by | KRaft`. Every one is
fabricated, and they wired MapReduce and Bigtable into the middle of the
Kafka/Raft/Paxos neighbourhood — the exact region the demo query walks through.

**A hallucinated edge is strictly worse than a missing one.** A missing edge costs
recall. A false edge is a shortcut the traversal will take, and it is invisible:
the trace shows a confident hop along a relation that reads perfectly well.

Two fixes, and the order matters. The prompt now names no example entities and says
"never introduce a name that does not appear in the passage" — but a prompt rule is
a *request*. `drop_ungrounded` is the *check*: every entity's words of three or more
characters must actually occur in the source chunk, or the triple is discarded.
After both changes, re-extracting the corpus produced 308 triples where the first
run produced 329, and all five fabrications above were gone — the two fixes landed
together, so the split between them is not separately measured. Being strict has a
real price — a passage that writes only "GFS" will reject the entity "Google File
System" — and that is the right side to err on, because the corpus is ground truth
and the model is not.

Two related entity-resolution bugs surfaced the same way, both found by measurement
rather than reasoning:

- `Apache ZooKeeper` and `ZooKeeper` were separate nodes holding two halves of one
  neighbourhood, and the demo query (which writes "ZooKeeper") seeded neither.
  Fixed by stripping a vendor qualifier — but only when a single word remains,
  because the first version turned `Google File System` into the generic
  `file system`. A unit test caught that one before it reached the graph.
- "What is **a tablet** in Bigtable?" never matched the node `tablets`, so Graph RAG
  answered "the context passages do not mention what a tablet is" on a plain
  single-hop question. Fixed with a trailing-`s` tolerance in entity matching — no
  stemmer, which would turn `paxos` into `paxo`.

The through-line: **a graph is only as good as its extractor, and the extractor is a
small model being asked to be precise.** Every guard here exists because something
measurable went wrong without it.
