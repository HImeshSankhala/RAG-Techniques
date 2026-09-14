"""Graph RAG — retrieve by following links instead of by measuring similarity.

Every earlier technique scores each passage against the question independently.
Standard RAG does it with cosine distance, Fusion RAG adds BM25 and merges the
ranks, Auto RAG picks which of those to run. All three share one assumption: the
passage that answers the question resembles the question.

A two-hop question breaks that assumption structurally, not by degree. Ask

    "What replaced ZooKeeper in newer Kafka, and what was that protocol
     designed to be easier than?"

and no passage in this corpus resembles the whole question, because no passage
contains the whole answer. `kafka.md#4` says KRaft replaced ZooKeeper and is
replicated by Raft; it never mentions Paxos. `raft.md#1` says Raft was designed
to be easier than Paxos; it never mentions Kafka. Nothing in the corpus mentions
both. A better embedding model does not fix this — the second passage is not
*about* the question, it is about a thing the question's answer names.

The fix is to retrieve along the link rather than across the gap:

    match query entities -> BFS <= 2 hops -> chunks the reached nodes live in

What it costs is paid somewhere else. The graph is extracted at index time by
one LLM call per chunk (43 calls, minutes), so query time makes exactly one LLM
call — the answer — and the walk itself is free. See core/graph.py.

Where it loses: this pipeline does not embed anything. On an ordinary
single-hop, conceptual query — "how does Dynamo handle conflicting writes" —
dense retrieval is scoring meaning while this is following whatever relations a
small model happened to extract, and the extraction is lossy. Graph RAG is a
specialist, and the compare view is where that shows.
"""

import networkx as nx

from core import graph as kg
from core import llm, retrieval, vectorstore
from core.config import settings
from core.pipeline import Chunk, Metadata, RAGPipeline, RAGResult, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness

NO_GRAPH_ANSWER = (
    "The knowledge graph has not been built yet, so there is nothing to traverse. "
    "Run `python -m core.graph` in backend/ (after `make index`) and try again."
)


class GraphRAG(RAGPipeline):
    name = "graph-rag"

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model

        with steps.record("Load knowledge graph") as step:
            stored = kg.load()
            if stored is None:
                step.detail = f"no graph at {kg.GRAPH_FILE.name} — run `python -m core.graph`"
            else:
                step.detail = (
                    f"{stored.graph.number_of_nodes()} entities, "
                    f"{stored.graph.number_of_edges()} relations from {stored.triples} triples "
                    f"· extracted at index time by {stored.llm_calls} LLM calls "
                    f"in {stored.build_seconds:.0f}s"
                    # Stale is surfaced, not hidden: the graph's chunk ids point
                    # into an index that has since moved, so a cited passage may
                    # not say what the edge claims.
                    + (" · STALE: the index changed since extraction" if stored.stale else "")
                )

        if stored is None:
            return RAGResult(
                answer=NO_GRAPH_ANSWER,
                steps=steps.steps,
                metadata=Metadata(
                    model=model,
                    backend=llm.resolve_backend(model),
                    latency_ms=steps.elapsed_ms,
                    termination_reason="no_graph",
                ),
            )

        with steps.record("Match entities in query") as step:
            seeds = kg.match_entities(stored.graph, query)
            step.detail = (
                "found " + ", ".join(f"{stored.graph.nodes[s]['label']!r}" for s in seeds)
                if seeds
                else "no query term matches a graph entity — falling back to dense retrieval"
            )

        with steps.record(f"Traverse graph (<= {kg.MAX_HOPS} hops)") as step:
            walk = kg.traverse(stored.graph, seeds)
            step.detail = _describe_walk(walk, stored.graph)

        # No seed entity means no walk. Falling back to dense retrieval rather
        # than returning nothing, because the alternative is a technique that
        # answers "I found no entities" to any question phrased without a proper
        # noun — and the fallback is visible in the trace and in
        # `termination_reason`, so it cannot be mistaken for a successful walk.
        depths = dict(kg.chunks_for_walk(walk, stored.graph, settings.top_k))

        with steps.record("Collect chunks from reached nodes") as step:
            texts = {c.chunk_id: c for c in vectorstore.all_chunks()}
            if not texts:
                step.detail = "no chunks found — is the index built? (make index)"
                chunks: list[Chunk] = []
            elif depths:
                chunks = [
                    Chunk(
                        text=texts[cid].text,
                        source=texts[cid].source,
                        # Graph distance, not similarity. 1.0 at a matched
                        # entity, 0.5 one hop out, 0.33 two hops out. Deliberately
                        # NOT on the same scale as a cosine score — the UI shows
                        # both, and pretending they are comparable would be a lie.
                        score=round(1.0 / (1 + depths[cid]), 4),
                        chunk_id=cid,
                    )
                    # `chunks_for_walk` returns them ranked, and dict preserves
                    # that order.
                    for cid in depths
                    if cid in texts
                ]
                step.detail = _describe_chunks(chunks, depths)
            else:
                chunks = retrieval.dense(query, settings.top_k)
                step.detail = (
                    f"traversal reached no chunks; dense fallback returned {len(chunks)}"
                )

        if not chunks:
            return RAGResult(
                answer=(
                    "Nothing is indexed yet, so there is no context to answer from. "
                    "Run `make index` and try again."
                ),
                steps=steps.steps,
                metadata=Metadata(
                    model=model,
                    backend=llm.resolve_backend(model),
                    latency_ms=steps.elapsed_ms,
                    retrieval_passes=1,
                    termination_reason="empty_index",
                ),
            )

        # The phase's entire claim, measured on every run rather than asserted
        # once in a doc: which of these chunks would a plain dense top-k have
        # missed? A run where this is empty is a run where the graph bought
        # nothing, and the trace should say so.
        with steps.record("Compare against plain dense retrieval") as step:
            step.detail = _describe_gain(query, chunks)

        with steps.record("Generate answer") as step:
            response = llm.generate(SYSTEM_PROMPT, build_prompt(query, chunks), model=model)
            step.detail = (
                f"{response.model} ({response.backend}): "
                f"{response.input_tokens} in / {response.output_tokens} out"
            )

        cost = (
            llm.estimate_cost_usd(response.input_tokens, response.output_tokens)
            if response.backend == "anthropic"
            else 0.0
        )

        return RAGResult(
            answer=response.text,
            retrieved_chunks=chunks,
            steps=steps.steps,
            metadata=Metadata(
                model=response.model,
                backend=response.backend,
                latency_ms=steps.elapsed_ms,
                # One. The 43 extraction calls were paid at index time and are
                # reported in the trace instead — counting them here would make
                # every query look 44x more expensive than it is, and hiding
                # them entirely would make the technique look free.
                llm_calls=1,
                retrieval_passes=1,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
                # A traversal outcome: why the walk stopped. Not a strategy
                # label — see PLAN.md:145.
                termination_reason=walk.reason,
                groundedness=groundedness(response.text, chunks),
                cost_estimate_usd=round(cost, 6),
            ),
        )


def _describe_walk(walk: kg.Walk, graph: nx.MultiDiGraph) -> str:
    """The hops, in the words the graph actually used.

    Printing the relation labels rather than just node counts is the difference
    between a trace that says "reached 12 nodes" and one that shows
    `'KRaft' -> is replicated by -> 'Raft'` — which is the lesson.
    """
    if not walk.seeds:
        return "no seeds, no walk"
    if not walk.hops:
        return f"seeds are isolated — 0 new nodes in {kg.MAX_HOPS} hops ({walk.reason})"

    parts = []
    for hop in walk.hops:
        shown = ", ".join(
            f"{graph.nodes[node]['label']!r} ({relation})"
            for node, relation in list(zip(hop.nodes, hop.relations))[:4]
        )
        more = f" +{len(hop.nodes) - 4} more" if len(hop.nodes) > 4 else ""
        parts.append(f"hop {hop.depth}: {len(hop.nodes)} new — {shown}{more}")

    return f"{'; '.join(parts)} · stopped: {walk.reason}"


def _describe_chunks(chunks: list[Chunk], depths: dict[str, int]) -> str:
    return f"{len(chunks)} chunks — " + ", ".join(
        f"{c.chunk_id} (hop {depths[c.chunk_id]})" for c in chunks
    )


def _describe_gain(query: str, chunks: list[Chunk]) -> str:
    """What the walk found that a plain dense top-k would not have.

    A second retrieval purely for the trace. It costs one embedding and one HNSW
    lookup — free locally, and no LLM call — and it is what turns "Graph RAG is
    good at multi-hop" from a claim into an observation the user can read.
    """
    dense_ids = {c.chunk_id for c in retrieval.dense(query, settings.top_k)}
    graph_ids = [c.chunk_id for c in chunks]
    new = [cid for cid in graph_ids if cid not in dense_ids]

    if not new:
        return (
            f"dense top-{settings.top_k} would have returned all {len(graph_ids)} of these — "
            "the graph added nothing on this query"
        )
    return (
        f"{len(new)} of {len(graph_ids)} chunks are NOT in dense top-{settings.top_k}: "
        f"{', '.join(new)}"
    )
