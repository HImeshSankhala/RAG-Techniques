"""The knowledge graph: extract it at index time, walk it at query time.

Every technique so far retrieves by *comparing the query to a passage* — cosine
similarity, BM25 term statistics, or both fused. That comparison is local: a
passage either resembles the question or it does not. It cannot follow a link,
so a question whose answer is assembled from two passages that do not resemble
each other is out of reach no matter how good the scorer is.

This module builds the structure that makes those links traversable. An LLM reads
each indexed chunk once, offline, and emits `(subject, relation, object)` triples;
the triples become a graph whose nodes remember which chunks they came from.
Retrieval is then a bounded breadth-first walk: match entities in the query,
expand a hop or two, and gather the chunks that the reached nodes live in.

Two costs move around here, and moving them is the whole design:

- Extraction is O(chunks) LLM calls and takes minutes. It is paid ONCE, at index
  time, and cached to disk — see `save`/`load`. Paying it per query would make
  the technique unusable.
- The walk is graph traversal, microseconds, and makes no LLM call at all.

Nothing here narrates. `implementations/graph_rag.py` owns the steps trace, the
same way `core/retrieval.py` owns the retrievers and the pipelines above it own
the story.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

import networkx as nx

from core import llm, vectorstore
from core.config import BACKEND_ROOT

# Cached graph, beside .chroma and gitignored for the same reason: it is derived
# from the corpus, it is machine-written, and it is rebuilt by a command rather
# than reviewed in a diff.
#
# Defined here rather than in core/config.py, where chroma_dir and usage_file
# live, only because nothing else in the engine needs to know it exists. If a
# second consumer ever appears it should move there.
GRAPH_FILE = BACKEND_ROOT / ".graph.json"

# --- Extraction ------------------------------------------------------------

# `reason=False`, and this is measured rather than assumed. On qwen3:8b, three
# chunks extracted both ways:
#
#   chunk        reason=False            reason=True
#   kafka.md#4   8.3s, 8 usable triples  34.3s, EMPTY reply
#   raft.md#1    4.0s, 8 usable triples  35.9s, EMPTY reply
#   dynamo.md#1  3.2s, 8 usable triples  35.0s, EMPTY reply
#
# The empty replies are the failure documented in core/config.py: Ollama draws
# thinking tokens from the same `num_predict` budget as the reply, and a chunk
# this size fills all 1024 with reasoning before a single triple is emitted.
# Even if it did not, 35s x 43 chunks is 25 minutes against 3.5.
#
# The deeper reason it does not need thinking: extraction is not a judgement
# call. Multi-Pass's critique has to decide whether something is *missing*, which
# is an inference about absent evidence; this call only has to copy relations the
# passage already states. That is the same shape as answering from context, which
# core/llm.py already runs with thinking off.
#
# The prompt names NO example entities, and that is a scar rather than a style
# choice. The first version listed "Kafka, KRaft, Raft, Paxos, ZooKeeper,
# Bigtable" to illustrate what an entity is, and the model copied those names
# into chunks that never mention them — mapreduce.md#0 came back with "MapReduce
# | was designed to be easier than | Paxos", three more like it, and those
# phantom edges wired MapReduce into the middle of the Kafka/Raft/Paxos
# neighbourhood. A hallucinated edge is worse than a missing one: it is a
# shortcut the walk will happily take. See `drop_ungrounded`, which enforces the
# same rule in code rather than trusting the prompt.
EXTRACTION_SYSTEM = """You extract entity-relation triples from a technical passage.

Output one triple per line, in exactly this form:

subject | relation | object

Rules:
- subject and object are NAMED things the passage itself mentions: systems,
  protocols, papers, components, companies, people. Copy the name as written.
- NEVER introduce a name that does not appear in the passage. If you are unsure
  whether the passage contains a name, leave that triple out.
- relation is 2-5 lowercase words describing the link, such as: replaces,
  depends on, is used by, is replicated by, was designed to be easier than.
- Only state relations the passage actually asserts. Add no outside knowledge.
- Never use a generic word as subject or object: system, data, record, user,
  message, node, key, value, client, server, write, read, request.
- At most 8 lines. No preamble, no numbering, no commentary, no blank lines."""

# Bounds one chunk's contribution so a rambling reply cannot dominate the graph.
MAX_TRIPLES_PER_CHUNK = 8

# Words the model reaches for when it runs out of real entities. Left in, they
# become hubs that connect everything to everything and make a 2-hop walk return
# the whole corpus — which is the graph equivalent of retrieving nothing.
GENERIC_ENTITIES = frozenset(
    {
        "system", "systems", "data", "record", "records", "user", "users",
        "message", "messages", "node", "nodes", "key", "keys", "value", "values",
        "client", "clients", "server", "servers", "write", "writes", "read",
        "reads", "request", "requests", "it", "this", "that", "they", "the paper",
        "paper", "papers", "the passage", "passage", "one value", "a value",
    }
)

_MARKUP = re.compile(r"[*`_#]|^\s*[-*\d.)]+\s+")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


# Organisational qualifiers the corpus uses inconsistently. Left in, "Apache
# ZooKeeper" and "ZooKeeper" are two nodes with two halves of the same
# neighbourhood, and a query that says "ZooKeeper" seeds neither reliably. This
# is entity resolution at its cheapest — one rule, no alias table — and it is
# genuinely lossy: it will also merge two different things that share a name
# under different vendors. At this corpus size that has not happened.
_QUALIFIERS = frozenset({"apache", "google", "amazon", "the"})


def normalize(entity: str) -> str:
    """Canonical node id for an entity mention.

    Node identity has to survive "KRaft", "kraft" and "KRaft " arriving from
    three different chunks, or the same thing becomes three disconnected nodes
    and the walk never crosses between them.

    It also has to survive a vendor prefix. Measured on this corpus: without the
    prefix strip the graph holds `apache zookeeper`, `apache kafka` (beside a
    separate `kafka`) and `apache cassandra`, and the two-hop demo query — which
    writes "ZooKeeper", not "Apache ZooKeeper" — matches neither.
    """
    words = entity.lower().split()
    # Strip a qualifier only when exactly one word is left. "Apache ZooKeeper"
    # is a product name wearing a vendor prefix; "Google File System" is a name
    # whose first word is load-bearing, and stripping it yields the generic
    # "file system" — which would then merge with anything else called that.
    if len(words) == 2 and words[0] in _QUALIFIERS:
        return words[1]
    return " ".join(words)


def parse_triples(reply: str) -> list[tuple[str, str, str]]:
    """Read `subject | relation | object` lines out of a small model's free text.

    Written to be lenient about presentation and strict about shape. The model
    decorates ("**Kafka** | replaces | ZooKeeper"), numbers its lines, adds a
    "Here are the triples:" preamble, and occasionally emits a line with four
    pipes or one. Every one of those is a formatting mistake rather than a
    content mistake, so the parser strips what it can and drops only the lines
    it genuinely cannot read.
    """
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_line in _THINK.sub("", reply).splitlines():
        line = _MARKUP.sub("", raw_line).strip()
        parts = [part.strip(" .;") for part in line.split("|")]
        if len(parts) != 3:
            continue

        subject, relation, obj = (" ".join(p.split()) for p in parts)
        if not (subject and relation and obj):
            continue
        if len(subject) < 2 or len(obj) < 2:
            continue
        if normalize(subject) in GENERIC_ENTITIES or normalize(obj) in GENERIC_ENTITIES:
            continue
        if normalize(subject) == normalize(obj):
            continue

        key = (normalize(subject), normalize(relation), normalize(obj))
        if key in seen:
            continue
        seen.add(key)

        triples.append((subject, relation.lower(), obj))
        if len(triples) == MAX_TRIPLES_PER_CHUNK:
            break

    return triples


def drop_ungrounded(
    triples: list[tuple[str, str, str]], passage: str
) -> list[tuple[str, str, str]]:
    """Keep only triples whose entities are actually words in the passage.

    The extraction prompt already says "never introduce a name the passage does
    not contain". This enforces it, because a prompt rule is a request and this
    is a check — and the difference showed up immediately: asked to extract from
    mapreduce.md#0, the model returned "MapReduce | was designed to be easier
    than | Paxos", plus the same relation to KRaft, Raft and ZooKeeper. None of
    those four words appear in that passage. They appeared in the prompt's list
    of example entities.

    A hallucinated edge is strictly worse than a missing one here. A missing edge
    costs recall; a false edge is a shortcut the traversal will take, and those
    four wired MapReduce into the middle of the Kafka/Raft/Paxos neighbourhood —
    exactly the region the two-hop demo query walks through.

    The rule is word-level containment rather than whole-string, so "internal
    Kafka topic" survives even though the passage writes it inside a longer
    sentence. Short words are ignored: they match everywhere and would let a
    junk entity through on the strength of "of".

    The cost of being strict is a real abbreviation loss — a passage that says
    only "GFS" rejects the entity "Google File System". That is the right side to
    err on: the corpus is the ground truth and the model is not.
    """
    lowered = passage.lower()

    def grounded(entity: str) -> bool:
        words = [w for w in re.findall(r"[\w-]+", entity.lower()) if len(w) >= 3]
        # An entity of only short words ("ISR", "R", "W") can't be checked this
        # way, so fall back to whole-string containment rather than guessing.
        if not words:
            return entity.lower() in lowered
        return all(re.search(rf"(?<![\w-]){re.escape(w)}", lowered) for w in words)

    return [t for t in triples if grounded(t[0]) and grounded(t[2])]


# --- Building --------------------------------------------------------------


@dataclass(frozen=True)
class Triple:
    """One extracted relation and the chunk that asserted it.

    `chunk_id` is the load-bearing field. Without it the graph would answer
    "Kafka is replicated by Raft" from its own structure and have no passage to
    show for it — a RAG pipeline that cannot cite is not a RAG pipeline.
    """

    subject: str
    relation: str
    obj: str
    chunk_id: str


def build_graph(triples: list[Triple]) -> nx.MultiDiGraph:
    """Assemble triples into a directed multigraph.

    MultiDiGraph rather than Graph for two reasons. Directed, because "KRaft
    replaces ZooKeeper" and "ZooKeeper replaces KRaft" are not the same claim and
    the trace should be able to print the relation as written. Multi, because two
    chunks can assert two different relations between the same pair, and
    collapsing them would silently drop one chunk's evidence.

    The walk itself ignores direction (see `traverse`) — a two-hop chain often
    runs with the arrows and then against them.
    """
    graph = nx.MultiDiGraph()

    for triple in triples:
        for name in (triple.subject, triple.obj):
            node = normalize(name)
            if node not in graph:
                # `label` keeps the first-seen casing for display; `chunks` is
                # the inverted index the walk actually retrieves through.
                graph.add_node(node, label=name, chunks=[])
            if triple.chunk_id not in graph.nodes[node]["chunks"]:
                graph.nodes[node]["chunks"].append(triple.chunk_id)

        graph.add_edge(
            normalize(triple.subject),
            normalize(triple.obj),
            relation=triple.relation,
            chunk_id=triple.chunk_id,
        )

    return graph


def extract_corpus(
    chunks: list[vectorstore.IndexedChunk], model: str | None = None
) -> tuple[list[Triple], int]:
    """One LLM call per chunk. Returns every triple and the number of calls made.

    Sequential on purpose. The local model is ~5GB and concurrent generations
    thrash memory on a 16GB machine, so parallelism here would make the build
    slower, not faster. This runs offline; latency is not the constraint.
    """
    triples: list[Triple] = []
    calls = 0

    for chunk in chunks:
        response = llm.generate(
            EXTRACTION_SYSTEM,
            f"Passage:\n\n{chunk.text}",
            model=model,
            helper=True,
            reason=False,
        )
        calls += 1
        triples.extend(
            Triple(subject=s, relation=r, obj=o, chunk_id=chunk.chunk_id)
            for s, r, o in drop_ungrounded(parse_triples(response.text), chunk.text)
        )

    return triples, calls


# --- Persistence -----------------------------------------------------------


def fingerprint(chunks: list[vectorstore.IndexedChunk]) -> str:
    """Content hash of the indexed corpus.

    A graph extracted from a previous corpus is worse than no graph: it points at
    chunk ids that may no longer exist, or may now hold different text, so the
    pipeline would cite passages that do not say what the edge claims. Chunk ids
    alone are not enough — they are positional, so an edited document can keep
    every id and change every passage. Hash the text too.
    """
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        digest.update(chunk.chunk_id.encode())
        digest.update(b"\0")
        digest.update(chunk.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class StoredGraph:
    """A loaded graph plus what is worth knowing about how it was made."""

    graph: nx.MultiDiGraph
    stale: bool  # the index has changed since this graph was extracted
    chunks: int
    llm_calls: int
    build_seconds: float
    triples: int


def save(triples: list[Triple], corpus_fingerprint: str, llm_calls: int, seconds: float) -> None:
    """Write the graph to `backend/.graph.json` (gitignored, beside .chroma).

    Triples are persisted, not the NetworkX object. Rebuilding the graph from
    them takes milliseconds, and a flat list of readable lines is inspectable
    with `less` — which matters a lot when the question is "did the model extract
    anything sensible?". Pickling a graph object would answer that question only
    through a Python prompt.
    """
    GRAPH_FILE.write_text(
        json.dumps(
            {
                "fingerprint": corpus_fingerprint,
                "chunks": len({t.chunk_id for t in triples}),
                "llm_calls": llm_calls,
                "build_seconds": round(seconds, 1),
                "triples": [
                    {
                        "subject": t.subject,
                        "relation": t.relation,
                        "object": t.obj,
                        "chunk_id": t.chunk_id,
                    }
                    for t in triples
                ],
            },
            indent=2,
        )
    )


def load() -> StoredGraph | None:
    """The cached graph, or None if it was never built or the file is unreadable.

    A missing graph is an ordinary state, not an error — a fresh clone has run
    `make index` but not the extraction — so this returns None and lets the
    pipeline say so in its trace. A *stale* graph is returned rather than
    withheld, flagged: it is usually still mostly right, and refusing to answer
    would be a worse experience than answering with the trace saying the index
    moved.
    """
    try:
        stored = json.loads(GRAPH_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    triples = [
        Triple(
            subject=t["subject"], relation=t["relation"], obj=t["object"], chunk_id=t["chunk_id"]
        )
        for t in stored.get("triples", [])
    ]

    return StoredGraph(
        graph=build_graph(triples),
        stale=stored.get("fingerprint") != fingerprint(vectorstore.all_chunks()),
        chunks=stored.get("chunks", 0),
        llm_calls=stored.get("llm_calls", 0),
        build_seconds=stored.get("build_seconds", 0.0),
        triples=len(triples),
    )


# --- Query-time walk -------------------------------------------------------

# How far the walk may travel from a matched entity. Two is the smallest bound
# that answers a two-hop question, and every extra hop is roughly a branching
# factor more nodes — an unbounded walk on a connected graph returns the whole
# corpus, which is the retrieval equivalent of returning nothing.
MAX_HOPS = 2

# A second, independent bound. The hop limit alone is not enough: one hub node
# with 200 neighbours blows past any sane chunk budget at depth 1. Belt and
# braces, the same way an agent loop caps both iterations and tokens.
MAX_NODES = 80


@dataclass(frozen=True)
class Hop:
    """The nodes first reached at one depth, and a sample of how they were reached."""

    depth: int
    nodes: list[str]
    relations: list[str]


@dataclass
class Walk:
    """The result of a bounded breadth-first traversal."""

    seeds: list[str] = field(default_factory=list)
    hops: list[Hop] = field(default_factory=list)
    depth_of: dict[str, int] = field(default_factory=dict)
    # Chunks that asserted an edge the walk actually crossed, and at which hop.
    # Kept separate from node membership because they are stronger evidence: a
    # chunk that merely mentions a reached entity is on-topic, but a chunk that
    # states the link is the citation for the step being taken.
    edge_chunks: list[tuple[int, str]] = field(default_factory=list)
    # Why the walk stopped — a loop outcome, which is what Metadata's
    # `termination_reason` is for (PLAN.md:145).
    reason: str = "traversal_exhausted"


def match_entities(graph: nx.MultiDiGraph, query: str) -> list[str]:
    """Graph nodes whose name appears in the query. No LLM call.

    A second model call to "extract entities from the question" was the obvious
    design and is the wrong one: the only entities that can seed a walk are the
    ones the graph already has, so the candidate set is small, known, and
    matchable with string containment. Asking a model to invent names that then
    have to be matched against this same list adds a second failure mode and a
    second second of latency to buy nothing.

    Matching tolerates a plural, because "What is a tablet in Bigtable?" must
    reach the node the extractor called `tablets`. Measured before the tolerance
    existed, that query seeded on `bigtable` alone, never retrieved the passage
    that defines a tablet, and the pipeline answered "the context passages do
    not mention what a tablet is". No stemmer: only a trailing "s" on a node of
    four characters or more, which cannot fire unless the query itself contains
    the truncated word.

    Longest first, so the walk starts from the most specific node available.
    """
    lowered = f" {' '.join(query.lower().split())} "

    def mentioned(name: str) -> bool:
        # Word-boundary-ish: pad both sides so "raft" does not match "kraft".
        return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", lowered) is not None

    matched = [
        node
        for node in graph.nodes
        if len(node) >= 3
        and (mentioned(node) or (len(node) >= 5 and node.endswith("s") and mentioned(node[:-1])))
    ]
    return sorted(matched, key=lambda n: (-len(n), n))


def _neighbours(graph: nx.MultiDiGraph, node: str) -> set[str]:
    """Both directions. A chain of facts does not care which way the arrows point."""
    return set(graph.successors(node)) | set(graph.predecessors(node))


def traverse(graph: nx.MultiDiGraph, seeds: list[str], max_hops: int = MAX_HOPS) -> Walk:
    """Breadth-first from every seed at once, bounded by hops and by node count.

    Multi-source BFS rather than one walk per seed: the seeds are all in the same
    question, so a node two hops from one seed and two from another is not more
    relevant than either — what matters is the shortest distance to *any* of
    them, which is exactly what a shared frontier computes. O(V + E) in the
    visited subgraph, and the bounds are what keep that subgraph small.
    """
    walk = Walk(seeds=list(seeds))
    if not seeds:
        walk.reason = "no_entities_matched"
        return walk

    walk.depth_of = {seed: 0 for seed in seeds}
    frontier = list(seeds)

    for depth in range(1, max_hops + 1):
        if not frontier:
            walk.reason = "traversal_exhausted"
            return walk

        reached: list[str] = []
        relations: list[str] = []
        budget_hit = False

        for node in frontier:
            for neighbour in sorted(_neighbours(graph, node)):
                if neighbour in walk.depth_of:
                    continue
                if len(walk.depth_of) >= MAX_NODES:
                    budget_hit = True
                    break
                walk.depth_of[neighbour] = depth
                reached.append(neighbour)
                relation, chunk_ids = _edge_between(graph, node, neighbour)
                relations.append(relation)
                walk.edge_chunks.extend((depth, chunk_id) for chunk_id in chunk_ids)
            if budget_hit:
                break

        if reached:
            walk.hops.append(Hop(depth=depth, nodes=reached, relations=relations))

        if budget_hit:
            walk.reason = "node_budget"
            return walk

        frontier = reached

    walk.reason = "max_hops" if frontier else "traversal_exhausted"
    return walk


def _edge_between(graph: nx.MultiDiGraph, a: str, b: str) -> tuple[str, list[str]]:
    """A readable label for the edge just crossed, plus every chunk asserting it.

    All parallel edges are credited, not just the first. Two chunks stating the
    same link are two independent citations for it, and dropping one would hide
    the passage that happens to phrase it best — which on this corpus is exactly
    what happens between `chubby.md` and `raft.md` on the Raft/Paxos link.
    """
    label = "-> linked"
    chunk_ids: list[str] = []

    for source, target in ((a, b), (b, a)):
        if not graph.has_edge(source, target):
            continue
        for data in graph[source][target].values():
            if not chunk_ids:
                arrow = "->" if (source, target) == (a, b) else "<-"
                label = f"{arrow} {data['relation']}"
            chunk_ids.append(data["chunk_id"])

    return label, chunk_ids


def chunks_for_walk(walk: Walk, graph: nx.MultiDiGraph, top_k: int) -> list[tuple[str, int]]:
    """Chunk ids the walk reached, best first, as (chunk_id, best hop depth).

    A chunk's depth is the shallowest walked node living in it, and within one
    depth chunks are ordered by how many walked nodes they host — a passage
    holding three of them is more likely to be the one that joins them than a
    passage with a single peripheral mention.

    Two corrections to the obvious ranking live here, and both were forced by
    measurement rather than taste.

    **Selection across depths is round-robin, not sorted.** Sorting by depth was
    the first version, and on the demo query it returned four hop-0 chunks: with
    top_k=4, every slot went to a passage that merely contains the words in the
    question. That is keyword retrieval with extra steps — the walk found Raft
    and then the ranker threw it away. A multi-hop question needs one passage
    *per hop*, not the k best overall, so depth is a bucket to sample from
    rather than a score to sort on. A single-hop question is unaffected: its
    deeper buckets are thin and the loop keeps drawing from the shallow one.

    **Within a bucket, order by how many of the query's own entities the chunk
    contains**, then by traversed edges asserted, then by raw node membership.
    Every later key is a volume measure, and volume alone rewards sitting in a
    dense neighbourhood rather than saying anything about the question —
    measured, it put `kafka.md#0` (eight generic Kafka facts) ahead of
    `kafka.md#4`, the one passage holding both entities the question names. Seed
    coverage is the same principle BM25 applies to terms, applied to resolved
    entities instead; edges rank next because a chunk that asserted a crossed
    link is the citation for that step, where a membership chunk is just nearby.
    """
    best: dict[str, int] = {}
    hits: dict[str, int] = {}
    edges: dict[str, int] = {}
    seed_coverage: dict[str, set[str]] = {}
    seeds = set(walk.seeds)

    for node, depth in walk.depth_of.items():
        for chunk_id in graph.nodes[node].get("chunks", ()):
            best[chunk_id] = min(best.get(chunk_id, depth), depth)
            hits[chunk_id] = hits.get(chunk_id, 0) + 1
            if node in seeds:
                seed_coverage.setdefault(chunk_id, set()).add(node)

    for depth, chunk_id in walk.edge_chunks:
        best[chunk_id] = min(best.get(chunk_id, depth), depth)
        edges[chunk_id] = edges.get(chunk_id, 0) + 1

    buckets: dict[int, list[str]] = {}
    for chunk_id in sorted(
        best,
        key=lambda cid: (
            -len(seed_coverage.get(cid, ())),
            -edges.get(cid, 0),
            -hits.get(cid, 0),
            cid,
        ),
    ):
        buckets.setdefault(best[chunk_id], []).append(chunk_id)

    picked: list[tuple[str, int]] = []
    while len(picked) < top_k and any(buckets.values()):
        for depth in sorted(buckets):
            if buckets[depth] and len(picked) < top_k:
                picked.append((buckets[depth].pop(0), depth))

    return picked


# --- CLI -------------------------------------------------------------------


def main() -> None:
    """`python -m core.graph` — rebuild the graph from the current index.

    Deliberately NOT wired into `core/index.py`. Indexing takes five seconds and
    everyone needs it; extraction takes minutes and only Graph RAG needs it.
    Folding the slow, optional step into the fast, mandatory one would tax every
    other technique's setup for this one's benefit.
    """
    chunks = vectorstore.all_chunks()
    if not chunks:
        raise SystemExit("Nothing indexed. Run `make index` first.")

    print(f"Extracting entities and relations from {len(chunks)} chunks (one LLM call each)...")
    started = time.perf_counter()
    triples, calls = extract_corpus(chunks)
    seconds = time.perf_counter() - started

    save(triples, fingerprint(chunks), calls, seconds)
    graph = build_graph(triples)
    print(
        f"{len(triples)} triples -> {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges in {seconds:.0f}s ({calls} LLM calls). "
        f"Wrote {GRAPH_FILE}."
    )


if __name__ == "__main__":
    main()
