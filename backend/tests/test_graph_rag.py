"""Graph RAG against the real corpus and a real graph, with the LLM call stubbed.

Requires the index (`make index`). The traversal tests build a small graph from
hand-written triples rather than the extracted one, so they assert the algorithm
rather than whatever the model happened to extract this week — the extracted
graph changes every time anyone re-runs `python -m core.graph`, and a test that
depends on it is a test that fails for reasons no one changed.

The end-to-end tests do use the real extracted graph when it is present, and skip
when it is not: the graph is gitignored, so a fresh clone has no way to have one.
"""

import json

import pytest

from core import graph as kg
from core import llm, vectorstore
from core.llm import LLMResponse
from implementations.graph_rag import GraphRAG

TWO_HOP_QUERY = (
    "What replaced ZooKeeper in newer Kafka, "
    "and what was that protocol designed to be easier than?"
)


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture
def require_graph() -> kg.StoredGraph:
    stored = kg.load()
    if stored is None:
        pytest.skip("no knowledge graph — run `python -m core.graph` first")
    return stored


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []

    def fake_generate(system: str, user: str, model=None, **kwargs) -> LLMResponse:
        captured.append(user)
        return LLMResponse(
            text="Stubbed answer citing kafka.md.",
            input_tokens=100,
            output_tokens=20,
            model=model or "stub-model",
            backend="ollama",
        )

    monkeypatch.setattr(llm, "generate", fake_generate)
    return captured


# A deliberately tiny, hand-written chain: the shape the technique exists for.
#
#   Kafka --replaces--> KRaft --is replicated by--> Raft --easier than--> Paxos
#
# Each link is asserted by a different chunk, so no single chunk holds the whole
# answer — which is exactly why similarity search cannot assemble it.
CHAIN = [
    kg.Triple("Kafka", "replaces zookeeper with", "KRaft", "kafka.md#4"),
    kg.Triple("KRaft", "is replicated by", "Raft", "kafka.md#4"),
    kg.Triple("Raft", "was designed to be easier than", "Paxos", "raft.md#1"),
    kg.Triple("Paxos", "was introduced by", "Leslie Lamport", "raft.md#0"),
    kg.Triple("Bigtable", "depends on", "Chubby", "bigtable.md#1"),
]


# --- Extraction parsing ----------------------------------------------------


def test_parses_clean_triples() -> None:
    reply = "Kafka | replaces | ZooKeeper\nKRaft | is replicated by | Raft"

    assert kg.parse_triples(reply) == [
        ("Kafka", "replaces", "ZooKeeper"),
        ("KRaft", "is replicated by", "Raft"),
    ]


def test_parses_through_the_decoration_a_small_model_adds() -> None:
    """Numbering, bold, trailing periods and a preamble are formatting mistakes,
    not content mistakes — the parser should see past all of them."""
    reply = (
        "Here are the triples:\n"
        "1. **Kafka** | replaces | ZooKeeper.\n"
        "- `KRaft` | is replicated by | Raft;\n"
    )

    assert kg.parse_triples(reply) == [
        ("Kafka", "replaces", "ZooKeeper"),
        ("KRaft", "is replicated by", "Raft"),
    ]


def test_drops_malformed_and_useless_lines() -> None:
    """Every line here is unusable for a different reason, and none may reach the
    graph: a bad line becomes a bad edge, and a bad edge is a shortcut the walk
    will take."""
    reply = "\n".join(
        [
            "a line with no pipes at all",
            "too | many | pipes | here",
            "only|two",
            "system | stores | data",  # both ends generic
            " | replaces | ZooKeeper",  # empty subject
            "Kafka | replaces | Kafka",  # self-loop
            "Kafka | replaces | ZooKeeper",  # the one good line
            "kafka | REPLACES | zookeeper",  # same triple, different casing
        ]
    )

    assert kg.parse_triples(reply) == [("Kafka", "replaces", "ZooKeeper")]


def test_caps_triples_per_chunk() -> None:
    reply = "\n".join(f"Entity{i} | relates to | Other{i}" for i in range(20))

    assert len(kg.parse_triples(reply)) == kg.MAX_TRIPLES_PER_CHUNK


def test_survives_a_reasoning_model_leaking_its_scratchpad() -> None:
    reply = "<think>Let me look for entities.\nKafka is one.</think>\nKafka | replaces | ZooKeeper"

    assert kg.parse_triples(reply) == [("Kafka", "replaces", "ZooKeeper")]


def test_drops_entities_the_passage_never_mentions() -> None:
    """The regression that made this check exist.

    The first extraction prompt listed example entities, and the model copied
    them into passages that never mention them — mapreduce.md#0 came back
    asserting "MapReduce | was designed to be easier than | Paxos". Those four
    phantom edges wired MapReduce into the Kafka/Raft/Paxos neighbourhood.
    """
    passage = "MapReduce was described by Jeffrey Dean and Sanjay Ghemawat in 2004."
    triples = [
        ("MapReduce", "was described by", "Jeffrey Dean"),
        ("MapReduce", "was designed to be easier than", "Paxos"),
        ("MapReduce", "was designed to be easier than", "KRaft"),
    ]

    assert kg.drop_ungrounded(triples, passage) == [
        ("MapReduce", "was described by", "Jeffrey Dean")
    ]


def test_multi_word_entities_survive_the_grounding_check() -> None:
    """Word-level containment, not whole-string: the passage writes the entity
    inside a sentence, and a stricter check would throw away good triples."""
    passage = "KRaft moves that metadata into an internal Kafka topic replicated by Raft."

    assert kg.drop_ungrounded([("KRaft", "moves metadata into", "internal Kafka topic")], passage)


def test_vendor_prefixes_resolve_to_one_node() -> None:
    """"Apache ZooKeeper" and "ZooKeeper" must be the same entity, or the graph
    holds two nodes with two halves of one neighbourhood and a query that writes
    either name seeds only half the walk."""
    assert kg.normalize("Apache ZooKeeper") == kg.normalize("ZooKeeper") == "zookeeper"
    # But a name that *is* the qualifier plus a real word is not butchered.
    assert kg.normalize("Google File System") == "google file system"


# --- The graph and the walk ------------------------------------------------


def test_nodes_remember_the_chunks_they_came_from() -> None:
    """Without this the graph could state a fact and cite nothing for it."""
    graph = kg.build_graph(CHAIN)

    assert graph.nodes["kraft"]["chunks"] == ["kafka.md#4"]
    assert graph.nodes["raft"]["chunks"] == ["kafka.md#4", "raft.md#1"]


def test_match_entities_finds_query_terms_without_an_llm_call() -> None:
    graph = kg.build_graph(CHAIN)

    # "Kafka" is a node and is named in the question; "Paxos" is a node the
    # question never names, and reaching it is the walk's job, not the matcher's.
    assert kg.match_entities(graph, TWO_HOP_QUERY) == ["kafka"]
    # "raft" must not match inside "KRaft" — a substring match would seed the
    # walk in the wrong place on a query that never named Raft.
    assert kg.match_entities(graph, "Tell me about KRaft") == ["kraft"]
    # Longest first, so a walk starts from the most specific node available.
    assert kg.match_entities(graph, "Leslie Lamport and Paxos") == ["leslie lamport", "paxos"]


def test_the_walk_crosses_two_hops_to_a_chunk_no_single_hop_reaches() -> None:
    """The technique's reason to exist, on a graph small enough to read.

    Seeded only on "Kafka", the walk reaches Raft at hop 2 — and Raft's evidence
    lives in raft.md#1, a chunk that never mentions Kafka and so cannot be found
    by comparing it to the question.
    """
    graph = kg.build_graph(CHAIN)
    walk = kg.traverse(graph, ["kafka"], max_hops=2)

    assert walk.depth_of["kraft"] == 1
    assert walk.depth_of["raft"] == 2
    assert "raft.md#1" in {chunk_id for chunk_id, _ in kg.chunks_for_walk(walk, graph, 4)}


def test_the_walk_follows_edges_in_both_directions() -> None:
    """A chain of facts does not care which way the arrows point: Bigtable
    depends on Chubby, so a question about Chubby should still reach Bigtable."""
    graph = kg.build_graph(CHAIN)
    walk = kg.traverse(graph, ["chubby"], max_hops=1)

    assert walk.depth_of["bigtable"] == 1


def test_the_depth_bound_actually_bounds() -> None:
    """An unbounded walk on a connected graph returns the whole corpus, which is
    the retrieval equivalent of returning nothing."""
    graph = kg.build_graph(CHAIN)

    one_hop = kg.traverse(graph, ["kafka"], max_hops=1)
    two_hop = kg.traverse(graph, ["kafka"], max_hops=2)
    three_hop = kg.traverse(graph, ["kafka"], max_hops=3)

    assert set(one_hop.depth_of) == {"kafka", "kraft"}
    assert "raft" in two_hop.depth_of and "paxos" not in two_hop.depth_of
    assert "paxos" in three_hop.depth_of
    assert max(one_hop.depth_of.values()) == 1
    assert max(two_hop.depth_of.values()) == 2


def test_the_node_budget_bounds_a_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hop limit alone is not enough — one hub with hundreds of neighbours
    blows any chunk budget at depth 1."""
    hub = [kg.Triple("Hub", "links to", f"Leaf{i}", f"doc.md#{i}") for i in range(200)]
    graph = kg.build_graph(hub)

    monkeypatch.setattr(kg, "MAX_NODES", 10)
    walk = kg.traverse(graph, ["hub"], max_hops=2)

    assert len(walk.depth_of) <= 10
    assert walk.reason == "node_budget"


def test_termination_reasons_are_traversal_outcomes() -> None:
    """`termination_reason` is a loop outcome (PLAN.md:145), not a route label."""
    graph = kg.build_graph(CHAIN)

    assert kg.traverse(graph, [], max_hops=2).reason == "no_entities_matched"
    # Kafka's component runs out after Leslie Lamport, so a deep walk exhausts.
    assert kg.traverse(graph, ["kafka"], max_hops=9).reason == "traversal_exhausted"
    assert kg.traverse(graph, ["kafka"], max_hops=1).reason == "max_hops"


def test_a_multi_hop_question_gets_a_chunk_from_each_hop() -> None:
    """Sorting by depth spends every slot on hop 0 — which is keyword retrieval
    with extra steps. Selection round-robins across hops instead."""
    graph = kg.build_graph(CHAIN)
    walk = kg.traverse(graph, ["kafka"], max_hops=3)

    depths = [depth for _, depth in kg.chunks_for_walk(walk, graph, 4)]

    assert len(set(depths)) > 1, "every chunk came from one hop — the walk bought nothing"


# --- Persistence -----------------------------------------------------------


def test_save_and_load_round_trip(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kg, "GRAPH_FILE", tmp_path / ".graph.json")
    kg.save(CHAIN, "some-fingerprint", llm_calls=3, seconds=1.5)

    stored = kg.load()

    assert stored is not None
    assert stored.triples == len(CHAIN)
    assert stored.llm_calls == 3
    assert stored.graph.has_edge("kafka", "kraft")
    # The real corpus fingerprint will not match "some-fingerprint".
    assert stored.stale is True


def test_a_graph_extracted_from_a_different_corpus_is_flagged_stale(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunk ids are positional, so an edited document keeps every id and changes
    every passage. Hashing ids alone would call that corpus unchanged and the
    pipeline would cite passages that do not say what the edge claims."""
    monkeypatch.setattr(kg, "GRAPH_FILE", tmp_path / ".graph.json")
    chunks = vectorstore.all_chunks()

    kg.save(CHAIN, kg.fingerprint(chunks), llm_calls=1, seconds=0.1)
    assert kg.load().stale is False

    edited = [
        vectorstore.IndexedChunk(chunk_id=chunks[0].chunk_id, text="rewritten", source="x.md"),
        *chunks[1:],
    ]
    monkeypatch.setattr(vectorstore, "all_chunks", lambda: edited)

    assert kg.load().stale is True


def test_a_missing_or_corrupt_graph_file_loads_as_none(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh clone has an index but no graph — an ordinary state, not an error."""
    missing = tmp_path / ".graph.json"
    monkeypatch.setattr(kg, "GRAPH_FILE", missing)
    assert kg.load() is None

    missing.write_text("{not json")
    assert kg.load() is None


def test_extraction_is_one_llm_call_per_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cost model of the whole technique: O(chunks) calls, paid once offline."""
    calls: list[str] = []

    def fake_generate(system: str, user: str, model=None, **kwargs) -> LLMResponse:
        calls.append(user)
        # Thinking must stay off — measured at 35s and an empty reply per chunk.
        assert kwargs.get("reason") is False
        assert kwargs.get("helper") is True
        return LLMResponse("Kafka | replaces | ZooKeeper", 10, 5, "stub", "ollama")

    monkeypatch.setattr(llm, "generate", fake_generate)
    chunks = [
        vectorstore.IndexedChunk("a.md#0", "Kafka replaces ZooKeeper.", "a.md"),
        vectorstore.IndexedChunk("a.md#1", "Kafka replaces ZooKeeper again.", "a.md"),
    ]

    triples, made = kg.extract_corpus(chunks)

    assert made == len(chunks) == len(calls)
    assert {t.chunk_id for t in triples} == {"a.md#0", "a.md#1"}


# --- The pipeline ----------------------------------------------------------


def test_run_returns_a_populated_result(stub_llm: list[str], require_graph) -> None:
    result = GraphRAG().run(TWO_HOP_QUERY)

    assert result.answer
    assert result.retrieved_chunks
    assert result.steps


def test_the_trace_shows_entities_hops_and_what_dense_would_miss(
    stub_llm: list[str], require_graph
) -> None:
    """The trace is the teaching surface for this technique — a reader has to be
    able to see which entities seeded the walk, where it went, and what that
    bought over plain retrieval."""
    steps = GraphRAG().run(TWO_HOP_QUERY).steps
    names = [s.name for s in steps]

    assert names == [
        "Load knowledge graph",
        "Match entities in query",
        f"Traverse graph (<= {kg.MAX_HOPS} hops)",
        "Collect chunks from reached nodes",
        "Compare against plain dense retrieval",
        "Generate answer",
    ]
    assert "entities" in steps[0].detail
    assert "found" in steps[1].detail
    assert "hop 1" in steps[2].detail
    assert "hop" in steps[3].detail
    assert "dense top-" in steps[4].detail


def test_one_query_time_llm_call_because_extraction_was_paid_at_index_time(
    stub_llm: list[str], require_graph
) -> None:
    metadata = GraphRAG().run(TWO_HOP_QUERY).metadata

    assert metadata.llm_calls == 1
    assert metadata.retrieval_passes == 1
    assert len(stub_llm) == 1


def test_termination_reason_is_a_traversal_outcome(stub_llm: list[str], require_graph) -> None:
    """A sibling pipeline put a route label in this field; it is a loop outcome."""
    reason = GraphRAG().run(TWO_HOP_QUERY).metadata.termination_reason

    assert reason in {"max_hops", "traversal_exhausted", "node_budget", "no_entities_matched"}


def test_a_query_naming_no_known_entity_falls_back_visibly(
    stub_llm: list[str], require_graph
) -> None:
    """Returning nothing because the question used no proper noun would be a
    worse technique than falling back — but a silent fallback looks identical to
    a successful walk, so it has to show in the metadata."""
    result = GraphRAG().run("what happens when two writes arrive at the same moment")

    assert result.retrieved_chunks
    if result.metadata.termination_reason == "no_entities_matched":
        assert "falling back" in result.steps[1].detail


def test_a_missing_graph_is_reported_not_raised(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stub_llm: list[str]
) -> None:
    """The first thing a new user hits: indexed, but never ran the extraction."""
    monkeypatch.setattr(kg, "GRAPH_FILE", tmp_path / "absent.json")

    result = GraphRAG().run(TWO_HOP_QUERY)

    assert result.retrieved_chunks == []
    assert result.metadata.termination_reason == "no_graph"
    assert result.metadata.backend == "ollama"
    assert "python -m core.graph" in result.answer
    assert stub_llm == [], "must not spend an LLM call when there is nothing to answer from"


def test_an_empty_index_is_reported_not_raised(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stub_llm: list[str]
) -> None:
    """A graph whose chunks are gone from the store must not raise a KeyError."""
    graph_file = tmp_path / ".graph.json"
    monkeypatch.setattr(kg, "GRAPH_FILE", graph_file)
    kg.save(CHAIN, "fp", llm_calls=1, seconds=0.1)
    monkeypatch.setattr(vectorstore, "all_chunks", list)
    monkeypatch.setattr(vectorstore, "query", lambda embedding, top_k: [])

    result = GraphRAG().run(TWO_HOP_QUERY)

    assert result.retrieved_chunks == []
    assert result.metadata.termination_reason == "empty_index"
    assert "make index" in result.answer


def test_the_persisted_graph_is_readable_json(require_graph) -> None:
    """Triples are stored, not a pickled graph object, so "did the model extract
    anything sensible?" is answerable with `less` rather than a Python prompt."""
    stored = json.loads(kg.GRAPH_FILE.read_text())

    assert {"fingerprint", "chunks", "llm_calls", "triples"} <= set(stored)
    assert set(stored["triples"][0]) == {"subject", "relation", "object", "chunk_id"}


def test_registered_and_runnable() -> None:
    from implementations.registry import get_pipeline

    assert get_pipeline("graph-rag") is not None
    assert get_pipeline("graph-rag").name == GraphRAG.name
