"""POST /api/compare — the endpoint the engine contract was designed for.

The LLM is stubbed so these run without a model; retrieval is real, because the
diff is about which chunks each technique retrieved.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import llm, vectorstore
from core.llm import LLMResponse

client = TestClient(app)


@pytest.fixture(autouse=True)
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_generate(system: str, user: str, model=None, **kwargs) -> LLMResponse:
        return LLMResponse(
            text="Stubbed answer citing dynamo.md.",
            input_tokens=100,
            output_tokens=20,
            model=model or "stub-model",
            backend="ollama",
        )

    monkeypatch.setattr(llm, "generate", fake_generate)


def compare(query: str, a: dict, b: dict):
    return client.post("/api/compare", json={"query": query, "a": a, "b": b})


def test_returns_both_sides_and_a_diff() -> None:
    response = compare(
        "What is hinted handoff?",
        {"technique": "standard-rag"},
        {"technique": "fusion-rag"},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["a"]["technique"] == "standard-rag"
    assert body["b"]["technique"] == "fusion-rag"
    assert body["a"]["retrieved_chunks"] and body["b"]["retrieved_chunks"]
    assert "diff" in body


def test_diff_identifies_the_axis_that_varied() -> None:
    body = compare(
        "What is hinted handoff?",
        {"technique": "standard-rag"},
        {"technique": "fusion-rag"},
    ).json()

    assert body["diff"]["same_technique"] is False
    assert body["diff"]["same_model"] is True


def test_same_technique_both_sides_retrieves_identically() -> None:
    """Retrieval is deterministic, so a technique compared against itself must
    produce 100% overlap. If this ever fails, retrieval has become unstable."""
    body = compare(
        "What is hinted handoff?",
        {"technique": "standard-rag"},
        {"technique": "standard-rag"},
    ).json()

    assert body["diff"]["chunk_overlap_pct"] == 100.0
    assert body["diff"]["only_a_chunk_ids"] == []
    assert body["diff"]["only_b_chunk_ids"] == []
    assert body["diff"]["same_technique"] is True


def test_techniques_retrieve_different_evidence() -> None:
    """The Phase 5 done-when: Standard and Fusion see different chunks.

    `hinted handoff` is the measured case — dense retrieval leads with raft.md,
    fusion leads with dynamo.md where the answer actually is.
    """
    body = compare(
        "What is hinted handoff?",
        {"technique": "standard-rag"},
        {"technique": "fusion-rag"},
    ).json()

    a_first = body["a"]["retrieved_chunks"][0]
    b_first = body["b"]["retrieved_chunks"][0]

    assert a_first["source"] == "raft.md", "dense unexpectedly led with the right doc"
    assert b_first["source"] == "dynamo.md", "fusion should surface the right doc first"
    assert body["diff"]["chunk_overlap_pct"] < 100


def test_unknown_technique_surfaces_its_status() -> None:
    """Errors from either side must not be swallowed by the comparison."""
    assert compare("q", {"technique": "nope"}, {"technique": "standard-rag"}).status_code == 404
    assert (
        compare("q", {"technique": "standard-rag"}, {"technique": "graph-rag"}).status_code
        == 409
    )


def test_rejects_an_empty_query() -> None:
    assert compare("", {"technique": "standard-rag"}, {"technique": "fusion-rag"}).status_code == 422
