"""Phase 0 smoke tests: the scaffold boots and the catalog endpoint holds its contract."""

from fastapi.testclient import TestClient

from api.main import app
from implementations.registry import CATALOG

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_techniques_returns_all_nine() -> None:
    response = client.get("/api/techniques")
    assert response.status_code == 200

    body = response.json()
    assert len(body) == 9
    assert len(CATALOG) == 9


def test_techniques_match_the_contract() -> None:
    body = client.get("/api/techniques").json()

    for entry in body:
        assert set(entry) == {"name", "display_name", "tagline", "implemented", "needs_human"}
        assert entry["name"] and entry["display_name"] and entry["tagline"]
        assert isinstance(entry["implemented"], bool)


def test_slugs_are_unique_and_url_safe() -> None:
    """Slugs become /learn/[slug] routes and MDX filenames, so they must be clean."""
    names = [entry["name"] for entry in client.get("/api/techniques").json()]

    assert len(set(names)) == len(names)
    for name in names:
        assert name == name.lower()
        assert all(char.isalnum() or char == "-" for char in name)


def test_implemented_flag_tracks_the_registry() -> None:
    """`implemented` is derived from PIPELINES — this asserts the derivation holds.

    Extend the expected set as each phase lands a technique; a mismatch here means
    the catalog and the registry have drifted apart.
    """
    body = client.get("/api/techniques").json()
    runnable = {entry["name"] for entry in body if entry["implemented"]}

    assert runnable == {
        "standard-rag",
        "fusion-rag",
        "multi-pass-rag",
        "auto-rag",
        "graph-rag",
        "agentic-rag",
        "interactive-rag",
        "feedback-rag",
    }


def test_only_interactive_rag_needs_a_human() -> None:
    """The compare view excludes exactly the techniques flagged here."""
    body = client.get("/api/techniques").json()
    assert {entry["name"] for entry in body if entry["needs_human"]} == {"interactive-rag"}


def test_run_rejects_an_unknown_technique() -> None:
    response = client.post("/api/run", json={"technique": "no-such-rag", "query": "hi"})
    assert response.status_code == 404


def test_run_distinguishes_docs_only_from_unknown() -> None:
    """A documented-but-unbuilt technique is a different mistake than a typo.

    Uses whichever technique is still docs-only; `graph-rag` was the example
    until Phase 8 made it runnable. `realm` is permanent — it cannot run locally
    at all — so it will not need swapping again.
    """
    response = client.post("/api/run", json={"technique": "realm", "query": "hi"})
    assert response.status_code == 409
    assert "not yet runnable" in response.json()["detail"]


def test_run_rejects_an_empty_query() -> None:
    response = client.post("/api/run", json={"technique": "standard-rag", "query": ""})
    assert response.status_code == 422


def test_run_rejects_a_whitespace_only_query() -> None:
    """A length check that counts characters lets "   " through.

    That is worse than a 500: it reaches the model, which invents a question and
    answers it, so the run looks successful and the answer is about nothing asked.
    """
    response = client.post("/api/run", json={"technique": "standard-rag", "query": "   "})
    assert response.status_code == 422
