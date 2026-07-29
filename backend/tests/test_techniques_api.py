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
        assert set(entry) == {"name", "display_name", "tagline", "implemented"}
        assert entry["name"] and entry["display_name"] and entry["tagline"]
        assert isinstance(entry["implemented"], bool)


def test_slugs_are_unique_and_url_safe() -> None:
    """Slugs become /learn/[slug] routes and MDX filenames, so they must be clean."""
    names = [entry["name"] for entry in client.get("/api/techniques").json()]

    assert len(set(names)) == len(names)
    for name in names:
        assert name == name.lower()
        assert all(char.isalnum() or char == "-" for char in name)


def test_nothing_is_implemented_yet() -> None:
    """Phase 0 ships no pipelines. This test should be updated in Phase 1, not deleted."""
    body = client.get("/api/techniques").json()
    assert all(entry["implemented"] is False for entry in body)
