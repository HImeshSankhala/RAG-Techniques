"""Uploaded corpora: ingest, isolation, gating and expiry.

No LLM call anywhere in here. Everything this phase added happens before
generation — extraction, chunking, which collection a query reads, and which
techniques are allowed to run — so the tests that would tell us something are the
ones that stop short of the model.

Requires the index for the isolation checks: run `make index` first.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import keyword, retrieval, uploads, vectorstore
from core.config import settings
from core.ingest import UnsupportedDocumentError, read_upload, safe_source

# A three-chunk corpus whose vocabulary shares nothing with the demo corpus, so a
# result that mentions Dynamo or Raft can only have come from the wrong collection.
UPLOADED = (
    "quokka.md",
    (
        "# Quokkas\n\nThe quokka is a small macropod found on Rottnest Island.\n\n"
        + "Quokkas browse on succulent foliage and can survive months without drinking. " * 30
        + "\n\nA quokka's lifespan in the wild is about ten years.\n"
    ).encode(),
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_uploads():
    """Every uploaded collection this test made goes, whatever the test did."""
    before = {name for name, _ in vectorstore.uploads()}
    yield
    for name, _ in vectorstore.uploads():
        if name not in before:
            vectorstore.drop(name)
    keyword.reset()


def minimal_pdf(text: str) -> bytes:
    """The smallest PDF that pypdf will extract `text` from.

    Handwritten rather than generated, because the only alternative is a second
    PDF library in the dev dependencies to test the one already there.
    """
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        start,
    )
    return bytes(out)


# --- Extraction ------------------------------------------------------------


def test_reads_markdown_and_text() -> None:
    assert read_upload("notes.md", b"# Title\n\nBody text.").text.startswith("# Title")
    assert read_upload("notes.txt", b"Plain body.").text == "Plain body."


def test_reads_pdf_text() -> None:
    assert "Hinted handoff" in read_upload("paper.pdf", minimal_pdf("Hinted handoff")).text


def test_rejects_unsupported_suffix() -> None:
    with pytest.raises(UnsupportedDocumentError, match="only"):
        read_upload("payload.exe", b"MZ\x90\x00")


def test_rejects_non_utf8_text() -> None:
    with pytest.raises(UnsupportedDocumentError, match="UTF-8"):
        read_upload("notes.txt", b"\xff\xfe\x00broken")


def test_rejects_a_pdf_with_no_extractable_text() -> None:
    """A scanned page is images of text. It extracts to nothing, and saying so is
    the difference between an empty corpus and an unexplained empty corpus."""
    with pytest.raises(UnsupportedDocumentError, match="no text"):
        read_upload("scan.pdf", minimal_pdf(" "))


def test_rejects_text_over_the_character_cap() -> None:
    oversized = b"a" * (settings.upload_max_text_chars + 1)
    with pytest.raises(UnsupportedDocumentError, match="characters of text"):
        read_upload("big.txt", oversized)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\config.txt", "config.txt"),
        ("/absolute/path/notes.md", "notes.md"),
        ("...", "document"),
        ("weird;name`$(id).txt", "weirdnameid.txt"),
    ],
)
def test_filename_is_reduced_to_a_label(filename: str, expected: str) -> None:
    """The name comes from a browser form. It is never opened — but it does reach
    chunk ids and the reader's screen, so it loses everything that names a path."""
    assert safe_source(filename) == expected


# --- Limits ----------------------------------------------------------------


def test_rejects_too_many_files() -> None:
    files = [(f"{i}.txt", b"text") for i in range(settings.upload_max_files + 1)]
    with pytest.raises(uploads.UploadRejected, match="at a time"):
        uploads.create(files)


def test_rejects_one_oversized_file() -> None:
    with pytest.raises(uploads.UploadRejected, match="limit for one file"):
        uploads.create([("big.txt", b"a" * (settings.upload_max_file_bytes + 1))])


def test_rejects_an_oversized_total_under_the_per_file_cap() -> None:
    """Each part passes on its own. The aggregate is what they add up to, and
    without this cap `upload_max_files` files just under the per-file limit do."""
    each = settings.upload_max_file_bytes
    count = settings.upload_max_total_bytes // each + 1
    assert count <= settings.upload_max_files, "fixture would trip the file-count cap first"
    with pytest.raises(uploads.UploadRejected, match="in total"):
        uploads.create([(f"{i}.txt", b"a" * each) for i in range(count)])


# --- Isolation -------------------------------------------------------------


@pytest.fixture
def require_index() -> None:
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


def test_upload_does_not_touch_the_curated_corpus(require_index: None) -> None:
    before = vectorstore.count()
    corpus = uploads.create([UPLOADED])
    assert corpus.chunks > 0
    assert vectorstore.count() == before
    assert vectorstore.active() == vectorstore.COLLECTION_NAME


def test_dense_retrieval_reads_only_the_uploaded_corpus(require_index: None) -> None:
    corpus = uploads.create([UPLOADED])
    with uploads.corpus(corpus.session_id):
        found = retrieval.dense("How does Dynamo handle conflicting writes?", settings.top_k)
    assert found, "an uploaded corpus should still answer every query with something"
    assert {c.source for c in found} == {"quokka.md"}


def test_bm25_does_not_score_an_upload_with_demo_corpus_statistics(require_index: None) -> None:
    """The bug this phase could most easily have shipped.

    BM25 weights a term by how rare it is *in this corpus*. The cache was keyed on
    nothing, so whichever corpus was queried first served its term statistics to
    the other — silently, as a different ranking rather than an error.
    """
    demo = keyword.query("quokka", settings.top_k)
    assert all(c.source != "quokka.md" for c in demo)

    corpus = uploads.create([UPLOADED])
    with uploads.corpus(corpus.session_id):
        uploaded = keyword.query("quokka", settings.top_k)
    assert uploaded and {c.source for c in uploaded} == {"quokka.md"}

    # Back to the demo corpus, in the same process, after the upload was indexed.
    assert all(c.source != "quokka.md" for c in keyword.query("quokka", settings.top_k))


def test_chunk_ids_do_not_collide_with_the_curated_corpus(require_index: None) -> None:
    """An upload named like a bundled document keeps its own ids because it has
    its own collection — positional ids would otherwise overwrite each other."""
    corpus = uploads.create([("raft.md", b"Quokkas are not a consensus protocol. " * 50)])
    with uploads.corpus(corpus.session_id):
        mine = {c.chunk_id: c.text for c in vectorstore.all_chunks()}
    theirs = {c.chunk_id: c.text for c in vectorstore.all_chunks()}
    shared = set(mine) & set(theirs)
    assert shared, "the fixture is pointless unless the ids really do overlap"
    assert all(mine[i] != theirs[i] for i in shared)


# --- Lifecycle -------------------------------------------------------------


def test_delete_removes_the_corpus() -> None:
    corpus = uploads.create([UPLOADED])
    uploads.delete(corpus.session_id)
    with pytest.raises(uploads.UnknownCorpusError):
        with uploads.corpus(corpus.session_id):
            pass


def test_expired_corpus_is_refused_and_swept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expiry is read from the collection's own metadata, so shortening the TTL
    ages every stored corpus — no clock to fake and no sleep to wait out."""
    corpus = uploads.create([UPLOADED])
    monkeypatch.setattr(settings, "upload_ttl_seconds", -1)

    with pytest.raises(uploads.UnknownCorpusError, match="expired"):
        with uploads.corpus(corpus.session_id):
            pass

    # And it is gone from disk, not merely refused.
    assert not [name for name, _ in vectorstore.uploads() if corpus.session_id in name]


def test_unknown_session_never_falls_back_to_the_demo_corpus() -> None:
    """The worst failure available here would be answering a question about the
    visitor's documents out of somebody else's."""
    with pytest.raises(uploads.UnknownCorpusError):
        with uploads.corpus("nosuchsession"):
            pass


# --- API -------------------------------------------------------------------


def test_upload_endpoint_returns_a_usable_corpus(client: TestClient) -> None:
    response = client.post("/api/documents", files={"files": UPLOADED})
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 1 and body["chunks"] > 0
    assert body["label"] == "quokka.md"
    assert client.delete(f"/api/documents/{body['session_id']}").status_code == 204


@pytest.mark.parametrize("technique", ["graph-rag", "interactive-rag", "feedback-rag"])
def test_stateful_techniques_are_refused_on_uploads(client: TestClient, technique: str) -> None:
    """Refused at the route, before the pipeline runs and before the corpus opens.

    Graph RAG is the reason this is a gate rather than a pipeline's own empty
    state: the graph on disk is real, describes the demo corpus, and would resolve
    its chunk ids against the uploaded collection — citing passages that are not
    there.
    """
    corpus = uploads.create([UPLOADED])
    response = client.post(
        "/api/run",
        json={"technique": technique, "query": "What is a quokka?", "session_id": corpus.session_id},
    )
    assert response.status_code == 409
    assert "cannot run on uploaded documents" in response.json()["detail"]


def test_run_against_an_unknown_corpus_is_a_404(client: TestClient) -> None:
    response = client.post(
        "/api/run",
        json={"technique": "standard-rag", "query": "x", "session_id": "nosuchsession0"},
    )
    assert response.status_code == 404


def test_upload_rejects_an_unreadable_file(client: TestClient) -> None:
    response = client.post("/api/documents", files={"files": ("payload.exe", b"MZ\x90\x00")})
    assert response.status_code == 413


def test_compare_cannot_express_two_corpora() -> None:
    """Structural, not behavioural: the comparison carries one session id, and a
    side carries none — so 'each half read a different corpus' has no encoding."""
    from api.schemas import ComparisonSide, CompareRequest

    assert "session_id" in CompareRequest.model_fields
    assert "session_id" not in ComparisonSide.model_fields
