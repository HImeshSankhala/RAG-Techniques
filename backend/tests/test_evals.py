"""The retrieval claim harness, as tests.

`evals/retrieval.py` is the same checks with a CLI around them. Both exist because
they answer to different readers: `make eval` prints a report for a human who just
added a document, and this file makes the same failures break `make test`, so a
stale claim cannot ride into main behind a green suite.

One test per claim, so pytest names the broken page rather than reporting a single
opaque "evals failed". The failure text is the harness's own report, which carries
the file and section to go fix.

No LLM call happens here, and none should: every assertion is retrieval only, which
is what keeps this fast enough to belong in the default suite.
"""

import pytest

from core import vectorstore
from evals.retrieval import CLAIMS, Claim, report, run_claim


@pytest.fixture(autouse=True, scope="session")
def require_index() -> None:
    """Session-scoped deliberately.

    `vectorstore.count()` opens a fresh Chroma client every call, and at one call
    per parametrised claim that guard cost more than all the retrieval it guards.
    """
    if vectorstore.count() == 0:
        pytest.skip("index is empty — run `make index` first")


@pytest.mark.parametrize("claim", CLAIMS, ids=[c.claim_id for c in CLAIMS])
def test_documented_claim_still_holds(claim: Claim) -> None:
    outcome = run_claim(claim)

    if outcome.status == "skip":
        pytest.skip(outcome.detail)

    assert outcome.status == "ok", "\n" + report(outcome)


def test_every_claim_names_a_file_to_go_fix() -> None:
    """The point of the harness is the failure message, so pin its one requirement.

    A claim whose `where` does not name a real file leaves the reader with a diff
    of chunk ids and nowhere to take it.
    """
    for claim in CLAIMS:
        assert any(
            marker in claim.where
            for marker in ("frontend/content/", "frontend/components/", "LEARNINGS/")
        ), f"{claim.claim_id} does not name a file to go fix: {claim.where!r}"
        assert claim.says, f"{claim.claim_id} does not state what it claims"


def test_claim_ids_are_unique() -> None:
    ids = [c.claim_id for c in CLAIMS]
    assert len(ids) == len(set(ids))
