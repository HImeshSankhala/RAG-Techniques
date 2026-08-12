"""Single source of truth for which techniques exist and which ones can run.

A technique graduates by adding its pipeline instance to ``PIPELINES`` — one line —
and ``implemented`` flips to True automatically because it is derived, never
hand-maintained. That derivation is the whole point: a hand-kept boolean is a second
source of truth waiting to drift from the first.
"""

from dataclasses import dataclass

from core.pipeline import RAGPipeline
from implementations.fusion_rag import FusionRAG
from implementations.standard_rag import StandardRAG

# A technique becomes runnable by appearing here — one line. `implemented` on the
# API is derived from membership, so there is no second place to remember.
PIPELINES: dict[str, RAGPipeline] = {
    StandardRAG.name: StandardRAG(),
    FusionRAG.name: FusionRAG(),
}


@dataclass(frozen=True)
class TechniqueInfo:
    """Catalog entry for one technique. Mirrors api.schemas.Technique."""

    name: str  # slug — matches the MDX filename and the pipeline's .name
    display_name: str
    tagline: str


# Order here is the order shown on the home page: roughly simple -> complex, which is
# also the order the phases build them in.
CATALOG: tuple[TechniqueInfo, ...] = (
    TechniqueInfo(
        name="standard-rag",
        display_name="Standard RAG",
        tagline="Embed the query, retrieve top-k chunks, answer from them. The baseline everything else is measured against.",
    ),
    TechniqueInfo(
        name="fusion-rag",
        display_name="Fusion RAG",
        tagline="Run dense and keyword retrieval in parallel, then merge by rank instead of score.",
    ),
    TechniqueInfo(
        name="multi-pass-rag",
        display_name="Multi-Pass RAG",
        tagline="Draft an answer, critique it for gaps, retrieve again to fill them. Latency bought with accuracy.",
    ),
    TechniqueInfo(
        name="auto-rag",
        display_name="Auto RAG",
        tagline="A cheap router call picks the retrieval strategy per query: vector, keyword, or hybrid.",
    ),
    TechniqueInfo(
        name="graph-rag",
        display_name="Graph RAG",
        tagline="Extract entities and relations at index time, then traverse the graph for multi-hop questions.",
    ),
    TechniqueInfo(
        name="agentic-rag",
        display_name="Agentic RAG",
        tagline="Plan, retrieve, assess, repeat — an agent loop with explicit stopping criteria.",
    ),
    TechniqueInfo(
        name="interactive-rag",
        display_name="Interactive RAG",
        tagline="Show a draft, let the user mark the useful chunks, then answer again. Human in the loop.",
    ),
    TechniqueInfo(
        name="feedback-rag",
        display_name="Feedback-Based RAG",
        tagline="Thumbs up/down on chunks persist and reweight future rankings.",
    ),
    TechniqueInfo(
        name="realm",
        display_name="REALM",
        tagline="Retrieval trained jointly with the language model. Docs-only here — it cannot run locally.",
    ),
)


def list_techniques() -> list[tuple[TechniqueInfo, bool]]:
    """Every technique in catalog order, paired with whether it has a runnable pipeline.

    Exists so the API layer never has to know PIPELINES exists — asking "can this run?"
    is an engine question.
    """
    return [(t, t.name in PIPELINES) for t in CATALOG]


def get_pipeline(name: str) -> RAGPipeline | None:
    """The runnable pipeline for a slug, or None if unknown or docs-only."""
    return PIPELINES.get(name)


def is_known(name: str) -> bool:
    """Whether the slug is in the catalog at all.

    Lets /api/run tell "no such technique" apart from "that one isn't built yet",
    which are different mistakes and deserve different messages.
    """
    return any(t.name == name for t in CATALOG)
