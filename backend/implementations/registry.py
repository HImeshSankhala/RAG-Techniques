"""Single source of truth for which techniques exist and which ones can run.

A technique graduates by adding its pipeline instance to ``PIPELINES`` — one line —
and ``implemented`` flips to True automatically because it is derived, never
hand-maintained. That derivation is the whole point: a hand-kept boolean is a second
source of truth waiting to drift from the first.
"""

from dataclasses import dataclass

from core.pipeline import RAGPipeline
from implementations.agentic_rag import AgenticRAG
from implementations.auto_rag import AutoRAG
from implementations.feedback_rag import FeedbackRAG
from implementations.fusion_rag import FusionRAG
from implementations.graph_rag import GraphRAG
from implementations.interactive_rag import InteractiveRAG
from implementations.multi_pass_rag import MultiPassRAG
from implementations.standard_rag import StandardRAG

# A technique becomes runnable by appearing here — one line. `implemented` on the
# API is derived from membership, so there is no second place to remember.
PIPELINES: dict[str, RAGPipeline] = {
    StandardRAG.name: StandardRAG(),
    FusionRAG.name: FusionRAG(),
    MultiPassRAG.name: MultiPassRAG(),
    AutoRAG.name: AutoRAG(),
    GraphRAG.name: GraphRAG(),
    AgenticRAG.name: AgenticRAG(),
    InteractiveRAG.name: InteractiveRAG(),
    FeedbackRAG.name: FeedbackRAG(),
}


@dataclass(frozen=True)
class TechniqueInfo:
    """Catalog entry for one technique. Mirrors api.schemas.Technique."""

    name: str  # slug — matches the MDX filename and the pipeline's .name
    display_name: str
    tagline: str

    # Editorial ranges for the home comparison table, in the same spirit as
    # `tagline`: prose about the technique, not a measurement. Strings, because
    # the honest answer is usually a range — `Metadata.llm_calls` counts one run,
    # and Auto RAG's count depends on the route that run happened to pick. Each
    # range is read off the pipeline's own code, never estimated, including the
    # floors that only an edge path reaches. LEARNINGS/phase-12-showcase.md walks
    # through the LLM-call derivations; the retrieval-pass ones are one line each
    # at the `retrieval_passes=` site in the pipeline concerned.
    llm_calls_range: str
    retrieval_passes_range: str

    # Documented here, impossible to run here, ever. Distinct from `implemented`,
    # which is derived from PIPELINES membership and means only "no pipeline yet".
    # Conflating the two made the playground offer REALM as "not built yet" — a
    # roadmap promise about a pre-training method that no amount of building would
    # keep. Declared rather than derived, because no membership test can tell
    # "nobody has built it" apart from "nobody can".
    docs_only: bool = False


# Order here is the order shown on the home page: roughly simple -> complex, which is
# also the order the phases build them in.
CATALOG: tuple[TechniqueInfo, ...] = (
    TechniqueInfo(
        name="standard-rag",
        display_name="Standard RAG",
        tagline="Embed the query, retrieve top-k chunks, answer from them. The baseline everything else is measured against.",
        llm_calls_range="1",
        retrieval_passes_range="1",
    ),
    TechniqueInfo(
        name="fusion-rag",
        display_name="Fusion RAG",
        tagline="Run dense and keyword retrieval in parallel, then merge by rank instead of score.",
        llm_calls_range="1",
        retrieval_passes_range="1",
    ),
    TechniqueInfo(
        name="multi-pass-rag",
        display_name="Multi-Pass RAG",
        tagline="Draft an answer, critique it for gaps, retrieve again to fill them. Latency bought with accuracy.",
        llm_calls_range="2–5",
        retrieval_passes_range="1–3",
    ),
    TechniqueInfo(
        name="auto-rag",
        display_name="Auto RAG",
        tagline="A cheap router call picks the retrieval strategy per query: vector, keyword, or hybrid.",
        llm_calls_range="2",
        retrieval_passes_range="1",
    ),
    TechniqueInfo(
        name="graph-rag",
        display_name="Graph RAG",
        tagline="Extract entities and relations at index time, then traverse the graph for multi-hop questions.",
        # Zero of each until `make graph` has been run, which is the state a
        # stranger is in after the quickstart: the optional graph build is what
        # takes this row off its floor, and without it the pipeline returns
        # `no_graph` having retrieved and generated nothing.
        llm_calls_range="0–1 (+1 per chunk at index time)",
        retrieval_passes_range="0–1",
    ),
    TechniqueInfo(
        name="agentic-rag",
        display_name="Agentic RAG",
        tagline="Plan, retrieve, assess, repeat — an agent loop with explicit stopping criteria.",
        # Floor is 3, not 2: iteration 1 can never exit the loop. A planner that
        # says ANSWER with no evidence is forced into a fallback search, and
        # neither `repeated_action` nor `no_new_evidence` can fire while the
        # history and the seen-set are still empty — so a second planner call
        # always happens. See agentic_rag.py's loop.
        llm_calls_range="3–4",
        retrieval_passes_range="1–3",
    ),
    TechniqueInfo(
        name="interactive-rag",
        display_name="Interactive RAG",
        tagline="Show a draft, let the user mark the useful chunks, then answer again. Human in the loop.",
        # 1 when the human keeps every passage and adds no hint: `finalize` returns
        # the draft unchanged and makes no call at all (`no_change`).
        llm_calls_range="1–2 (draft + final)",
        retrieval_passes_range="1–2",
    ),
    TechniqueInfo(
        name="feedback-rag",
        display_name="Feedback-Based RAG",
        tagline="Thumbs up/down on chunks persist and reweight future rankings.",
        llm_calls_range="1",
        retrieval_passes_range="1",
    ),
    TechniqueInfo(
        name="realm",
        display_name="REALM",
        tagline="Retrieval trained jointly with the language model. Docs-only here — it cannot run locally.",
        llm_calls_range="—",
        retrieval_passes_range="—",
        docs_only=True,
    ),
)


def list_techniques() -> list[tuple[TechniqueInfo, bool]]:
    """Every technique in catalog order, paired with whether it has a runnable pipeline.

    Exists so the API layer never has to know PIPELINES exists — asking "can this run?"
    is an engine question.
    """
    return [(t, t.name in PIPELINES) for t in CATALOG]


def needs_human(name: str) -> bool:
    """Whether the technique pauses mid-run for a person, so cannot run unattended.

    Derived from the pipeline's type rather than declared as a flag on
    `RAGPipeline`: one technique has this property, and a second "special
    technique" attribute on the engine contract is not worth one member. The
    compare view excludes these — a side whose human is stubbed out would be a
    demo that lies about what the technique does.
    """
    return isinstance(PIPELINES.get(name), InteractiveRAG)


def get_pipeline(name: str) -> RAGPipeline | None:
    """The runnable pipeline for a slug, or None if unknown or docs-only."""
    return PIPELINES.get(name)


def is_docs_only(name: str) -> bool:
    """Whether the slug is documented but can never run here.

    Reads the catalog's declaration rather than inferring it from `PIPELINES`:
    "no pipeline" and "no possible pipeline" are different facts, and only one of
    them is fixed by writing more code.
    """
    return any(t.name == name and t.docs_only for t in CATALOG)


def is_known(name: str) -> bool:
    """Whether the slug is in the catalog at all.

    Lets /api/run tell "no such technique" apart from "that one isn't built yet",
    which are different mistakes and deserve different messages.
    """
    return any(t.name == name for t in CATALOG)
