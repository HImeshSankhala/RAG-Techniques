"""Auto RAG — a cheap classifier decides how to retrieve, then the workers run.

Fusion RAG answered "dense or BM25?" with "always both". That is the safe answer
and it is not free: every query pays two retrievals and a merge, including the
many queries where one retriever alone would have returned the identical list.
Standard RAG made the opposite bet — always dense — and loses whenever the answer
hides behind a literal token an embedding blurs.

Auto RAG stops choosing once, at design time, and chooses per query:

    router (one cheap LLM call) -> vector | keyword | hybrid -> answer

This is the router pattern, and it is everywhere in production systems: a small
model in front of large ones, a rule engine in front of a service call, a cache
lookup in front of a computation. The whole pattern rests on one inequality —

    cost(router) << cost(always running the most expensive worker)

— and it is easy to violate by accident. This router is `helper=True,
reason=False` for exactly that reason. Measured on this machine, the identical
call with thinking on takes ~10s against ~0.5s off, while the work it dispatches
is a sub-second retrieval. A thinking router would cost twenty times the retrieval
it is choosing between, and Auto RAG would become a strictly slower Fusion RAG.
See `core/llm.py`.

Note what the router is NOT: an accuracy improvement. Its ceiling is whatever the
best single path would have returned, and hybrid already reaches that ceiling on
every query. Routing buys latency, and it buys it by risking the occasional wrong
path. Compared against Fusion RAG is where that trade shows up honestly.
"""

from core import keyword, llm, retrieval
from core.config import settings
from core.pipeline import Chunk, Metadata, RAGPipeline, RAGResult, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness

# The three retrieval paths, in the order the router prompt lists them.
ROUTES = ("vector", "keyword", "hybrid")

# Unparseable or unexpected router output falls back to hybrid.
#
# Hybrid rather than a specialist, because the routes are not peers: hybrid runs
# both retrievers and fuses, so its result set is a superset of what either
# specialist would have found. Choosing it can only cost the extra retrieval —
# never recall. Falling back to a specialist means betting on the exact question
# the router just failed to answer, and losing that bet drops the right chunk
# entirely.
#
# Retrieval here is local and free, so the fallback is cheap. That it is a
# corpus- and deployment-dependent judgement rather than a law is the point: with
# a billed retrieval API the safe default is worth re-deriving.
FALLBACK_ROUTE = "hybrid"

# One word out, no prose. Every extra token is latency on a step whose entire
# justification is being cheaper than the work it dispatches — and with thinking
# off the model has no scratch space anyway, so asking it to explain would only
# buy a confabulated explanation written after the decision.
ROUTER_SYSTEM = """You route a search query to one of three retrieval strategies.

Reply with EXACTLY ONE word, lowercase, and nothing else:

vector — the question is conceptual or paraphrased. It asks how or why something
works, and the answer could be written in many different words. Overlap between
the question's vocabulary and the document's is not what will find it.

keyword — the question names a rare, literal, technical term: an identifier, an
acronym, a filename, a specific API or protocol name. Finding the documents that
contain that exact token is what matters.

hybrid — the question does both: a conceptual question about a specifically named
thing, where you need the exact term AND the surrounding explanation.

Answer with one of: vector, keyword, hybrid"""

_ROUTE_LABELS = {
    "vector": "dense only",
    "keyword": "BM25 only",
    "hybrid": "dense + BM25, fused by rank",
}


class AutoRAG(RAGPipeline):
    name = "auto-rag"
    display_name = "Auto RAG"
    tagline = (
        "A cheap router call picks the retrieval strategy per query: "
        "vector, keyword, or hybrid."
    )

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model

        with steps.record("Route query") as step:
            router = llm.generate(
                ROUTER_SYSTEM,
                f"Query: {query}",
                model=model,
                # The two flags that make this a router rather than a second
                # worker. `helper` takes the tighter output budget; `reason=False`
                # is load-bearing — see the module docstring, core/llm.py, and
                # tests/test_llm_backends.py.
                helper=True,
                reason=False,
            )
            route, understood = parse_route(router.text)

            # The raw reply is in the trace, not just the parsed route. When the
            # router chooses badly, the difference between "the model said
            # keyword" and "the model said something unreadable" is the entire
            # diagnosis, and a trace showing only the outcome hides it.
            raw = " ".join(router.text.split())[:120] or "(empty reply)"
            step.detail = (
                f"route={route}"
                f"{'' if understood else ' (FALLBACK — router output unusable)'}"
                f" · router replied {raw!r}"
                f" · {router.input_tokens} in / {router.output_tokens} out"
            )

        with steps.record(f"Retrieve ({_ROUTE_LABELS[route]})") as step:
            chunks, step.detail = _retrieve(route, query)

        if not chunks:
            return RAGResult(
                answer=(
                    "Nothing is indexed yet, so there is no context to answer from. "
                    "Run `make index` and try again."
                ),
                steps=steps.steps,
                metadata=Metadata(
                    # Reported by the router rather than derived: unlike the other
                    # techniques, this one has already made a real call by the
                    # time it discovers the index is empty.
                    model=router.model,
                    backend=router.backend,
                    latency_ms=steps.elapsed_ms,
                    llm_calls=1,
                    retrieval_passes=1,
                    tokens_in=router.input_tokens,
                    tokens_out=router.output_tokens,
                    termination_reason="empty_index",
                ),
            )

        with steps.record("Generate answer") as step:
            response = llm.generate(SYSTEM_PROMPT, build_prompt(query, chunks), model=model)
            step.detail = (
                f"{response.model} ({response.backend}): "
                f"{response.input_tokens} in / {response.output_tokens} out"
            )

        tokens_in = router.input_tokens + response.input_tokens
        tokens_out = router.output_tokens + response.output_tokens
        cost = (
            llm.estimate_cost_usd(tokens_in, tokens_out)
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
                # Two, and the router is one of them. Hiding it would misreport
                # the technique's real cost in the compare view — the router is
                # cheap, not free, and that distinction is the phase's lesson.
                llm_calls=2,
                retrieval_passes=1,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                # The route is the lesson, so it goes in the field the compare
                # view already lines up side by side, not only in the trace.
                termination_reason=f"routed_{route}" if understood else f"routed_{route}_fallback",
                groundedness=groundedness(response.text, chunks),
                cost_estimate_usd=round(cost, 6),
            ),
        )


def _retrieve(route: str, query: str) -> tuple[list[Chunk], str]:
    """Run the chosen path. Returns its chunks and what the trace should say.

    Both specialists return [] on an unbuilt index rather than raising (see
    core/keyword.py), so every route reaches the pipeline's empty-index branch
    the same way.
    """
    if route == "vector":
        chunks = retrieval.dense(query, settings.top_k)
        return chunks, _sources_detail(chunks)

    if route == "keyword":
        chunks = keyword.query(query, settings.top_k)
        return chunks, _sources_detail(chunks)

    result = retrieval.hybrid(query, settings.top_k)
    if not result.fused:
        return [], _sources_detail([])

    # How much the two retrievers disagreed is the number worth reading here:
    # full overlap means the merge changed nothing and either specialist would
    # have done — i.e. the router had no decision to get wrong.
    dense_ids = {c.chunk_id for c in result.dense[: settings.top_k]}
    sparse_ids = {c.chunk_id for c in result.sparse[: settings.top_k]}
    return result.fused, (
        f"{_sources_detail(result.fused)}; retrievers agreed on "
        f"{len(dense_ids & sparse_ids)}/{settings.top_k} of their tops"
    )


def _sources_detail(chunks: list[Chunk]) -> str:
    if not chunks:
        return "no chunks found — is the index built? (make index)"
    return (
        f"{len(chunks)} chunks from {', '.join(sorted({c.source for c in chunks}))} "
        f"(best score {chunks[0].score})"
    )


def parse_route(reply: str) -> tuple[str, bool]:
    """Read the router's free text. Returns (route, whether it was understood).

    Substring search rather than equality on the whole reply, because this parses
    free text from a small local model that decorates its answers: "**keyword**",
    "Route: keyword", "keyword.". Earliest match wins, so a reply that names a
    route and then rules others out ("vector, not keyword") is read the way it is
    written.

    Returning the understood flag rather than just the route keeps the fallback
    visible. A silent fallback looks identical to a confident correct decision in
    the trace and in the metadata, which is exactly the case worth telling apart.
    """
    lowered = reply.lower()
    found = [(lowered.find(route), route) for route in ROUTES if route in lowered]
    if not found:
        return FALLBACK_ROUTE, False
    return min(found)[1], True
