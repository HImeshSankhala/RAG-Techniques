"""Standard RAG — the baseline every other technique is measured against.

Three stages, one LLM call, no loops:

    embed query -> retrieve top-k by vector similarity -> answer from those chunks

Everything the other eight techniques add is a response to a way this shape fails.
It retrieves once, so it cannot notice that it retrieved the wrong thing
(Multi-Pass). It matches on meaning only, so an exact term the embedding blurs is
lost (Fusion). It follows no links, so a question whose answer spans two documents
gets whichever single document scored best (Graph).
"""

from core import embeddings, llm, vectorstore
from core.config import settings
from core.pipeline import Chunk, Metadata, RAGPipeline, RAGResult, StepRecorder

SYSTEM_PROMPT = """You answer questions using only the context passages provided.

Rules:
- Use only information present in the passages. Do not add outside knowledge.
- If the passages do not contain the answer, say so plainly. Do not guess.
- Cite the source filename in parentheses after each claim, e.g. (dynamo.md).
- Be concise: a short paragraph, or a few sentences."""


def _build_prompt(query: str, chunks: list[Chunk]) -> str:
    """Lay out the retrieved passages, then the question.

    Question last on purpose: the passages are the bulk of the prompt and the same
    for everyone reading the same corpus, so keeping them first makes the prefix
    cacheable later. It also puts the question closest to the answer.

    Each passage is truncated, because on the paid backend an unusually long
    chunk is money rather than just latency.
    """
    passages = "\n\n".join(
        f"[Passage {i + 1}] (source: {chunk.source})\n{chunk.text[: settings.max_chunk_chars]}"
        for i, chunk in enumerate(chunks)
    )
    return f"Context passages:\n\n{passages}\n\nQuestion: {query}"


def _groundedness(answer: str, chunks: list[Chunk]) -> float:
    """Fraction of retrieved sources the answer cites.

    A compliance check, not an accuracy check. It measures whether the model
    followed the citation instruction — it cannot tell whether a cited claim is
    actually supported by the passage. Treated as a cheap signal that something
    is off (a 0.0 usually means the model ignored the context entirely), not as
    a quality score.
    """
    sources = {chunk.source for chunk in chunks}
    if not sources:
        return 0.0
    cited = sum(1 for source in sources if source in answer)
    return round(cited / len(sources), 3)


class StandardRAG(RAGPipeline):
    name = "standard-rag"
    display_name = "Standard RAG"
    tagline = (
        "Embed the query, retrieve top-k chunks, answer from them. "
        "The baseline everything else is measured against."
    )

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model

        with steps.record("Embed query") as step:
            query_vector = embeddings.embed_query(query)
            step.detail = f"{len(query_vector)}-dimensional vector"

        with steps.record("Retrieve chunks") as step:
            chunks = vectorstore.query(query_vector, settings.top_k)
            if chunks:
                sources = ", ".join(sorted({c.source for c in chunks}))
                step.detail = (
                    f"top {len(chunks)} of {vectorstore.count()} chunks "
                    f"(best score {chunks[0].score}) from {sources}"
                )
            else:
                step.detail = "no chunks found — is the index built? (make index)"

        if not chunks:
            return RAGResult(
                answer=(
                    "Nothing is indexed yet, so there is no context to answer from. "
                    "Run `make index` and try again."
                ),
                steps=steps.steps,
                metadata=Metadata(
                    model=model,
                    # No call was made, so the backend is derived rather than
                    # reported. Without it the compare view shows a blank field
                    # on one side and a real one on the other.
                    backend=llm.resolve_backend(model),
                    latency_ms=steps.elapsed_ms,
                    retrieval_passes=1,
                    termination_reason="empty_index",
                ),
            )

        with steps.record("Generate answer") as step:
            response = llm.generate(SYSTEM_PROMPT, _build_prompt(query, chunks), model=model)
            step.detail = (
                f"{response.model} ({response.backend}): "
                f"{response.input_tokens} in / {response.output_tokens} out"
            )

        cost = (
            llm.estimate_cost_usd(response.input_tokens, response.output_tokens)
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
                llm_calls=1,
                retrieval_passes=1,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
                termination_reason="single_pass",
                groundedness=_groundedness(response.text, chunks),
                cost_estimate_usd=round(cost, 6),
            ),
        )
