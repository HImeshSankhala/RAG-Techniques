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
from core.pipeline import RAGPipeline, RAGResult, StepRecorder

SYSTEM_PROMPT = """You answer questions using only the context passages provided.

Rules:
- Use only information present in the passages. Do not add outside knowledge.
- If the passages do not contain the answer, say so plainly. Do not guess.
- Cite the source filename in parentheses after each claim, e.g. (dynamo.md).
- Be concise: a short paragraph, or a few sentences."""


def _build_prompt(query: str, chunks: list) -> str:
    """Lay out the retrieved passages, then the question.

    Question last on purpose: the passages are the bulk of the prompt and the same
    for everyone reading the same corpus, so keeping them first makes the prefix
    cacheable later. It also puts the question closest to the answer.
    """
    passages = "\n\n".join(
        f"[Passage {i + 1}] (source: {chunk.source})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    return f"Context passages:\n\n{passages}\n\nQuestion: {query}"


class StandardRAG(RAGPipeline):
    name = "standard-rag"
    display_name = "Standard RAG"
    tagline = (
        "Embed the query, retrieve top-k chunks, answer from them. "
        "The baseline everything else is measured against."
    )

    def run(self, query: str) -> RAGResult:
        steps = StepRecorder()

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
                metadata={"latency_ms": steps.elapsed_ms, "llm_calls": 0},
            )

        with steps.record("Generate answer") as step:
            response = llm.complete(SYSTEM_PROMPT, _build_prompt(query, chunks))
            step.detail = (
                f"{settings.answer_model}: "
                f"{response.input_tokens} in / {response.output_tokens} out"
            )

        return RAGResult(
            answer=response.text,
            retrieved_chunks=chunks,
            steps=steps.steps,
            metadata={
                "latency_ms": steps.elapsed_ms,
                "llm_calls": 1,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": settings.answer_model,
            },
        )
