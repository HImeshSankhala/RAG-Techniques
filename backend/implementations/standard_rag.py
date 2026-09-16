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
from core.ledger import LLMLedger
from core.pipeline import RAGPipeline, RAGResult, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness


class StandardRAG(RAGPipeline):
    name = "standard-rag"

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model
        ledger = LLMLedger(model)

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
                # No call was made, so the ledger reports zero calls and derives
                # the backend from the model. Without that the compare view shows
                # a blank field on one side and a real one on the other.
                metadata=ledger.metadata(
                    latency_ms=steps.elapsed_ms,
                    retrieval_passes=1,
                    termination_reason="empty_index",
                ),
            )

        with steps.record("Generate answer") as step:
            response = ledger.record(
                llm.generate(SYSTEM_PROMPT, build_prompt(query, chunks), model=model)
            )
            step.detail = (
                f"{response.model} ({response.backend}): "
                f"{response.input_tokens} in / {response.output_tokens} out"
            )

        return RAGResult(
            answer=response.text,
            retrieved_chunks=chunks,
            steps=steps.steps,
            metadata=ledger.metadata(
                latency_ms=steps.elapsed_ms,
                retrieval_passes=1,
                termination_reason="single_pass",
                groundedness=groundedness(response.text, chunks),
            ),
        )
