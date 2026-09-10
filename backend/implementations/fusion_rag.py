"""Fusion RAG — two retrievers with opposite blind spots, merged by rank.

Standard RAG retrieves once, with one method, and inherits that method's
weaknesses wholesale. Dense retrieval is strong on paraphrase and weak on exact
tokens; BM25 is the mirror image. Fusion runs both and merges.

The shape is scatter-gather: fan out to independent retrievers, then combine.
The retrievers never see each other, which is what makes them safe to run
concurrently and easy to add to — a third retriever is one more entry in the list
handed to `reciprocal_rank_fusion`.

The mechanism lives in `core/retrieval.py`, not here, because Auto RAG's hybrid
route runs the identical scatter-gather-and-merge. What is left in this file is
what a technique is actually for: naming the stages and narrating them.
"""

from core import llm, retrieval
from core.config import settings
from core.fusion import RRF_K
from core.pipeline import Metadata, RAGPipeline, RAGResult, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness


class FusionRAG(RAGPipeline):
    name = "fusion-rag"
    display_name = "Fusion RAG"
    tagline = (
        "Run dense and keyword retrieval in parallel, then merge by rank instead of score."
    )

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model

        with steps.record("Retrieve (dense + BM25 in parallel)") as step:
            # Scatter-gather and the merge both live in core.retrieval.hybrid,
            # because Auto RAG's hybrid route needs the identical thing. That
            # means the fuse is timed inside this step rather than the next one;
            # RRF over 24 candidates is microseconds against two retrievals, so
            # the split below is about naming the two ideas for a reader, not
            # about attributing cost.
            result = retrieval.hybrid(query, settings.top_k)
            dense, sparse, chunks = result.dense, result.sparse, result.fused

            step.detail = (
                f"dense {len(dense)} (top {dense[0].chunk_id if dense else '—'}), "
                f"BM25 {len(sparse)} (top {sparse[0].chunk_id if sparse else '—'})"
            )

        with steps.record("Fuse by reciprocal rank") as step:
            # The interesting number for a reader: how much the two retrievers
            # actually disagreed. Full overlap means fusion changed nothing and
            # Standard RAG would have answered identically.
            dense_ids = {c.chunk_id for c in dense[: settings.top_k]}
            sparse_ids = {c.chunk_id for c in sparse[: settings.top_k]}
            overlap = len(dense_ids & sparse_ids)
            step.detail = (
                f"merged {len(dense)} + {len(sparse)} candidates into {len(chunks)} "
                f"(k={RRF_K}); retrievers agreed on {overlap}/{settings.top_k} of their tops"
            )

        if not chunks:
            return RAGResult(
                answer=(
                    "Nothing is indexed yet, so there is no context to answer from. "
                    "Run `make index` and try again."
                ),
                steps=steps.steps,
                metadata=Metadata(
                    model=model,
                    # Derived, not reported — no call was made. See standard_rag.
                    backend=llm.resolve_backend(model),
                    latency_ms=steps.elapsed_ms,
                    retrieval_passes=1,
                    termination_reason="empty_index",
                ),
            )

        with steps.record("Generate answer") as step:
            response = llm.generate(SYSTEM_PROMPT, build_prompt(query, chunks), model=model)
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
                # Still one pass: two retrievers ran, but only once each. The
                # count that changes here is retrievers, not passes — which is
                # exactly what distinguishes Fusion from Multi-Pass.
                retrieval_passes=1,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
                termination_reason="single_pass",
                groundedness=groundedness(response.text, chunks),
                cost_estimate_usd=round(cost, 6),
            ),
        )
