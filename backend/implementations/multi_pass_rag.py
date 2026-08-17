"""Multi-Pass RAG — draft, find your own gaps, retrieve again, answer properly.

Every technique so far retrieves once. That is a bet that the question's wording
is close enough to the answer's wording for one lookup to find it, and the bet
loses on questions whose answer lives under vocabulary the question never uses.
Fusion widened *how* we search; it still only searched once, so it cannot help
when the missing evidence is only identifiable after you have read what you got.

The fix is to let the first answer tell you what the first search missed:

    retrieve -> draft -> critique for gaps -> retrieve for those gaps -> redraft

This is the first technique in the project with a loop, which makes termination
the real design problem rather than retrieval. A loop that decides for itself
whether to go around again is a loop that can decide "yes" forever, and on a paid
backend every extra turn is money. Three independent things stop it:

1. A hard pass cap (`settings.multi_pass_max_passes`). Not a fallback — the
   guarantee. Everything else is best-effort; only this one cannot be talked out of
   stopping by a confused model.
2. The critique reporting no gaps.
3. Re-retrieval surfacing nothing new. A critique that keeps asking for evidence
   the corpus does not contain would otherwise burn every remaining pass fetching
   the same chunks.

(2) and (3) are what make the loop usually stop early; (1) is what makes it always
stop.
"""

from core import embeddings, llm, vectorstore
from core.config import settings
from core.pipeline import Chunk, Metadata, RAGPipeline, RAGResult, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness

# Retrieval passes come from config, beside the other spend caps — an iteration
# cap on a self-terminating loop is a budget guarantee, not a tuning knob. At the
# configured 3, the worst case is 5 LLM calls: draft, critique, redraft, critique,
# redraft. There is deliberately no critique after the final redraft; nothing could
# act on its answer, so it would be a call bought purely to be discarded.
#
# Gap queries honoured per critique. The model will happily list six things it
# would like; each one is retrieval and prompt tokens, so the pipeline decides how
# many it can afford rather than letting the critique decide for it.
MAX_GAPS = 2

# Chunks fetched per gap query. Deliberately half of `top_k`: context accumulates
# across passes, and the total has to stay inside Ollama's num_ctx.
# Worst case = top_k + (passes - 1) * MAX_GAPS * GAP_TOP_K = 4 + 2*2*2 = 12
# chunks, ~18k chars, ~4.5k tokens — comfortably under the 8192 configured in
# Phase 1, with room for the answer. Raising any of these three constants without
# redoing that arithmetic silently reintroduces the truncation bug.
GAP_TOP_K = max(1, settings.top_k // 2)

# Judge the EVIDENCE, not the writing. The first version of this prompt asked
# whether "the draft answers the question from the passages", and a draft saying
# "the passages do not mention Spanner" satisfies that reading perfectly — it is a
# correct response to those passages. The critique answered COMPLETE on every
# query, including ones the corpus provably cannot answer, and the loop never ran.
# The model was being obedient; the question was wrong.
CRITIQUE_SYSTEM = """You find what a draft answer is missing.

You are given a question, the passages retrieved to answer it, and a draft written
from them. Judge the PASSAGES, not the writing: do they contain everything the
question asks for?

Reply with EITHER:
- the single word COMPLETE, if the passages fully cover every part of the question; OR
- up to 2 short search queries, one per line, naming what still has to be found.

A draft that says the passages do not contain the answer, or that answers only part
of the question, is NOT complete. That is the clearest signal that something is
missing — reply with search queries for the missing material.

A search query is the text you would type to find the missing passage — keywords,
not a sentence. Do not explain. Do not apologise. Do not repeat the question."""

# A critique that answers "COMPLETE." or "complete - the draft is fine" is saying
# the same thing as "COMPLETE"; only the first word is load-bearing.
_COMPLETE = "complete"


class MultiPassRAG(RAGPipeline):
    name = "multi-pass-rag"
    display_name = "Multi-Pass RAG"
    tagline = (
        "Draft an answer, critique it for gaps, retrieve again to fill them. "
        "Latency bought with accuracy."
    )

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model

        with steps.record("Retrieve (pass 1)") as step:
            chunks = vectorstore.query(embeddings.embed_query(query), settings.top_k)
            step.detail = (
                f"{len(chunks)} chunks from "
                f"{', '.join(sorted({c.source for c in chunks}))}"
                if chunks
                else "no chunks found — is the index built? (make index)"
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
                    backend=llm.resolve_backend(model),
                    latency_ms=steps.elapsed_ms,
                    retrieval_passes=1,
                    termination_reason="empty_index",
                ),
            )

        with steps.record("Draft answer") as step:
            response = llm.generate(SYSTEM_PROMPT, build_prompt(query, chunks), model=model)
            step.detail = _generation_detail(response)

        tokens_in, tokens_out = response.input_tokens, response.output_tokens
        llm_calls = 1
        passes = 1
        reason = "max_iterations"

        while passes < settings.multi_pass_max_passes:
            with steps.record(f"Critique draft (pass {passes})") as step:
                # helper: a list of search queries needs a fraction of an answer's
                # budget, and this runs once per pass.
                # reason: without it this model answers COMPLETE to everything and
                # the loop is a no-op. Measured — see core/llm.py.
                critique = llm.generate(
                    CRITIQUE_SYSTEM,
                    _build_critique_prompt(query, chunks, response.text),
                    model=model,
                    helper=True,
                    reason=True,
                )
                llm_calls += 1
                tokens_in += critique.input_tokens
                tokens_out += critique.output_tokens

                gaps = _parse_gaps(critique.text)
                if gaps:
                    step.detail = f"gaps: {'; '.join(gaps)}"
                elif critique.text.strip():
                    step.detail = "no gaps — the draft stands"
                else:
                    # An empty reply is indistinguishable from agreement once it
                    # reaches `gaps`, and that is exactly how this loop disabled
                    # itself for an afternoon: reasoning consumed the whole output
                    # budget and left no reply. Same behaviour, but the trace says
                    # so rather than reporting a verdict that was never given.
                    step.detail = "critique returned nothing — treating as no gaps"

            if not gaps:
                # Distinguishing these two matters in the compare view: one says
                # the extra machinery was never needed, the other says it worked.
                reason = "single_pass" if passes == 1 else "gaps_closed"
                break

            with steps.record(f"Retrieve for gaps (pass {passes + 1})") as step:
                found = _retrieve_gaps(gaps)
                merged = _merge(chunks, found)
                added = len(merged) - len(chunks)
                step.detail = (
                    f"{len(found)} chunks for {len(gaps)} gap "
                    f"quer{'y' if len(gaps) == 1 else 'ies'}, {added} new"
                )

            # Counted here, not after the check below: the retrieval happened and
            # cost time whether or not it helped. Reporting it only on success
            # would make the trace disagree with the metadata — the step above
            # already calls itself pass 2 — and would flatter the technique by
            # hiding its wasted work, which is the thing the compare view exists
            # to expose.
            passes += 1

            if not added:
                # The critique asked for evidence this corpus does not have.
                # Redrafting on identical context would produce an identical
                # answer, so spending the remaining passes on it is pure cost.
                reason = "no_new_evidence"
                break

            chunks = merged

            with steps.record(f"Redraft with {len(chunks)} chunks (pass {passes})") as step:
                response = llm.generate(SYSTEM_PROMPT, build_prompt(query, chunks), model=model)
                llm_calls += 1
                tokens_in += response.input_tokens
                tokens_out += response.output_tokens
                step.detail = _generation_detail(response)

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
                llm_calls=llm_calls,
                retrieval_passes=passes,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                termination_reason=reason,
                groundedness=groundedness(response.text, chunks),
                cost_estimate_usd=round(cost, 6),
            ),
        )


def _build_critique_prompt(query: str, chunks: list[Chunk], draft: str) -> str:
    """Show the critique what the drafter saw, plus what it wrote.

    The passages have to be included: asking "is anything missing?" without them
    invites the model to compare the draft against its own pretrained knowledge
    and report every fact the corpus happens not to contain.
    """
    return f"{build_prompt(query, chunks)}\n\nDraft answer:\n{draft}"


# A search query is keywords, so anything long or sentence-shaped is the model
# narrating rather than searching. Thinking is enabled for this call (see
# core/llm.py), and a reasoning model reliably prefaces its queries with a line
# like "The passages do not mention Spanner. Therefore the answer is incomplete."
# Retrieving for that sentence would embed the critique's prose instead of the
# missing topic — a wasted pass that also pollutes the context.
_MAX_QUERY_WORDS = 8


def _parse_gaps(critique: str) -> list[str]:
    """Turn the critique's reply into search queries, or [] if it found none.

    Tolerant by construction. This parses free text from a small local model, so
    both failure directions cost a pass: misreading "COMPLETE" as a gap spends one
    on nothing, and misreading prose as a query spends one on the wrong thing.
    """
    lines = [line.strip(" -*•\t").strip() for line in critique.splitlines()]
    lines = [line for line in lines if line]

    if not lines or lines[0].lower().startswith(_COMPLETE):
        return []

    queries = [
        line
        for line in lines
        if not line.lower().startswith(_COMPLETE) and _looks_like_a_query(line)
    ]
    return queries[:MAX_GAPS]


def _looks_like_a_query(line: str) -> bool:
    """Keywords, not a sentence. Ends-with-period is the cheapest reliable tell."""
    return len(line.split()) <= _MAX_QUERY_WORDS and not line.endswith((".", ":", "!"))


def _retrieve_gaps(gaps: list[str]) -> list[Chunk]:
    """Retrieve for each gap separately.

    One embedding per gap rather than one for all of them concatenated: averaging
    two unrelated questions into a single vector lands between them, which is
    where neither answer is. This is what "targeted" re-retrieval means — the
    second search is aimed at a specific hole, not at the topic again.
    """
    found: list[Chunk] = []
    for gap in gaps:
        found.extend(vectorstore.query(embeddings.embed_query(gap), GAP_TOP_K))
    return found


def _merge(existing: list[Chunk], found: list[Chunk]) -> list[Chunk]:
    """Append genuinely new chunks, preserving order.

    Order is kept rather than re-ranked by score: scores from different queries
    are not comparable (the same reason Fusion merges by rank), and the reader of
    the trace should see pass 1's evidence followed by what each later pass added.
    """
    seen = {chunk.chunk_id for chunk in existing}
    merged = list(existing)

    for chunk in found:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    return merged


def _generation_detail(response: llm.LLMResponse) -> str:
    return (
        f"{response.model} ({response.backend}): "
        f"{response.input_tokens} in / {response.output_tokens} out"
    )
