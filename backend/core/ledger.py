"""Per-run LLM accounting: how many calls a technique made, and what they cost.

Six pipelines were each keeping this by hand. The single-call three (Standard,
Fusion, Graph) read the counts straight off one `LLMResponse`; the looping three
(Multi-Pass, Auto, Agentic) carried `llm_calls`/`tokens_in`/`tokens_out` as three
running locals and incremented them at every call site. All six then ended with
a byte-identical ternary:

    cost = estimate_cost_usd(tokens_in, tokens_out) if backend == "anthropic" else 0.0

and spread the same five fields into `Metadata`. That is the rule of three
satisfied twice over, on the one piece of bookkeeping in this project that is
also a spend guardrail — so it stops being a style question.

WHY THE COST GUARDRAIL IS SAFE BY CONSTRUCTION
The old ternary was correct in all six copies, but only because six authors each
remembered to write it. Three things could have gone wrong in a seventh copy, and
none of them can go wrong here:

* Tokens without a backend. `record` takes a whole `LLMResponse`, which is frozen
  and carries `backend` beside `input_tokens`. There is no way to add tokens to
  this ledger without also telling it who served them.
* A paid call rounded down to free. `_paid` is sticky — set by any recorded
  Anthropic response and never cleared. Cost cannot fall back to 0.0 because a
  later free call overwrote a `backend` variable.
* A pipeline reporting its own number. `metadata()` does not accept `backend`,
  `llm_calls`, `tokens_*` or `cost_estimate_usd` as arguments. A pipeline cannot
  pass `cost_estimate_usd=0.0` because there is no parameter to pass it to. The
  fields it does take — latency, passes, termination, groundedness — are the ones
  no shared code could know.

That last point is the reason `metadata()` exists at all rather than six
accessors read into six `Metadata(...)` literals. Accessors would deduplicate the
arithmetic and leave the assembly — the step where a copy-paste omits the cost
line — exactly as hand-written as before.

WHAT IS DELIBERATELY NOT HERE
No timing (`StepRecorder` already owns that), no step recording, no per-call
history, no "helper vs answer" split, no config, no hooks. A ledger that could
also break spend down by call kind would be a metrics framework, and nothing in
this project reads such a breakdown: the cumulative record is `.usage.json`,
written by `core.llm` at call time, and the compare view reads exactly the five
fields below.

Why its own module rather than the two obvious homes. `core/pipeline.py` states
in its own docstring that it imports no LLM client, and it must keep that
property — the engine contract should not depend on which backend served a run.
`core/llm.py` is the backend adapter and knows nothing about `RAGResult`;
importing `Metadata` there would make the LLM client depend on the engine
contract, which is backwards. This module sits above both and depends on both,
the same shape as `core/retrieval.py` over vectorstore/keyword/fusion.
"""

from core import llm
from core.pipeline import Metadata


class LLMLedger:
    """Counts a single run's LLM calls and turns them into `Metadata`.

        ledger = LLMLedger(model)
        response = ledger.record(llm.generate(SYSTEM_PROMPT, prompt, model=model))
        ...
        metadata = ledger.metadata(
            latency_ms=steps.elapsed_ms,
            retrieval_passes=1,
            termination_reason="single_pass",
            groundedness=groundedness(response.text, chunks),
        )

    `record` returns the response it was given so it can wrap the call rather
    than follow it — a call that is made but not recorded should look wrong at a
    glance, and `llm.generate(...)` standing alone on a line does.
    """

    def __init__(self, model: str) -> None:
        # The model the run was asked for, not one read back off a response. A
        # run can end before any call is made (an empty index), and the compare
        # view still has to show which model the run was for.
        self.model = model
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._paid = False

    def record(self, response: llm.LLMResponse) -> llm.LLMResponse:
        """Count one call and hand the response straight back."""
        self.calls += 1
        self.tokens_in += response.input_tokens
        self.tokens_out += response.output_tokens
        # Sticky, and OR rather than assignment: "was any call paid" is the
        # question the cost depends on, and it must not be answerable "no"
        # because the last call happened to be local.
        self._paid = self._paid or response.backend == "anthropic"
        return response

    @property
    def backend(self) -> llm.Backend:
        """Who served this run.

        Derived from the model when nothing has been recorded yet, because the
        empty-index paths report a backend without having called one — and a
        blank backend renders as an empty badge beside a populated one in the
        compare view, which reads as a bug rather than as an unbuilt index.
        """
        return "anthropic" if self._paid else llm.resolve_backend(self.model)

    @property
    def cost_estimate_usd(self) -> float:
        """Estimated USD for this run. Exactly 0.0 unless a paid call was made."""
        if not self._paid:
            return 0.0
        return round(llm.estimate_cost_usd(self.tokens_in, self.tokens_out), 6)

    def metadata(
        self,
        *,
        latency_ms: float,
        retrieval_passes: int,
        termination_reason: str,
        groundedness: float = 0.0,
        feedback_votes: int = 0,
    ) -> Metadata:
        """The run's `Metadata`, with every LLM field filled from this ledger.

        Keyword-only, and the parameters are the things a pipeline knows that a
        ledger cannot: how long it took, how many times it retrieved, why it
        stopped, how well the answer cited its evidence, and — Feedback RAG only —
        how many stored votes shaped its ranking. `groundedness` defaults because
        the early-return paths have no answer to measure; `feedback_votes`
        defaults because seven of the eight techniques have no vote store.
        """
        return Metadata(
            model=self.model,
            backend=self.backend,
            latency_ms=latency_ms,
            llm_calls=self.calls,
            retrieval_passes=retrieval_passes,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            termination_reason=termination_reason,
            groundedness=groundedness,
            cost_estimate_usd=self.cost_estimate_usd,
            feedback_votes=feedback_votes,
        )
