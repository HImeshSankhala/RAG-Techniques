"""Agentic RAG — the model decides what to retrieve next, not just whether to retry.

Multi-Pass also loops, so the interesting question for this phase is what an agent
adds over it. The honest answer is one thing, and it is not the loop:

    Multi-Pass:  retrieve(dense, the user's question) -> draft -> critique
                 -> retrieve(dense, gap text) -> redraft
    Agentic:     plan -> search(<strategy the model chose>, <query the model wrote>)
                 -> assess & re-plan -> ... -> answer

Every retrieval Multi-Pass makes is dense, and the only thing its critique
controls is the text embedded. The planner here chooses the *strategy* too —
vector, BM25, or fused — with the evidence gathered so far in front of it. That
is Auto RAG's router decision, except Auto RAG makes it once, blind, before it
has seen a single passage. Making it repeatedly, informed, is the whole of what
"agentic" buys on a corpus this size. Whether it is worth the calls is measured
on the learn page, not asserted here.

Two structural differences follow from planning before drafting:

* There is no draft. Multi-Pass spends a full answer generation per pass and then
  a critique to review it — 2 calls per pass. An iteration here is one planner
  call, and the answer is generated once at the end. Three iterations cost 4
  calls; Multi-Pass's three passes cost 5.
* The planner cannot be fooled the way the critique was. Multi-Pass's first
  critique prompt asked whether the *draft* answered the question, and a draft
  saying "the passages do not mention Spanner" answers it perfectly — so the loop
  never ran (see multi_pass_rag.CRITIQUE_SYSTEM). With no draft in existence
  there is nothing to grade but the passages.

Termination, which is the phase's real subject. Four things stop this loop:

1. `settings.agentic_max_iterations` — the hard cap, and the only guarantee.
2. The planner replying ANSWER: it believes the evidence is sufficient.
3. `repeated_action` — the planner asked for a search it already ran. An agent
   that can choose its next action can choose the same one forever, and this is
   the failure the other three guards do not name.
4. `no_new_evidence` — a *new* search returned nothing the agent did not have.
   The corpus is out of relevant material; further planning cannot conjure any.

(3) and (4) look alike and diagnose opposite problems: (3) is a bad planner, (4)
is a thin corpus. Collapsing them into one reason would hide which.
"""

from dataclasses import dataclass

from core import keyword, llm, retrieval
from core.config import settings
from core.pipeline import Chunk, Metadata, RAGPipeline, RAGResult, StepRecorder
from core.prompting import SYSTEM_PROMPT, build_prompt, groundedness

# The strategies the planner may call, sharing Auto RAG's vocabulary on purpose:
# the same three words mean the same three retrievers across the site, and a
# reader comparing the two traces should not have to learn a second set of names.
STRATEGIES = ("vector", "keyword", "hybrid")

# Chunks per search. Smaller than `top_k` because evidence accumulates and the
# planner is shown all of it every iteration:
#   worst case = max_iterations * ACTION_TOP_K = 3 * 3 = 9 chunks, ~13.5k chars,
#   ~3.4k tokens — inside the 8192 num_ctx configured in Phase 1, with room for
#   the planner's instructions and the final answer.
# Raising either constant without redoing that arithmetic silently reintroduces
# the truncation bug Phase 1 fixed.
ACTION_TOP_K = 3

# What the pipeline does when the planner's reply is unreadable, or when it asks
# to answer before it has retrieved anything. Dense search on the user's literal
# question is exactly Standard RAG's single step, so the fallback degrades to the
# baseline rather than to nothing.
#
# Deliberately not a special termination path: on the next iteration the planner
# either recovers or produces this same fallback again, and `repeated_action`
# stops it. One rule, no second exit to get wrong.
FALLBACK_STRATEGY = "vector"

PLANNER_SYSTEM = """You are the retrieval planner for a question-answering system.
You decide what to search for next. You never answer the question yourself.

You are given the question, the passages gathered so far, and the searches already
run. Judge the PASSAGES: do they contain everything the question asks for?

Reply with EXACTLY ONE line, and nothing else. Either:

ANSWER
    — the passages already cover every part of the question.

SEARCH <strategy> <keywords>
    — something is still missing. <strategy> is one of:

    vector   the missing material is conceptual, and could be written in many
             different words. Wording overlap will not find it.
    keyword  the missing material is named by a rare literal token: an identifier,
             an acronym, a protocol or system name. Finding that exact token is
             what matters.
    hybrid   both — a conceptual question about a specifically named thing.

<keywords> is what you would type into a search box: words, not a sentence, and
never a repeat of a search already run. Do not explain. Do not apologise.

If the question has several parts, check each part on its own. A passage that
answers one part is not evidence for another, and a passage that gives a reason
for something is not the same as naming the thing the question asks for. Reply
ANSWER only when every part of the question is covered by a passage."""

# That last paragraph is measured, not decoration, and it is worth knowing how
# little it buys. Asked to assess evidence covering only the first half of a
# two-part question, the planner without it replied: empty 3/3 on one evidence
# set, ANSWER 1/3 and a search for material it already had 2/3 on another. With
# it, one run of three named the actual missing hop — the only time in 12 samples
# anything did. It does not regress the opposite case: on evidence that genuinely
# covers the question, both versions answer ANSWER 3/3.
#
# So it is a small improvement to a step that is still wrong most of the time.
# See LEARNINGS/phase-9-agentic-rag.md; the honest summary is that this loop's
# stopping decision is the weakest component in the project.


@dataclass(frozen=True)
class Action:
    """One planned search: which retriever, and what to ask it."""

    strategy: str
    query: str

    @property
    def key(self) -> str:
        """Identity for repeat detection.

        Strategy included: the same words through BM25 and through embeddings are
        genuinely different searches, and an agent switching retrievers after a
        miss is the one repeat worth allowing.
        """
        return f"{self.strategy}:{' '.join(self.query.lower().split())}"

    def __str__(self) -> str:
        return f"{self.strategy}: {self.query}"


class AgenticRAG(RAGPipeline):
    name = "agentic-rag"

    def run(self, query: str, model: str | None = None) -> RAGResult:
        steps = StepRecorder()
        model = model or settings.default_model

        evidence: list[Chunk] = []
        seen: set[str] = set()
        history: list[Action] = []
        llm_calls = 0
        tokens_in = 0
        tokens_out = 0
        reason = "max_iterations"

        for iteration in range(1, settings.agentic_max_iterations + 1):
            # Iteration 1 has nothing to assess, so it is planning only. Later
            # ones are the "assess" half of PLAN.md's plan/retrieve/assess loop
            # and the planning half in a single call — the assessment's only
            # consumer is the next plan, so splitting them buys a verdict that
            # would immediately have to be re-supplied as input.
            label = "Plan" if iteration == 1 else "Assess & re-plan"
            with steps.record(f"{label} (iteration {iteration})") as step:
                plan = llm.generate(
                    PLANNER_SYSTEM,
                    _build_plan_prompt(query, evidence, history),
                    model=model,
                    # helper: one line of output needs a fraction of an answer's
                    # budget, and this runs once per iteration.
                    # reason: load-bearing. With thinking off this model answers
                    # the sufficiency question the agreeable way and the loop is a
                    # no-op — the same result Multi-Pass measured. See core/llm.py.
                    helper=True,
                    reason=True,
                )
                llm_calls += 1
                tokens_in += plan.input_tokens
                tokens_out += plan.output_tokens

                action, understood = parse_plan(plan.text)
                # The raw reply, not just the decision. When an agent loop goes
                # wrong, "the planner said hybrid" and "the planner said
                # something unreadable" are different diagnoses, and a trace
                # showing only the parsed outcome hides which one happened.
                raw = " ".join(plan.text.split())[:120] or "(empty reply)"

                if not understood:
                    action = Action(FALLBACK_STRATEGY, query)
                    step.detail = f"UNREADABLE PLAN — falling back to {action} · replied {raw!r}"
                elif action is None and not evidence:
                    # Answering with zero passages is the one outcome worse than
                    # an extra search: the model would answer from memory, which
                    # is the failure RAG exists to prevent.
                    action = Action(FALLBACK_STRATEGY, query)
                    step.detail = (
                        f"planner said ANSWER with no evidence yet — "
                        f"forcing {action} · replied {raw!r}"
                    )
                else:
                    step.detail = f"{action or 'ANSWER — evidence is sufficient'} · replied {raw!r}"

            if action is None:
                # The agent elected to stop. Distinct from Multi-Pass's
                # `no_gaps_found`/`gaps_closed`, which report a critique's verdict
                # on a draft that already exists; this is a prediction about
                # evidence made before anything was written. How many searches it
                # took is already in `retrieval_passes`, so it does not need a
                # second termination value.
                reason = "agent_stopped"
                break

            if action.key in {a.key for a in history}:
                reason = "repeated_action"
                break

            history.append(action)

            with steps.record(f"Search {action.strategy} (iteration {iteration})") as step:
                found = _search(action)
                new = [chunk for chunk in found if chunk.chunk_id not in seen]
                seen.update(chunk.chunk_id for chunk in new)
                evidence.extend(new)
                step.detail = _search_detail(action, found, new)

            if not evidence:
                return _empty_index(steps, plan, llm_calls, tokens_in, tokens_out)

            if not new:
                # A search the agent had not run before, returning only chunks it
                # already had. Planning again would re-read identical evidence and
                # produce an identical plan.
                reason = "no_new_evidence"
                break

        # The phase's claim, measured on every run rather than asserted once: did
        # planning find anything that embedding the user's question would not
        # have? One embedding and one HNSW lookup, no LLM call — free locally, and
        # it is what keeps "the agent adapts" from being an unchecked boast.
        with steps.record("Compare against plain dense retrieval") as step:
            step.detail = _describe_gain(query, evidence)

        with steps.record(f"Answer from {len(evidence)} chunks") as step:
            response = llm.generate(SYSTEM_PROMPT, build_prompt(query, evidence), model=model)
            llm_calls += 1
            tokens_in += response.input_tokens
            tokens_out += response.output_tokens
            step.detail = (
                f"{response.model} ({response.backend}): "
                f"{response.input_tokens} in / {response.output_tokens} out"
            )

        cost = (
            llm.estimate_cost_usd(tokens_in, tokens_out)
            if response.backend == "anthropic"
            else 0.0
        )

        return RAGResult(
            answer=response.text,
            retrieved_chunks=evidence,
            steps=steps.steps,
            metadata=Metadata(
                model=response.model,
                backend=response.backend,
                latency_ms=steps.elapsed_ms,
                llm_calls=llm_calls,
                # Searches actually executed. A planner call that decided to stop
                # retrieved nothing and is counted in `llm_calls` instead — the
                # two numbers diverging is how the compare view shows that this
                # technique spends calls on deciding, not only on retrieving.
                retrieval_passes=len(history),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                termination_reason=reason,
                groundedness=groundedness(response.text, evidence),
                cost_estimate_usd=round(cost, 6),
            ),
        )


def _empty_index(
    steps: StepRecorder, plan: llm.LLMResponse, llm_calls: int, tokens_in: int, tokens_out: int
) -> RAGResult:
    """Nothing indexed. Reported, not raised — and the planning already paid for
    is reported with it, the way Auto RAG reports its router's tokens."""
    return RAGResult(
        answer=(
            "Nothing is indexed yet, so there is no context to answer from. "
            "Run `make index` and try again."
        ),
        steps=steps.steps,
        metadata=Metadata(
            model=plan.model,
            backend=plan.backend,
            latency_ms=steps.elapsed_ms,
            llm_calls=llm_calls,
            retrieval_passes=1,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            termination_reason="empty_index",
        ),
    )


def _build_plan_prompt(query: str, evidence: list[Chunk], history: list[Action]) -> str:
    """Question, evidence so far, and the searches already run.

    The passages have to be included in full: asking "is anything missing?"
    without them invites the model to compare the question against its own
    pretrained knowledge and demand every fact this corpus happens not to hold.
    Multi-Pass learned that one the expensive way.

    `history` is the loop's memory, and the same caveat applies as there — it
    reduces repetition rather than eliminating it. Unlike Multi-Pass, a repeat
    here is caught structurally: `repeated_action` compares the parsed action, so
    an ignored instruction ends the loop instead of costing a pass.
    """
    if not evidence:
        return (
            f"Question: {query}\n\nNo passages have been retrieved yet. "
            "Plan the first search."
        )

    prompt = build_prompt(query, evidence)
    if history:
        prompt += "\n\nSearches already run:\n" + "\n".join(f"- {a}" for a in history)
        prompt += "\nDo not repeat any of these. If nothing else is missing, reply ANSWER."
    return prompt


def parse_plan(reply: str) -> tuple[Action | None, bool]:
    """Read the planner's reply. Returns (action, whether it was understood).

    `(None, True)` means ANSWER — a real decision. `(None, False)` means the
    reply was unusable, which the caller turns into the fallback search. Keeping
    them apart matters: a silent fallback and a confident stop look identical in
    the metadata otherwise, and they are the two things worth telling apart when
    an agent answers badly.

    Tolerant by construction. This parses free text from a small local model with
    thinking enabled, which reliably prefaces its reply with a sentence of
    narration — the exact shape Multi-Pass's gap parser had to survive. First
    usable line wins, so narration above the decision is skipped rather than
    mistaken for it.
    """
    for line in reply.splitlines():
        line = line.strip(" -*•\t#").strip()
        lowered = line.lower()

        if lowered.startswith("answer"):
            return None, True

        if lowered.startswith("search"):
            rest = line[len("search") :].strip(" :")
            strategy, _, text = rest.partition(" ")
            strategy = strategy.strip(" :*").lower()
            # Quotes come off. Measured, this model writes
            # `SEARCH keyword "Kafka ZooKeeper replacement protocol"` about half
            # the time — harmless to BM25, which tokenizes them away, but two
            # spellings of one search would otherwise slip past `repeated_action`.
            text = text.strip(" :\"'")
            # A strategy the menu does not contain, or a search for nothing, is
            # not a plan this pipeline can execute.
            if strategy in STRATEGIES and text:
                return Action(strategy, text), True

    return None, False


def _search(action: Action) -> list[Chunk]:
    """Run one planned search.

    A six-line dispatch rather than a shared router: Auto RAG's `_retrieve` is the
    only other one, it returns trace copy this pipeline does not want, and two
    occurrences are not three. The retrievers themselves are already shared — this
    calls the same `core.retrieval` and `core.keyword` Auto RAG does.
    """
    if action.strategy == "keyword":
        return keyword.query(action.query, ACTION_TOP_K)
    if action.strategy == "hybrid":
        return retrieval.hybrid(action.query, ACTION_TOP_K).fused
    return retrieval.dense(action.query, ACTION_TOP_K)


def _search_detail(action: Action, found: list[Chunk], new: list[Chunk]) -> str:
    if not found:
        return "no chunks found — is the index built? (make index)"
    return (
        f"{len(found)} chunks from {', '.join(sorted({c.source for c in found}))}, "
        f"{len(new)} new: {', '.join(c.chunk_id for c in new) or 'none'}"
    )


def _describe_gain(query: str, evidence: list[Chunk]) -> str:
    """What the agent's own queries found that the user's question would not.

    Borrowed from Graph RAG, which added it for the same reason: a technique that
    claims to retrieve better should show, per run, whether it did.
    """
    dense_ids = {c.chunk_id for c in retrieval.dense(query, settings.top_k)}
    extra = [c.chunk_id for c in evidence if c.chunk_id not in dense_ids]

    if not extra:
        return (
            f"dense top-{settings.top_k} on the raw question would have returned all "
            f"{len(evidence)} of these — planning changed nothing on this query"
        )
    return (
        f"{len(extra)} of {len(evidence)} chunks are NOT in dense top-{settings.top_k} "
        f"on the raw question: {', '.join(extra)}"
    )
