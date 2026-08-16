"""Prompt construction and the grounding check, shared by every technique.

These started in `standard_rag.py` because it was the only technique that needed
them. Fusion RAG then imported them across module boundaries, reaching into
another pipeline's privates; Multi-Pass is the third consumer, which is where the
rule of three fires. They are not Standard RAG's logic — they are the project's
answer to "how do you show passages to a model and check it used them" — so they
belong in core/ where CLAUDE.md says shared engine logic lives.

Nothing here knows how the chunks were retrieved, which is the point: dense,
BM25-fused, and multi-pass evidence all reach the model in the same shape, so a
difference in two answers is a difference in retrieval rather than in framing.
"""

from core.config import settings
from core.pipeline import Chunk

SYSTEM_PROMPT = """You answer questions using only the context passages provided.

Rules:
- Use only information present in the passages. Do not add outside knowledge.
- If the passages do not contain the answer, say so plainly. Do not guess.
- Cite the source filename in parentheses after each claim, e.g. (dynamo.md).
- Be concise: a short paragraph, or a few sentences."""


def build_prompt(query: str, chunks: list[Chunk]) -> str:
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


def groundedness(answer: str, chunks: list[Chunk]) -> float:
    """Fraction of retrieved sources the answer cites.

    A compliance check, not an accuracy check. It measures whether the model
    followed the citation instruction — it cannot tell whether a cited claim is
    actually supported by the passage. Treated as a cheap signal that something
    is off (a 0.0 usually means the model ignored the context entirely), not as
    a quality score.

    Note for Multi-Pass: the denominator is every source retrieved across all
    passes, so a technique that retrieves more evidence is scored against more
    sources. That is intentional — citing 3 of 8 sources is genuinely weaker
    grounding than citing 3 of 3.
    """
    sources = {chunk.source for chunk in chunks}
    if not sources:
        return 0.0
    cited = sum(1 for source in sources if source in answer)
    return round(cited / len(sources), 3)
