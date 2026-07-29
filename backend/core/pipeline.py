"""The engine contract every technique implements.

This module is the reason the compare feature is cheap. Nine techniques retrieve
in nine different ways, but they all hand back the same `RAGResult`, so anything
built on top — the playground, the compare view, the steps trace — is written once
against this shape rather than once per technique.

Nothing here imports FastAPI, Chroma, or the LLM client. The engine must stay
usable from a plain Python script.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A retrieved passage plus where it came from and how well it matched."""

    text: str
    source: str  # filename it was chunked out of
    score: float  # relevance; higher is better, comparable only within one retriever
    chunk_id: str


@dataclass(frozen=True)
class Step:
    """One stage of a pipeline run, for the UI trace.

    `detail` is shown to a human reading the trace, so it should say what actually
    happened ("retrieved 4 chunks, top score 0.71"), not restate the stage name.
    """

    name: str
    detail: str
    duration_ms: float


@dataclass
class RAGResult:
    """What every pipeline returns, regardless of how it retrieved."""

    answer: str
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class StepRecorder:
    """Times pipeline stages and collects them into `RAGResult.steps`.

    Every pipeline needs identical timing code, so it lives here once. The
    `detail` is set inside the block because most stages only know what to say
    after they have run — how many chunks came back, how many gaps were found.

        with steps.record("Retrieve") as step:
            chunks = store.query(q)
            step.detail = f"{len(chunks)} chunks"
    """

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._started = time.perf_counter()

    @contextmanager
    def record(self, name: str) -> Iterator["_PendingStep"]:
        pending = _PendingStep(name)
        start = time.perf_counter()
        try:
            yield pending
        finally:
            self.steps.append(
                Step(
                    name=pending.name,
                    detail=pending.detail,
                    duration_ms=round((time.perf_counter() - start) * 1000, 2),
                )
            )

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock time since this recorder was created."""
        return round((time.perf_counter() - self._started) * 1000, 2)


@dataclass
class _PendingStep:
    """Mutable handle yielded by `StepRecorder.record` so a block can set its detail."""

    name: str
    detail: str = ""


class RAGPipeline(ABC):
    """One RAG technique.

    Subclasses set the three class attributes and implement `run`. `name` is the
    slug and must match both the MDX filename and the registry key.
    """

    name: str
    display_name: str
    tagline: str

    @abstractmethod
    def run(self, query: str) -> RAGResult:
        """Answer `query`, recording every stage in the returned result's `steps`."""
