/**
 * The branch that decides which lesson the compare page teaches.
 *
 * `summarise()` picks one of five sentences. Each one is a different claim about
 * why two answers differ — "retrieval differed", "generation differed", "nothing
 * is indexed". Pick the wrong branch and the page states something false with
 * full confidence, and nothing in the type system objects: every branch returns
 * a string, and both sides of `if (overlap === 0)` typecheck.
 *
 * So the assertions here are about MEANING, not wording. They check which claim
 * the sentence makes (does it say retrieval, or generation, or empty index),
 * not its exact prose, so that rewording the copy does not break the suite while
 * flipping a comparison still does.
 */

import { describe, expect, it, test } from "vitest";

import { formatDelta, hasNoEvidence, summarise } from "@/components/DiffSummary";
import type { Chunk, ComparisonDiff, RunResponse } from "@/lib/api";

function chunks(...ids: string[]): Chunk[] {
  return ids.map((chunk_id) => ({ text: "…", source: "dynamo.md", score: 0.5, chunk_id }));
}

function run(overrides: Partial<RunResponse> = {}): RunResponse {
  return {
    technique: "standard-rag",
    query: "How does Dynamo handle conflicting concurrent writes?",
    answer: "…",
    retrieved_chunks: chunks("dynamo.md#0", "dynamo.md#1"),
    steps: [],
    ...overrides,
    metadata: {
      model: "qwen3:8b",
      backend: "ollama",
      latency_ms: 900,
      llm_calls: 1,
      retrieval_passes: 1,
      tokens_in: 1000,
      tokens_out: 200,
      termination_reason: "complete",
      groundedness: 1,
      cost_estimate_usd: 0,
      ...overrides.metadata,
    },
  };
}

/** Mirrors `backend/api/routes/compare.py::_diff` — pct denominator is the LARGER side. */
function diff(overrides: Partial<ComparisonDiff> = {}): ComparisonDiff {
  return {
    chunk_overlap: 2,
    chunk_overlap_pct: 100,
    shared_chunk_ids: [],
    only_a_chunk_ids: [],
    only_b_chunk_ids: [],
    latency_delta_ms: 0,
    llm_calls_delta: 0,
    steps_delta: 0,
    tokens_in_delta: 0,
    tokens_out_delta: 0,
    cost_delta_usd: 0,
    same_technique: true,
    same_model: true,
    ...overrides,
  };
}

// --- The axis clause: which pair of things is being compared ----------------

describe("summarise — the axis clause", () => {
  it("names run-to-run variation when technique and model are both the same", () => {
    const sentence = summarise(
      diff({ same_technique: true, same_model: true }),
      run(),
      run(),
      false,
    );

    expect(sentence).toContain("same technique on the same model");
  });

  it("names both models when only the model varies", () => {
    const sentence = summarise(
      diff({ same_technique: true, same_model: false }),
      run({ metadata: { ...run().metadata, model: "qwen3:8b" } }),
      run({ metadata: { ...run().metadata, model: "claude-haiku-4-5" } }),
      false,
    );

    expect(sentence).toContain("Same technique, different models");
    expect(sentence).toContain("qwen3:8b");
    expect(sentence).toContain("claude-haiku-4-5");
  });

  it("names both techniques when only the technique varies", () => {
    const sentence = summarise(
      diff({ same_technique: false, same_model: true }),
      run({ technique: "standard-rag" }),
      run({ technique: "fusion-rag" }),
      false,
    );

    expect(sentence).toContain("Different techniques on the same model");
    expect(sentence).toContain("standard-rag");
    expect(sentence).toContain("fusion-rag");
    // The model is the controlled variable here, so naming it would be noise.
    expect(sentence).not.toContain("qwen3:8b");
  });

  it("pairs each technique with its own model when both axes vary", () => {
    const sentence = summarise(
      diff({ same_technique: false, same_model: false }),
      run({ technique: "standard-rag" }),
      run({
        technique: "fusion-rag",
        metadata: { ...run().metadata, model: "claude-haiku-4-5" },
      }),
      false,
    );

    expect(sentence).toContain("standard-rag/qwen3:8b");
    expect(sentence).toContain("fusion-rag/claude-haiku-4-5");
  });
});

// --- The lesson clause: what the reader is told actually differed -----------

describe("summarise — the lesson clause", () => {
  it("blames the model's own variation when the same technique and model saw identical evidence", () => {
    const sentence = summarise(
      diff({ chunk_overlap_pct: 100, chunk_overlap: 2 }),
      run(),
      run(),
      false,
    );

    expect(sentence).toContain("run-to-run variation");
    expect(sentence).not.toContain("retrieval");
  });

  it("blames generation, not retrieval, when different setups saw identical evidence", () => {
    const sentence = summarise(
      diff({
        chunk_overlap_pct: 100,
        chunk_overlap: 2,
        same_technique: false,
        same_model: true,
      }),
      run({ technique: "standard-rag" }),
      run({ technique: "fusion-rag" }),
      false,
    );

    expect(sentence).toContain("identical evidence");
    expect(sentence).toContain("generation, not retrieval");
  });

  it("says the evidence was disjoint when both sides retrieved and shared nothing", () => {
    const sentence = summarise(
      diff({
        chunk_overlap: 0,
        chunk_overlap_pct: 0,
        same_technique: false,
        only_a_chunk_ids: ["dynamo.md#0"],
        only_b_chunk_ids: ["raft.md#3"],
      }),
      run({ technique: "standard-rag", retrieved_chunks: chunks("dynamo.md#0") }),
      run({ technique: "fusion-rag", retrieved_chunks: chunks("raft.md#3") }),
      false,
    );

    expect(sentence).toContain("completely different evidence");
    expect(sentence).toContain("different source passages");
  });

  it("counts the shared chunks when the overlap is partial", () => {
    const sentence = summarise(
      diff({ chunk_overlap: 3, chunk_overlap_pct: 75, same_technique: false }),
      run({ technique: "standard-rag" }),
      run({ technique: "fusion-rag" }),
      false,
    );

    expect(sentence).toContain("agreed on 3 of the retrieved chunks");
    expect(sentence).toContain("retrieval — not just generation");
  });

  it("tells the reader to build the index when neither side retrieved anything", () => {
    const empty = run({ retrieved_chunks: [] });
    const sentence = summarise(
      diff({ chunk_overlap: 0, chunk_overlap_pct: 0 }),
      empty,
      empty,
      hasNoEvidence(empty, empty),
    );

    expect(sentence).toContain("neither side retrieved anything");
    expect(sentence).toContain("make index");
    // The empty-index case must not also be reported as a retrieval disagreement.
    expect(sentence).not.toContain("completely different evidence");
  });
});

// --- The boundaries between those branches ---------------------------------

describe("summarise — branch boundaries", () => {
  it("treats 99.9% as a partial overlap, not identical evidence", () => {
    // The backend rounds to one decimal, so 99.9 is the closest a genuinely
    // different retrieval can get to the identical-evidence branch.
    const sentence = summarise(
      diff({ chunk_overlap_pct: 99.9, chunk_overlap: 999, same_technique: false }),
      run(),
      run(),
      false,
    );

    expect(sentence).not.toContain("identical evidence");
    expect(sentence).toContain("agreed on 999 of the retrieved chunks");
  });

  it("treats a single shared chunk as partial, not as disjoint evidence", () => {
    const sentence = summarise(
      diff({ chunk_overlap: 1, chunk_overlap_pct: 25, same_technique: false }),
      run(),
      run(),
      false,
    );

    expect(sentence).not.toContain("completely different evidence");
    expect(sentence).toContain("agreed on 1 of the retrieved chunks");
  });

  it("lets no-evidence win over the identical-evidence branch", () => {
    // Defensive: 100% and "nothing retrieved" cannot both be true from the real
    // backend (an empty side makes the denominator 0, so pct is 0). This pins
    // the precedence anyway, because the branch order is what enforces it.
    const sentence = summarise(diff({ chunk_overlap_pct: 100 }), run(), run(), true);

    expect(sentence).toContain("neither side retrieved anything");
  });
});

// --- Known bugs ------------------------------------------------------------
//
// `test.fails` asserts that the test currently FAILS. These two encode the
// CORRECT expectation, not the current behaviour: when the bug is fixed the
// suite goes red here, and whoever fixes it deletes the `.fails`. Written this
// way on purpose — writing them to match today's output would bless a sentence
// that is false.

describe("summarise — KNOWN BUGS (see the report)", () => {
  test.fails(
    "BUG: one side retrieving nothing is reported as a retrieval disagreement",
    () => {
      // Reachable today: Graph RAG with no .graph.json returns zero chunks and a
      // boilerplate answer, while any other technique on the same built index
      // returns chunks. `hasNoEvidence` is false (only ONE side is empty), so the
      // overlap-0 branch fires and tells the reader the two answers are "grounded
      // in different source passages". One of them is grounded in nothing at all.
      const sentence = summarise(
        diff({
          chunk_overlap: 0,
          chunk_overlap_pct: 0,
          same_technique: false,
          only_b_chunk_ids: ["dynamo.md#0", "dynamo.md#1"],
        }),
        run({ technique: "graph-rag", retrieved_chunks: [] }),
        run({ technique: "standard-rag" }),
        false,
      );

      expect(sentence).not.toContain("different source passages");
    },
  );

  test.fails("BUG: one empty technique is blamed on an empty index", () => {
    // Same root cause, other side of it: comparing Graph RAG with itself when no
    // graph has been built leaves BOTH sides empty on a perfectly good index, and
    // the reader is told to run `make index` — a command that will not help.
    const noGraph = run({ technique: "graph-rag", retrieved_chunks: [] });
    const sentence = summarise(
      diff({ chunk_overlap: 0, chunk_overlap_pct: 0 }),
      noGraph,
      noGraph,
      hasNoEvidence(noGraph, noGraph),
    );

    expect(sentence).not.toContain("the index is empty");
  });
});

// --- hasNoEvidence ---------------------------------------------------------

describe("hasNoEvidence", () => {
  it("is true only when both sides retrieved nothing", () => {
    const empty = run({ retrieved_chunks: [] });
    const full = run();

    expect(hasNoEvidence(empty, empty)).toBe(true);
    expect(hasNoEvidence(empty, full)).toBe(false);
    expect(hasNoEvidence(full, empty)).toBe(false);
    expect(hasNoEvidence(full, full)).toBe(false);
  });
});

// --- formatDelta -----------------------------------------------------------

describe("formatDelta", () => {
  it("says 'same' for zero rather than printing a signed nothing", () => {
    expect(formatDelta(0, "ms")).toBe("same");
    expect(formatDelta(0, "ms", true)).toBe("same");
  });

  it("signs the direction with a real minus, not a hyphen", () => {
    expect(formatDelta(12, "ms")).toBe("+12ms");
    // U+2212. A hyphen here reads as a list bullet at this font size.
    expect(formatDelta(-12, "ms")).toBe("−12ms");
  });

  it("rounds the magnitude and keeps the sign separate from it", () => {
    expect(formatDelta(-12.4, "ms")).toBe("−12ms");
    expect(formatDelta(12.6, "ms")).toBe("+13ms");
  });

  it("switches to seconds at exactly 1000ms, but only when humanising", () => {
    expect(formatDelta(1000, "ms", true)).toBe("+1.0s");
    expect(formatDelta(999, "ms", true)).toBe("+999ms");
    expect(formatDelta(-1500, "ms", true)).toBe("−1.5s");
    // The LLM-calls and steps columns pass humanise=false: "1.0s calls" would be
    // nonsense, so a big count stays a count.
    expect(formatDelta(1500, "")).toBe("+1500");
  });
});
