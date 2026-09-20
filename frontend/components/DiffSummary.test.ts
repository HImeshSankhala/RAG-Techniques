/**
 * The branch that decides which lesson the compare page teaches.
 *
 * `summarise()` picks one of seven sentences. Each one is a different claim about
 * why two answers differ — "retrieval differed", "generation differed", "nothing
 * is indexed", "one side never ran". Pick the wrong branch and the page states
 * something false with full confidence, and nothing in the type system objects:
 * every branch returns a string, and both sides of `if (overlap === 0)` typecheck.
 *
 * So the assertions here are about MEANING, not wording. They check which claim
 * the sentence makes (does it say retrieval, or generation, or empty index), not
 * its exact prose, so that rewording the copy does not break the suite while
 * flipping a comparison still does.
 */

import { describe, expect, it } from "vitest";

import { emptySideCount, formatDelta, summarise } from "@/components/DiffSummary";
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
    draft_id: null,
    corpus: "demo corpus",
    ...overrides,
    metadata: {
      model: "qwen3:8b",
      backend: "ollama",
      latency_ms: 900,
      llm_calls: 1,
      retrieval_passes: 1,
      tokens_in: 1000,
      tokens_out: 200,
      termination_reason: "single_pass",
      groundedness: 1,
      cost_estimate_usd: 0,
      feedback_votes: 0,
      ...overrides.metadata,
    },
  };
}

/** A side that retrieved nothing, carrying the pipeline's own reason for it. */
function silentRun(technique: string, termination_reason: string): RunResponse {
  const base = run({ technique, retrieved_chunks: [] });
  return { ...base, metadata: { ...base.metadata, termination_reason } };
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
    const sentence = summarise(diff({ same_technique: true, same_model: true }), run(), run(), 0);

    expect(sentence).toContain("same technique on the same model");
  });

  it("names both models when only the model varies", () => {
    const sentence = summarise(
      diff({ same_technique: true, same_model: false }),
      run({ metadata: { ...run().metadata, model: "qwen3:8b" } }),
      run({ metadata: { ...run().metadata, model: "claude-haiku-4-5" } }),
      0,
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
      0,
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
      0,
    );

    expect(sentence).toContain("standard-rag/qwen3:8b");
    expect(sentence).toContain("fusion-rag/claude-haiku-4-5");
  });
});

// --- The lesson clause: what the reader is told actually differed -----------

describe("summarise — the lesson clause", () => {
  it("blames the model's own variation when the same technique and model saw identical evidence", () => {
    const sentence = summarise(diff({ chunk_overlap_pct: 100, chunk_overlap: 2 }), run(), run(), 0);

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
      0,
    );

    expect(sentence).toContain("identical evidence");
    expect(sentence).toContain("generation, not retrieval");
    // The demo corpus is the default and goes unnamed: naming it in every
    // sentence is noise, and the reader on the default path never chose it.
    expect(sentence).not.toContain(" on demo corpus");
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
      0,
    );

    expect(sentence).toContain("completely different evidence");
    expect(sentence).toContain("different source passages");
  });

  it("counts the shared chunks when the overlap is partial", () => {
    const sentence = summarise(
      diff({ chunk_overlap: 3, chunk_overlap_pct: 75, same_technique: false }),
      run({ technique: "standard-rag" }),
      run({ technique: "fusion-rag" }),
      0,
    );

    expect(sentence).toContain("agreed on 3 of the retrieved chunks");
    expect(sentence).toContain("retrieval — not just generation");
  });
});

// --- No evidence: three situations that used to be two ---------------------
//
// These are the regression tests for the bug that shipped. `noEvidence` was
// `a.length === 0 && b.length === 0`, so only the both-empty case was special;
// a single empty side fell through to the overlap-0 branch and was reported as
// a retrieval disagreement. Each of the three now says something different, and
// only one of them mentions `make index`.

describe("summarise — when a side retrieved nothing", () => {
  it("tells the reader to build the index when BOTH sides blame an empty index", () => {
    const empty = silentRun("standard-rag", "empty_index");
    const sentence = summarise(
      diff({ chunk_overlap: 0, chunk_overlap_pct: 0 }),
      empty,
      empty,
      emptySideCount(empty, empty),
    );

    expect(sentence).toContain("neither side retrieved anything");
    expect(sentence).toContain("make index");
    expect(sentence).not.toContain("completely different evidence");
  });

  it("does NOT blame the index when both sides are empty for another reason", () => {
    // Graph RAG compared with itself and no .graph.json: the corpus is fine, and
    // `make index` is advice that cannot help.
    const noGraph = silentRun("graph-rag", "no_graph");
    const sentence = summarise(
      diff({ chunk_overlap: 0, chunk_overlap_pct: 0 }),
      noGraph,
      noGraph,
      emptySideCount(noGraph, noGraph),
    );

    expect(sentence).not.toContain("the index is empty");
    expect(sentence).not.toContain("make index");
    expect(sentence).toContain("The index is not what stopped them");
    // The pipeline's own reason is surfaced rather than translated here, so a
    // technique added later needs no edit to this file.
    expect(sentence).toContain("no_graph");
  });

  it("does NOT call one empty side a retrieval disagreement", () => {
    // The bug that was live in the UI: Graph RAG with no graph against any
    // working technique. `grounded in different source passages` is false — one
    // side is grounded in nothing.
    const sentence = summarise(
      diff({
        chunk_overlap: 0,
        chunk_overlap_pct: 0,
        same_technique: false,
        only_b_chunk_ids: ["dynamo.md#0", "dynamo.md#1"],
      }),
      silentRun("graph-rag", "no_graph"),
      run({ technique: "standard-rag" }),
      1,
    );

    expect(sentence).not.toContain("different source passages");
    expect(sentence).not.toContain("completely different evidence");
    expect(sentence).toContain("no evidence disagreement");
    expect(sentence).toContain("A (graph-rag) retrieved nothing at all (no_graph)");
    expect(sentence).toContain("only B (standard-rag) retrieved anything");
  });

  it("names the correct side when it is B that retrieved nothing", () => {
    const sentence = summarise(
      diff({ chunk_overlap: 0, chunk_overlap_pct: 0, same_technique: false }),
      run({ technique: "standard-rag" }),
      silentRun("graph-rag", "no_graph"),
      1,
    );

    expect(sentence).toContain("B (graph-rag) retrieved nothing at all");
    expect(sentence).toContain("only A (standard-rag) retrieved anything");
  });

  it("omits the parenthetical when the pipeline gave no reason", () => {
    const sentence = summarise(
      diff({ chunk_overlap: 0, chunk_overlap_pct: 0, same_technique: false }),
      silentRun("graph-rag", ""),
      run(),
      1,
    );

    expect(sentence).toContain("retrieved nothing at all,");
    expect(sentence).not.toContain("()");
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
      0,
    );

    expect(sentence).not.toContain("identical evidence");
    expect(sentence).toContain("agreed on 999 of the retrieved chunks");
  });

  it("treats a single shared chunk as partial, not as disjoint evidence", () => {
    const sentence = summarise(
      diff({ chunk_overlap: 1, chunk_overlap_pct: 25, same_technique: false }),
      run(),
      run(),
      0,
    );

    expect(sentence).not.toContain("completely different evidence");
    expect(sentence).toContain("agreed on 1 of the retrieved chunks");
  });

  it("lets an empty side win over the identical-evidence branch", () => {
    // Defensive: 100% and "nothing retrieved" cannot both be true from the real
    // backend (an empty side makes the denominator 0, so pct is 0). This pins
    // the precedence anyway, because the branch order is what enforces it.
    const empty = silentRun("standard-rag", "empty_index");

    expect(summarise(diff({ chunk_overlap_pct: 100 }), empty, empty, 2)).toContain(
      "neither side retrieved anything",
    );
    expect(summarise(diff({ chunk_overlap_pct: 100 }), empty, run(), 1)).toContain(
      "retrieved nothing at all",
    );
  });
});

// --- the feedback caveat ---------------------------------------------------

describe("summarise — a side reranked by stored votes", () => {
  /** Feedback RAG, reporting `feedback_votes` stored votes matched to its candidates. */
  function voted(votes: number): RunResponse {
    const base = run({ technique: "feedback-rag" });
    return { ...base, metadata: { ...base.metadata, feedback_votes: votes } };
  }

  it("says nothing extra when neither side used stored votes", () => {
    const sentence = summarise(diff({ same_technique: false }), run(), run(), 0);

    expect(sentence).not.toContain("stored vote");
    expect(sentence).not.toContain("feedback history");
  });

  it("warns that the result depends on history when a side counted votes", () => {
    // Without this clause the row says "they retrieved different evidence",
    // which reads as a disagreement about THIS query. It is not: one side is
    // carrying judgments cast on earlier ones.
    const sentence = summarise(
      diff({ chunk_overlap: 1, chunk_overlap_pct: 25, same_technique: false }),
      run(),
      voted(6),
      0,
    );

    expect(sentence).toContain("B (feedback-rag) counted 6 stored votes");
    expect(sentence).toContain("feedback history");
  });

  it("names both sides when both counted votes, and keeps the singular honest", () => {
    const sentence = summarise(diff(), voted(1), voted(3), 0);

    expect(sentence).toContain("A (feedback-rag) counted 1 stored vote");
    expect(sentence).not.toContain("counted 1 stored votes");
    expect(sentence).toContain("B (feedback-rag) counted 3 stored votes");
  });

  it("does NOT claim the votes changed anything", () => {
    // `feedback_votes` counts matched rows, not rows that moved a passage. An
    // upvote on what was already first, a downvote on the last candidate, or a
    // +1 and a -1 that cancel all leave the ranking exactly as the retriever
    // had it — and this sentence fires on all three. Claiming the comparison
    // "gave a different result before those votes" would be false in every one
    // of them, so the claim is forward-looking and the trace is where a reader
    // finds out what actually moved.
    const sentence = summarise(diff({ same_technique: false }), run(), voted(2), 0);

    expect(sentence).not.toContain("gave a different result");
    expect(sentence).toContain("can come out differently");
    expect(sentence).toContain("trace");
  });

  it("keeps the caveat when the two sides retrieved identical evidence", () => {
    // 100% overlap here does not mean feedback is absent — it can mean the
    // votes have not moved anything yet, and the next vote will.
    const sentence = summarise(diff({ same_technique: false }), run(), voted(2), 0);

    expect(sentence).toContain("identical evidence");
    expect(sentence).toContain("counted 2 stored votes");
  });
});

// --- emptySideCount --------------------------------------------------------

describe("emptySideCount", () => {
  it("distinguishes none, one and both — the `&&` that caused the bug could not", () => {
    const empty = run({ retrieved_chunks: [] });
    const full = run();

    expect(emptySideCount(full, full)).toBe(0);
    expect(emptySideCount(empty, full)).toBe(1);
    expect(emptySideCount(full, empty)).toBe(1);
    expect(emptySideCount(empty, empty)).toBe(2);
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

// --- The null result on somebody else's corpus -----------------------------

describe("summarise — an uploaded corpus that did not diverge", () => {
  const identical = diff({
    chunk_overlap_pct: 100,
    chunk_overlap: 4,
    same_technique: false,
    same_model: true,
  });
  const uploaded = (technique: string) => run({ technique, corpus: "notes.pdf" });

  it("names the corpus that produced the identical result", () => {
    // The likeliest outcome of the whole upload feature, and the one a reader is
    // most likely to read as "the techniques don't matter". It has to be stated
    // as a measurement of a named corpus, not left as a shrug.
    const sentence = summarise(identical, uploaded("standard-rag"), uploaded("fusion-rag"), 0);

    expect(sentence).toContain("notes.pdf");
    expect(sentence).toContain("identical evidence");
    expect(sentence).toContain("4 passages");
  });

  it("does not claim WHY the two techniques agreed", () => {
    // `chunk_overlap_pct` is computed from retrieved ids alone. It cannot tell a
    // small corpus from a query whose terms are everywhere from two rankings that
    // happened to coincide — and a sentence that picks one is inventing a cause
    // the number does not carry. This is the assertion that stops the next
    // rewrite from "explaining" the null result.
    const sentence = summarise(identical, uploaded("standard-rag"), uploaded("fusion-rag"), 0);

    for (const cause of ["too small", "because", "not enough", "only "]) {
      expect(sentence).not.toContain(cause);
    }
  });

  it("still says nothing diverged when one passage was retrieved, not four", () => {
    // Plural agreement is part of the claim: "the same 1 passages" reads as a
    // formatting bug and undermines a sentence whose whole job is to be believed.
    const sentence = summarise(
      diff({ chunk_overlap_pct: 100, chunk_overlap: 1, same_technique: false, same_model: true }),
      uploaded("standard-rag"),
      uploaded("auto-rag"),
      0,
    );

    expect(sentence).toContain("1 passage on notes.pdf");
  });
});
