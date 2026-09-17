import type { ComparisonDiff, RunResponse } from "@/lib/api";
import { formatMs, money } from "@/lib/format";

/**
 * The row that turns two results into a comparison.
 *
 * Two answers side by side mostly look alike — both are fluent paragraphs about
 * the same topic. The differences that matter are in the evidence and the cost,
 * and those are numbers rather than prose. This row is where the lesson actually
 * lands, which is why the diff is computed server-side and shipped as part of the
 * API contract rather than recomputed per client.
 */
export function DiffSummary({
  diff,
  a,
  b,
}: {
  diff: ComparisonDiff;
  a: RunResponse;
  b: RunResponse;
}) {
  const empty = emptySideCount(a, b);

  return (
    <section className="rounded-lg border border-slate-200 p-5 dark:border-slate-800">
      <h2 className="text-xs font-medium uppercase tracking-wide text-slate-500">
        What differed
      </h2>

      <p className="mt-3 text-sm leading-relaxed">{summarise(diff, a, b, empty)}</p>

      <div className="mt-5 grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
        {/* A percentage needs both sides to have retrieved something. The server
            divides by the LARGER side, so one empty side pins this at 0% — a
            number that looks like total disagreement and actually means "there
            was nothing to agree with". Either empty case shows an em dash
            instead, and neither is highlighted: nothing diverged. */}
        <Metric
          label="Evidence overlap"
          value={empty > 0 ? "—" : `${diff.chunk_overlap_pct}%`}
          detail={
            empty === 2
              ? "nothing retrieved"
              : empty === 1
                ? "one side retrieved nothing"
                : `${diff.chunk_overlap} shared chunk${diff.chunk_overlap === 1 ? "" : "s"}`
          }
          tone={empty === 0 && diff.chunk_overlap_pct < 100 ? "notable" : "neutral"}
        />
        <Metric
          label="Latency"
          value={formatDelta(diff.latency_delta_ms, "ms", true)}
          detail={`${formatMs(a.metadata.latency_ms)} vs ${formatMs(b.metadata.latency_ms)}`}
        />
        <Metric
          label="LLM calls"
          value={diff.llm_calls_delta === 0 ? "same" : formatDelta(diff.llm_calls_delta, "")}
          detail={`${a.metadata.llm_calls} vs ${b.metadata.llm_calls}`}
        />
        {/* The shape of the run, not its cost. Two techniques can make the same
            number of LLM calls and still differ here — this is the column where
            Multi-Pass's loop is visible as structure rather than as latency. */}
        <Metric
          label="Steps"
          value={diff.steps_delta === 0 ? "same" : formatDelta(diff.steps_delta, "")}
          detail={`${a.steps.length} vs ${b.steps.length}`}
        />
        <Metric
          label="Cost"
          value={
            diff.cost_delta_usd === 0
              ? "same"
              : `${diff.cost_delta_usd > 0 ? "+" : ""}$${diff.cost_delta_usd.toFixed(4)}`
          }
          detail={`${money(a.metadata.cost_estimate_usd)} vs ${money(b.metadata.cost_estimate_usd)}`}
          tone={diff.cost_delta_usd !== 0 ? "notable" : "neutral"}
        />
      </div>

      {(diff.only_a_chunk_ids.length > 0 || diff.only_b_chunk_ids.length > 0) && (
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <ChunkList label="Only A retrieved" ids={diff.only_a_chunk_ids} />
          <ChunkList label="Only B retrieved" ids={diff.only_b_chunk_ids} />
        </div>
      )}
    </section>
  );
}

/**
 * Every pipeline's own name for "the collection was empty". Anything else on a
 * side that retrieved nothing means the index may be perfectly fine and *this
 * technique* could not run — a missing knowledge graph, say.
 *
 * Reading the reason the backend already reports is what keeps this file from
 * having to know which technique needs which setup step. The frontend knows
 * "the index is the problem" vs "it isn't"; the pipeline that knows why puts the
 * actual remedy in its own answer, which renders below.
 */
const EMPTY_INDEX = "empty_index";

/**
 * How many of the two sides retrieved nothing: 0, 1 or 2.
 *
 * This used to be a boolean `a.length === 0 && b.length === 0`, and that `&&`
 * was a bug. It collapsed "one side produced no evidence" into the same bucket
 * as "both sides retrieved fine", so a side that never ran was then described by
 * the overlap-0 branch as a side that had *disagreed*. Three situations, three
 * different truths — so the count, not a boolean.
 *
 * Exported because the sentence and the Evidence-overlap tile must agree about
 * it; the failure mode is those two telling different stories about one run.
 */
export function emptySideCount(a: RunResponse, b: RunResponse): number {
  return [a, b].filter((side) => side.retrieved_chunks.length === 0).length;
}

/** The pipelines' own words for why nothing came back, deduped. "" if they said nothing. */
function reasons(sides: RunResponse[]): string {
  const given = sides.map((side) => side.metadata.termination_reason).filter(Boolean);
  return [...new Set(given)].join(", ");
}

/** ` (no_graph)`, or nothing at all when no side gave a reason. */
function reasonNote(sides: RunResponse[]): string {
  const note = reasons(sides);
  return note ? ` (${note})` : "";
}

/**
 * A sentence naming what actually varied, plus the feedback caveat when a side
 * was reranked by stored votes.
 *
 * Written per case rather than as one generic template: "they retrieved
 * different evidence" and "they retrieved the same evidence and the model
 * differed" are different lessons, and a reader should not have to infer which
 * one they are looking at from four numbers.
 */
export function summarise(
  diff: ComparisonDiff,
  a: RunResponse,
  b: RunResponse,
  empty: number,
): string {
  return `${comparison(diff, a, b, empty)}${feedbackNote(a, b)}`;
}

/**
 * The caveat a feedback-reranked side needs, or "" when neither side has votes.
 *
 * Without it the row above is true but incomplete in the most misleading way:
 * every other technique is a function of its query, so "they retrieved different
 * evidence" reads as "these two techniques disagree". A side whose ranking was
 * shifted by stored votes did not disagree about this query — it is carrying the
 * history of earlier ones, and re-running the same comparison after more votes
 * can give a different answer.
 */
function feedbackNote(a: RunResponse, b: RunResponse): string {
  const sides = [
    { label: "A", run: a },
    { label: "B", run: b },
  ].filter((side) => side.run.metadata.feedback_votes > 0);

  if (sides.length === 0) return "";

  const applied = sides
    .map(
      (side) =>
        `${side.label} (${side.run.technique}) applied ${side.run.metadata.feedback_votes} stored ` +
        `vote${side.run.metadata.feedback_votes === 1 ? "" : "s"}`,
    )
    .join(" and ");

  return ` ${applied} to its ranking, so this comparison depends on feedback history as well as the query — the same two techniques on the same query gave a different result before those votes were cast.`;
}

function comparison(
  diff: ComparisonDiff,
  a: RunResponse,
  b: RunResponse,
  empty: number,
): string {
  const axis = diff.same_technique
    ? diff.same_model
      ? "Both sides ran the same technique on the same model"
      : `Same technique, different models (${a.metadata.model} vs ${b.metadata.model})`
    : diff.same_model
      ? `Different techniques on the same model (${a.technique} vs ${b.technique})`
      : `Different techniques and different models (${a.technique}/${a.metadata.model} vs ${b.technique}/${b.metadata.model})`;

  // Both empty. Only the pipelines can say whether the index is why — every one
  // of them reports `empty_index` from its own no-chunks branch, so if neither
  // said that, `make index` is the wrong advice and the corpus is probably fine.
  if (empty === 2) {
    return [a, b].every((side) => side.metadata.termination_reason === EMPTY_INDEX)
      ? `${axis}, but neither side retrieved anything — the index is empty. Run \`make index\`, then compare again.`
      : `${axis}, but neither side could run${reasonNote([a, b])}, so there is nothing to compare. The index is not what stopped them — each answer below says what it still needs.`;
  }

  // Exactly one empty. This is NOT a retrieval disagreement, and calling it one
  // is the most damaging thing this row can say: it reads as a real finding.
  if (empty === 1) {
    const [silent, other] = a.retrieved_chunks.length === 0 ? [a, b] : [b, a];
    const label = silent === a ? "A" : "B";
    return `${axis}. ${label} (${silent.technique}) retrieved nothing at all${reasonNote([silent])}, so there is no evidence disagreement here to read — only ${label === "A" ? "B" : "A"} (${other.technique}) retrieved anything. ${label}'s answer below says what it needs.`;
  }

  if (diff.chunk_overlap_pct === 100) {
    return diff.same_technique && diff.same_model
      ? `${axis}, so any difference in the answers is the model's own run-to-run variation.`
      : `${axis}. They retrieved identical evidence, so any difference in the answers came from generation, not retrieval.`;
  }

  if (diff.chunk_overlap === 0) {
    return `${axis}. They retrieved completely different evidence — no shared chunks — so the two answers are grounded in different source passages.`;
  }

  return `${axis}. They agreed on ${diff.chunk_overlap} of the retrieved chunks and differed on the rest, so retrieval — not just generation — is part of why the answers differ.`;
}

function ChunkList({ label, ids }: { label: string; ids: string[] }) {
  return (
    <div>
      <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</h3>
      {ids.length === 0 ? (
        <p className="mt-1 text-sm text-slate-500">—</p>
      ) : (
        <ul className="mt-1 space-y-0.5">
          {ids.map((id) => (
            <li key={id} className="font-mono text-xs text-slate-600 dark:text-slate-400">
              {id}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "neutral" | "notable";
}) {
  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div
        className={`mt-1 text-lg font-semibold tabular-nums ${
          tone === "notable" ? "text-amber-600 dark:text-amber-400" : ""
        }`}
      >
        {value}
      </div>
      <div className="text-xs text-slate-500">{detail}</div>
    </div>
  );
}

export function formatDelta(value: number, unit: string, humanise = false): string {
  if (value === 0) return "same";
  const sign = value > 0 ? "+" : "−";
  const magnitude = Math.abs(value);
  if (humanise && magnitude >= 1000) return `${sign}${(magnitude / 1000).toFixed(1)}s`;
  return `${sign}${Math.round(magnitude)}${unit}`;
}
