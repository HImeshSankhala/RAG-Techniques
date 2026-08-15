import type { ComparisonDiff, RunResponse } from "@/lib/api";

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
  // Nothing retrieved on either side is not "they disagreed completely" — it is
  // an unbuilt index. The overlap arithmetic gives 0% for both, so the two cases
  // have to be told apart here rather than from the number.
  const noEvidence = a.retrieved_chunks.length === 0 && b.retrieved_chunks.length === 0;

  return (
    <section className="rounded-lg border border-slate-200 p-5 dark:border-slate-800">
      <h2 className="text-xs font-medium uppercase tracking-wide text-slate-500">
        What differed
      </h2>

      <p className="mt-3 text-sm leading-relaxed">{summarise(diff, a, b, noEvidence)}</p>

      <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="Evidence overlap"
          value={noEvidence ? "—" : `${diff.chunk_overlap_pct}%`}
          detail={
            noEvidence
              ? "nothing retrieved"
              : `${diff.chunk_overlap} shared chunk${diff.chunk_overlap === 1 ? "" : "s"}`
          }
          tone={!noEvidence && diff.chunk_overlap_pct < 100 ? "notable" : "neutral"}
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
 * A sentence naming what actually varied.
 *
 * Written per case rather than as one generic template: "they retrieved
 * different evidence" and "they retrieved the same evidence and the model
 * differed" are different lessons, and a reader should not have to infer which
 * one they are looking at from four numbers.
 */
function summarise(
  diff: ComparisonDiff,
  a: RunResponse,
  b: RunResponse,
  noEvidence: boolean,
): string {
  const axis = diff.same_technique
    ? diff.same_model
      ? "Both sides ran the same technique on the same model"
      : `Same technique, different models (${a.metadata.model} vs ${b.metadata.model})`
    : diff.same_model
      ? `Different techniques on the same model (${a.technique} vs ${b.technique})`
      : `Different techniques and different models (${a.technique}/${a.metadata.model} vs ${b.technique}/${b.metadata.model})`;

  if (noEvidence) {
    return `${axis}, but neither side retrieved anything — the index is empty. Run \`make index\`, then compare again.`;
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

function formatDelta(value: number, unit: string, humanise = false): string {
  if (value === 0) return "same";
  const sign = value > 0 ? "+" : "−";
  const magnitude = Math.abs(value);
  if (humanise && magnitude >= 1000) return `${sign}${(magnitude / 1000).toFixed(1)}s`;
  return `${sign}${Math.round(magnitude)}${unit}`;
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function money(usd: number): string {
  return usd > 0 ? `$${usd.toFixed(4)}` : "free";
}
