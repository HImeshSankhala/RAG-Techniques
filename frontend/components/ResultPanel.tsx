import type { Chunk, RunResponse } from "@/lib/api";
import { StepsTrace } from "@/components/StepsTrace";

/**
 * One completed run: the answer, the numbers behind it, the passages it was given,
 * and the trace of how it got there.
 *
 * Built to be embeddable twice side by side — Phase 5's compare view renders two
 * of these — so it takes a whole `RunResponse` and owns no layout width of its own.
 */
export function ResultPanel({ result }: { result: RunResponse }) {
  const { metadata } = result;

  return (
    <div className="space-y-6">
      <section>
        <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">Answer</h3>
        <p className="mt-2 whitespace-pre-wrap leading-relaxed">{result.answer}</p>
      </section>

      <section className="flex flex-wrap gap-2">
        <Badge label="model" value={metadata.model} />
        <Badge label="latency" value={formatMs(metadata.latency_ms)} />
        <Badge label="tokens" value={`${metadata.tokens_in} in / ${metadata.tokens_out} out`} />
        <Badge
          label="cost"
          value={metadata.cost_estimate_usd > 0 ? `$${metadata.cost_estimate_usd.toFixed(4)}` : "free"}
          tone={metadata.cost_estimate_usd > 0 ? "paid" : "free"}
        />
        <Badge
          label="cited sources"
          value={`${Math.round(metadata.groundedness * 100)}%`}
          // A groundedness of 0 means the model cited nothing it was given, which
          // usually means it answered from its own knowledge instead of the context.
          tone={metadata.groundedness === 0 ? "warn" : "neutral"}
        />
      </section>

      <details className="group rounded-lg border border-slate-200 dark:border-slate-800">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium marker:content-none">
          <span className="mr-2 inline-block transition group-open:rotate-90">›</span>
          Retrieved passages ({result.retrieved_chunks.length})
          <span className="ml-2 font-normal text-slate-500">— what the model was shown</span>
        </summary>
        <div className="space-y-3 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
          {result.retrieved_chunks.map((chunk, index) => (
            <ChunkCard key={chunk.chunk_id} chunk={chunk} rank={index + 1} />
          ))}
        </div>
      </details>

      <section>
        <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">Trace</h3>
        <div className="mt-3">
          <StepsTrace steps={result.steps} />
        </div>
      </section>
    </div>
  );
}

function ChunkCard({ chunk, rank }: { chunk: Chunk; rank: number }) {
  return (
    <article className="rounded border border-slate-200 p-3 dark:border-slate-800">
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="font-mono text-slate-500">
          #{rank} · {chunk.chunk_id}
        </span>
        <span className="shrink-0 font-mono text-slate-500 tabular-nums">
          score {chunk.score.toFixed(4)}
        </span>
      </div>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-slate-600 dark:text-slate-400">
        {chunk.text}
      </p>
    </article>
  );
}

function Badge({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "free" | "paid" | "warn";
}) {
  const tones = {
    neutral: "border-slate-200 text-slate-600 dark:border-slate-800 dark:text-slate-400",
    free: "border-emerald-300 text-emerald-700 dark:border-emerald-800 dark:text-emerald-400",
    paid: "border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400",
    warn: "border-red-300 text-red-700 dark:border-red-800 dark:text-red-400",
  };

  return (
    <span className={`rounded-full border px-2.5 py-1 text-xs ${tones[tone]}`}>
      <span className="text-slate-400">{label}</span> <span className="font-medium">{value}</span>
    </span>
  );
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}
