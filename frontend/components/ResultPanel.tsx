"use client";

import { useState } from "react";
import type { Chunk, RunResponse } from "@/lib/api";
import { formatMs, money } from "@/lib/format";
import { StepsTrace } from "@/components/StepsTrace";

/** Rate one passage. Resolves when the vote is stored, rejects with the API error. */
export type RateChunk = (chunkId: string, rating: 1 | -1) => Promise<void>;

/**
 * One completed run: the answer, the numbers behind it, the passages it was given,
 * and the trace of how it got there.
 *
 * Built to be embeddable twice side by side — Phase 5's compare view renders two
 * of these — so it takes a whole `RunResponse` and owns no layout width of its own.
 *
 * `onRate` adds thumbs to each passage. It is passed only by the playground, and
 * only for Feedback RAG: votes rerank *that* technique's future runs, so offering
 * them under another technique's answer would change something the reader was not
 * looking at. The compare view never passes it — voting mid-comparison would move
 * one side's ranking while the two are being read against each other.
 */
export function ResultPanel({ result, onRate }: { result: RunResponse; onRate?: RateChunk }) {
  const { metadata } = result;

  return (
    <div className="space-y-6">
      <section>
        <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">Answer</h3>
        <p className="mt-2 whitespace-pre-wrap leading-relaxed">{result.answer}</p>
      </section>

      <section className="flex flex-wrap gap-2">
        {/* Which corpus answered. Shown on every run, not only in compare's
            summary: an answer sourced from an uploaded file reads exactly like
            one from the demo corpus, and the reader cannot tell them apart. */}
        <Badge label="corpus" value={result.corpus} />
        <Badge label="model" value={metadata.model} />
        <Badge label="latency" value={formatMs(metadata.latency_ms)} />
        <Badge label="tokens" value={`${metadata.tokens_in} in / ${metadata.tokens_out} out`} />
        <Badge
          label="cost"
          value={money(metadata.cost_estimate_usd)}
          tone={metadata.cost_estimate_usd > 0 ? "paid" : "free"}
        />
        <Badge
          label="cited sources"
          value={`${Math.round(metadata.groundedness * 100)}%`}
          // A groundedness of 0 means the model cited nothing it was given, which
          // usually means it answered from its own knowledge instead of the context.
          tone={metadata.groundedness === 0 ? "warn" : "neutral"}
        />
        {/* Multi-Pass's entire signal. Every pipeline has populated these since
            Phase 1 and nothing rendered them, so the technique whose whole point
            is looping looked, in the UI, exactly like the ones that do not. */}
        <Badge label="passes" value={String(metadata.retrieval_passes)} />
        <Badge label="stopped" value={metadata.termination_reason} />
      </section>

      <details className="group rounded-lg border border-slate-200 dark:border-slate-800">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium marker:content-none">
          <span className="mr-2 inline-block transition group-open:rotate-90">›</span>
          Retrieved passages ({result.retrieved_chunks.length})
          <span className="ml-2 font-normal text-slate-500">— what the model was shown</span>
        </summary>
        <div className="space-y-3 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
          {result.retrieved_chunks.map((chunk, index) => (
            <ChunkCard key={chunk.chunk_id} chunk={chunk} rank={index + 1} onRate={onRate} />
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

function ChunkCard({
  chunk,
  rank,
  onRate,
}: {
  chunk: Chunk;
  rank: number;
  onRate?: RateChunk;
}) {
  // Not optimistic: a vote that failed to store must not be shown as stored, or
  // the reader waits for an effect on the next run that can never arrive.
  //
  // The status is held WITH the chunk object it was recorded against. A new run
  // hands this card a new object while React keeps the instance (the list is
  // keyed by chunk_id), so tying the two together is what stops "stored — run
  // again to see it move" from outliving the run it asked for.
  const [vote, setVote] = useState<{
    chunk: Chunk;
    status: "sending" | "sent" | "failed";
    failure: string;
  } | null>(null);
  const current = vote?.chunk === chunk ? vote : null;
  const status = current?.status ?? "idle";

  async function rate(rating: 1 | -1) {
    if (!onRate) return;
    setVote({ chunk, status: "sending", failure: "" });
    try {
      await onRate(chunk.chunk_id, rating);
      setVote({ chunk, status: "sent", failure: "" });
    } catch (error) {
      // The server's own words: the likeliest failure is a 422 saying the index
      // was rebuilt and the query has to be run again, which "try again" would
      // flatly contradict — re-clicking fails identically.
      setVote({
        chunk,
        status: "failed",
        failure: error instanceof Error ? error.message : "try again",
      });
    }
  }

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
      {onRate && (
        <div className="mt-3 flex items-center gap-2 text-xs">
          <button
            type="button"
            onClick={() => rate(1)}
            disabled={status === "sending"}
            className="rounded border border-slate-200 px-2 py-1 transition hover:border-slate-400 disabled:opacity-40 dark:border-slate-800 dark:hover:border-slate-600"
          >
            👍 helpful
          </button>
          <button
            type="button"
            onClick={() => rate(-1)}
            disabled={status === "sending"}
            className="rounded border border-slate-200 px-2 py-1 transition hover:border-slate-400 disabled:opacity-40 dark:border-slate-800 dark:hover:border-slate-600"
          >
            👎 not helpful
          </button>
          <span className="text-slate-500">
            {status === "sent"
              ? "stored — run again to see it move. Every click counts."
              : status === "failed"
                ? `not stored: ${current?.failure ?? "try again"}`
                : "stored votes rerank future runs of this technique"}
          </span>
        </div>
      )}
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
