import type { Step } from "@/lib/api";

/**
 * The stage-by-stage trace of a run, with a bar per step proportional to its
 * duration.
 *
 * This is the component the whole `RAGResult.steps` contract exists for. Standard
 * RAG's three steps look unremarkable alone — the point arrives in Phase 5, when
 * this sits beside Multi-Pass's nine and the difference in shape is the lesson.
 * Bars are relative to the slowest step in *this* run, so the visual answers
 * "where did the time go here", not "how does this compare to other runs".
 */
export function StepsTrace({ steps }: { steps: Step[] }) {
  const slowest = Math.max(...steps.map((s) => s.duration_ms), 1);

  return (
    <ol className="space-y-3">
      {steps.map((step, index) => (
        <li key={`${step.name}-${index}`}>
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className="font-medium">
              <span className="mr-2 text-slate-400 tabular-nums">{index + 1}</span>
              {step.name}
            </span>
            <span className="shrink-0 font-mono text-xs text-slate-500 tabular-nums">
              {formatMs(step.duration_ms)}
            </span>
          </div>

          <div
            className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800"
            role="presentation"
          >
            <div
              className="h-full rounded-full bg-slate-400 dark:bg-slate-500"
              style={{ width: `${Math.max((step.duration_ms / slowest) * 100, 2)}%` }}
            />
          </div>

          <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
            {step.detail}
          </p>
        </li>
      ))}
    </ol>
  );
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}
