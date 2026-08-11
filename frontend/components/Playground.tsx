"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ResultPanel } from "@/components/ResultPanel";
import {
  ApiError,
  getUsage,
  runTechnique,
  type ModelInfo,
  type RunResponse,
  type Technique,
  type Usage,
} from "@/lib/api";

const PRESET_QUERIES = [
  "How does Dynamo handle conflicting concurrent writes?",
  "What is a memtable in Bigtable?",
  "Why did Raft's authors say they designed it?",
  "How are stragglers handled in MapReduce?",
];

export function Playground({
  techniques,
  models,
}: {
  techniques: Technique[];
  models: ModelInfo[];
}) {
  const runnable = techniques.filter((t) => t.implemented);

  // "Try it in the playground" on a learn page arrives as ?technique=<slug>.
  // Honour it only if that technique can actually run, so a stale link to a
  // docs-only technique lands on a usable selection instead of a dead control.
  const requested = useSearchParams().get("technique");
  const preselected = runnable.find((t) => t.name === requested)?.name;

  const [technique, setTechnique] = useState(preselected ?? runnable[0]?.name ?? "");
  const [model, setModel] = useState(models.find((m) => m.is_default)?.id ?? models[0]?.id ?? "");
  const [query, setQuery] = useState(PRESET_QUERIES[0]);

  const [result, setResult] = useState<RunResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [usage, setUsage] = useState<Usage | null>(null);

  const selectedModel = models.find((m) => m.id === model);

  useEffect(() => {
    // Best-effort: the spend badge is informational, so a failure here must not
    // break the page.
    getUsage().then(setUsage).catch(() => {});
  }, []);

  async function handleRun(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim() || isRunning) return;

    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      setResult(await runTechnique(technique, query.trim(), model || undefined));
    } catch (cause) {
      setError(
        cause instanceof ApiError ? cause : new ApiError("Unexpected error running query.", 0),
      );
    } finally {
      setIsRunning(false);
      getUsage().then(setUsage).catch(() => {});
    }
  }

  return (
    <div className="space-y-8">
      <form onSubmit={handleRun} className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Technique
            </span>
            <select
              value={technique}
              onChange={(e) => setTechnique(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm dark:border-slate-700"
            >
              {techniques.map((t) => (
                // Docs-only techniques stay visible but disabled: seeing what is
                // coming is part of the point, and a missing option would read as
                // a bug rather than a roadmap.
                <option key={t.name} value={t.name} disabled={!t.implemented}>
                  {t.display_name}
                  {t.implemented ? "" : " — not built yet"}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Model
            </span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm dark:border-slate-700"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.display_name}
                  {m.is_paid ? " — paid" : " — free"}
                  {m.available ? "" : " (not configured)"}
                </option>
              ))}
            </select>
          </label>
        </div>

        {selectedModel && (
          <p className="text-xs text-slate-500">
            {selectedModel.note}
            {selectedModel.is_paid && " Charged to your Anthropic credit."}
          </p>
        )}

        <label className="block">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Query</span>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            placeholder="Ask something about the indexed documents…"
            className="mt-1 w-full resize-y rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm leading-relaxed dark:border-slate-700"
          />
        </label>

        <div className="flex flex-wrap gap-2">
          {PRESET_QUERIES.map((preset) => (
            <button
              key={preset}
              type="button"
              onClick={() => setQuery(preset)}
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 transition hover:border-slate-400 dark:border-slate-800 dark:text-slate-400 dark:hover:border-slate-600"
            >
              {preset.length > 42 ? `${preset.slice(0, 42)}…` : preset}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-4">
          <button
            type="submit"
            disabled={isRunning || !query.trim() || !technique}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
          >
            {isRunning ? "Running…" : "Run"}
          </button>

          {isRunning && <ElapsedTimer />}

          {usage && usage.calls > 0 && (
            <span className="ml-auto text-xs text-slate-500">
              Anthropic spend estimate:{" "}
              <span className="font-mono">${usage.spend_estimate_usd.toFixed(4)}</span> over{" "}
              {usage.calls} call{usage.calls === 1 ? "" : "s"}
            </span>
          )}
        </div>
      </form>

      {error && <ErrorNotice error={error} />}

      {result && (
        <section className="rounded-lg border border-slate-200 p-5 dark:border-slate-800">
          <ResultPanel result={result} />
        </section>
      )}
    </div>
  );
}

/**
 * A local model answers in ~15 seconds on a laptop. Without a visible clock that
 * reads as a hung page, so the elapsed count is doing real work here.
 */
function ElapsedTimer() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    // Read the clock inside the effect, not during render: render must be pure,
    // and this component only exists while a run is in flight anyway.
    const startedAt = Date.now();
    const id = setInterval(() => setSeconds((Date.now() - startedAt) / 1000), 100);
    return () => clearInterval(id);
  }, []);

  return (
    <span className="font-mono text-xs text-slate-500 tabular-nums">
      {seconds.toFixed(1)}s — local models take ~15s
    </span>
  );
}

function ErrorNotice({ error }: { error: ApiError }) {
  // The backend already returns an actionable sentence per status; this only adds
  // the one piece the server cannot know — what the user should do in this UI.
  const hint =
    error.status === 503
      ? "Add ANTHROPIC_API_KEY to backend/.env, or switch the model back to the local one."
      : error.status === 429
        ? "The paid-call safety cap was hit. Restart the backend to reset, or use the local model."
        : error.status === 0
          ? "Start the backend with `make dev`, then try again."
          : null;

  return (
    <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40">
      <p className="text-sm font-medium text-amber-900 dark:text-amber-200">
        Could not complete the run{error.status ? ` (${error.status})` : ""}
      </p>
      <p className="mt-1 text-sm text-amber-800 dark:text-amber-300">{error.message}</p>
      {hint && <p className="mt-2 text-sm text-amber-800 dark:text-amber-300">{hint}</p>}
    </div>
  );
}
