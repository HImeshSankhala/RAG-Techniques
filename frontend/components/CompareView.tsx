"use client";

import { useEffect, useState } from "react";
import { DiffSummary } from "@/components/DiffSummary";
import { ResultPanel } from "@/components/ResultPanel";
import {
  ApiError,
  compareTechniques,
  getUsage,
  type CompareResponse,
  type ModelInfo,
  type Technique,
  type Usage,
} from "@/lib/api";

/**
 * Preset queries chosen because they *are* the lesson.
 *
 * The first three are measured cases where dense retrieval returns the wrong
 * document and BM25 does not — running Standard vs Fusion on them shows the two
 * techniques retrieving different evidence. The last is the opposite control: a
 * paraphrased question with no rare tokens, where both retrievers agree and the
 * comparison correctly shows almost no difference.
 */
const PRESETS = [
  { query: "What is hinted handoff?", note: "dense lands on raft.md; BM25 on dynamo.md" },
  { query: "What is Chubby used for?", note: "dense lands on mapreduce.md" },
  { query: "What are reversed hostnames used for?", note: "dense misses bigtable.md" },
  {
    query: "How does Dynamo handle conflicting concurrent writes?",
    note: "control — a paraphrased question; both retrievers lead with the right document",
  },
  {
    query:
      "How does Dynamo achieve high write availability, and how does Raft handle log compaction?",
    note: "two-part question — Multi-Pass loops for the half its first search missed (3 steps vs 8)",
  },
];

export function CompareView({
  techniques,
  models,
}: {
  techniques: Technique[];
  models: ModelInfo[];
}) {
  const runnable = techniques.filter((t) => t.implemented);
  const defaultModel = models.find((m) => m.is_default)?.id ?? models[0]?.id ?? "";

  const [query, setQuery] = useState(PRESETS[0].query);
  const [aTechnique, setATechnique] = useState(runnable[0]?.name ?? "");
  const [bTechnique, setBTechnique] = useState(runnable[1]?.name ?? runnable[0]?.name ?? "");
  const [aModel, setAModel] = useState(defaultModel);
  const [bModel, setBModel] = useState(defaultModel);

  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [usage, setUsage] = useState<Usage | null>(null);

  useEffect(() => {
    getUsage().then(setUsage).catch(() => {});
  }, []);

  const bothLocal = [aModel, bModel].every(
    (id) => models.find((m) => m.id === id)?.backend === "ollama",
  );

  async function handleCompare(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim() || isRunning) return;

    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      setResult(
        await compareTechniques(
          query.trim(),
          { technique: aTechnique, model: aModel || null },
          { technique: bTechnique, model: bModel || null },
        ),
      );
    } catch (cause) {
      setError(cause instanceof ApiError ? cause : new ApiError("Unexpected error.", 0));
    } finally {
      setIsRunning(false);
      getUsage().then(setUsage).catch(() => {});
    }
  }

  return (
    <div className="space-y-8">
      <form onSubmit={handleCompare} className="space-y-5">
        <div className="grid gap-5 lg:grid-cols-2">
          <SidePicker
            label="A"
            techniques={techniques}
            models={models}
            technique={aTechnique}
            model={aModel}
            onTechnique={setATechnique}
            onModel={setAModel}
          />
          <SidePicker
            label="B"
            techniques={techniques}
            models={models}
            technique={bTechnique}
            model={bModel}
            onTechnique={setBTechnique}
            onModel={setBModel}
          />
        </div>

        <label className="block">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Query — both sides answer this
          </span>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={2}
            className="mt-1 w-full resize-y rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm leading-relaxed dark:border-slate-700"
          />
        </label>

        <div className="flex flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.query}
              type="button"
              onClick={() => setQuery(preset.query)}
              title={preset.note}
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 transition hover:border-slate-400 dark:border-slate-800 dark:text-slate-400 dark:hover:border-slate-600"
            >
              {preset.query.length > 40 ? `${preset.query.slice(0, 40)}…` : preset.query}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <button
            type="submit"
            disabled={isRunning || !query.trim()}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
          >
            {isRunning ? "Running both…" : "Compare"}
          </button>

          {isRunning && (
            <ElapsedTimer
              // No estimate in seconds. It was "~30s" when the slowest technique
              // was Fusion; Multi-Pass loops to ~45s on one side alone, and any
              // hardcoded number goes stale the next time a technique lands. A
              // timer that overshoots its own estimate reads as a hang.
              hint={bothLocal ? "sequential — B starts when A finishes" : "running in parallel"}
            />
          )}

          {usage && usage.calls > 0 && (
            <span className="ml-auto text-xs text-slate-500">
              Anthropic spend estimate:{" "}
              <span className="font-mono">${usage.spend_estimate_usd.toFixed(4)}</span>
            </span>
          )}
        </div>

        {bothLocal && (
          <p className="text-xs text-slate-500">
            Both sides use the local model, so they run one after the other — two concurrent
            copies of an 8B model do not fit in memory comfortably.
          </p>
        )}
      </form>

      {error && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40">
          <p className="text-sm font-medium text-amber-900 dark:text-amber-200">
            Could not complete the comparison{error.status ? ` (${error.status})` : ""}
          </p>
          <p className="mt-1 text-sm text-amber-800 dark:text-amber-300">{error.message}</p>
        </div>
      )}

      {result && (
        <>
          <DiffSummary diff={result.diff} a={result.a} b={result.b} />

          <div className="grid gap-6 lg:grid-cols-2">
            <SideResult label="A" result={result.a} />
            <SideResult label="B" result={result.b} />
          </div>
        </>
      )}
    </div>
  );
}

function SideResult({ label, result }: { label: string; result: CompareResponse["a"] }) {
  return (
    <section className="rounded-lg border border-slate-200 p-5 dark:border-slate-800">
      <div className="mb-4 flex items-baseline gap-2 border-b border-slate-200 pb-3 dark:border-slate-800">
        <span className="rounded bg-slate-900 px-1.5 py-0.5 text-xs font-bold text-white dark:bg-slate-100 dark:text-slate-900">
          {label}
        </span>
        <span className="font-semibold tracking-tight">{result.technique}</span>
        <span className="text-xs text-slate-500">on {result.metadata.model}</span>
      </div>
      <ResultPanel result={result} />
    </section>
  );
}

function SidePicker({
  label,
  techniques,
  models,
  technique,
  model,
  onTechnique,
  onModel,
}: {
  label: string;
  techniques: Technique[];
  models: ModelInfo[];
  technique: string;
  model: string;
  onTechnique: (value: string) => void;
  onModel: (value: string) => void;
}) {
  return (
    <fieldset className="rounded-lg border border-slate-200 p-4 dark:border-slate-800">
      <legend className="px-2 text-xs font-bold uppercase tracking-wide text-slate-500">
        Side {label}
      </legend>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block">
          <span className="text-xs text-slate-500">Technique</span>
          <select
            value={technique}
            onChange={(e) => onTechnique(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm dark:border-slate-700"
          >
            {techniques.map((t) => (
              <option key={t.name} value={t.name} disabled={!t.implemented}>
                {t.display_name}
                {t.implemented ? "" : " — not built yet"}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="text-xs text-slate-500">Model</span>
          <select
            value={model}
            onChange={(e) => onModel(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm dark:border-slate-700"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available}>
                {m.display_name}
                {m.is_paid ? " — paid" : " — free"}
              </option>
            ))}
          </select>
        </label>
      </div>
    </fieldset>
  );
}

function ElapsedTimer({ hint }: { hint: string }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const id = setInterval(() => setSeconds((Date.now() - startedAt) / 1000), 100);
    return () => clearInterval(id);
  }, []);

  return (
    <span className="font-mono text-xs text-slate-500 tabular-nums">
      {seconds.toFixed(1)}s — {hint}
    </span>
  );
}
