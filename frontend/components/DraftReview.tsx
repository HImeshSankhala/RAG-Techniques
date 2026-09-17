"use client";

import { useState } from "react";
import type { RunResponse } from "@/lib/api";

/**
 * The human's turn in Interactive RAG: untick passages that do not help, add a
 * hint for what is missing, ask for the final answer.
 *
 * Every passage starts ticked, so the user's job is to remove what is off-topic.
 * Clicking straight through (all ticked, no hint) is allowed on purpose — the
 * engine answers it with `no_change` and no LLM call, and the trace says why.
 *
 * Mount with `key={draft.draft_id}` so a new draft starts from fresh state.
 */
export function DraftReview({
  draft,
  isRunning,
  onSubmit,
}: {
  draft: RunResponse;
  isRunning: boolean;
  onSubmit: (chunkIds: string[], hint: string) => void;
}) {
  const [kept, setKept] = useState(() => new Set(draft.retrieved_chunks.map((c) => c.chunk_id)));
  const [hint, setHint] = useState("");

  // Mirrors the API's 422: nothing kept and no hint leaves nothing to answer from.
  const canSubmit = !isRunning && (kept.size > 0 || hint.trim() !== "");

  function toggle(chunkId: string) {
    setKept((previous) => {
      const next = new Set(previous);
      if (!next.delete(chunkId)) next.add(chunkId);
      return next;
    });
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    const ids = draft.retrieved_chunks.map((c) => c.chunk_id).filter((id) => kept.has(id));
    onSubmit(ids, hint.trim());
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-4 rounded-lg border border-sky-300 p-5 dark:border-sky-800"
    >
      <div>
        <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">
          Your review
        </h3>
        {/* The draft's own query, not the textarea's: the form may have been
            edited since, and the final answer is to the question drafted. */}
        <p className="mt-1 text-sm">
          For <span className="font-medium">“{draft.query}”</span>: untick passages that do not
          help, and add a hint if something is missing.
        </p>
      </div>

      <ul className="space-y-2">
        {draft.retrieved_chunks.map((chunk) => (
          <li key={chunk.chunk_id}>
            <label className="flex cursor-pointer gap-3 text-sm">
              <input
                type="checkbox"
                checked={kept.has(chunk.chunk_id)}
                onChange={() => toggle(chunk.chunk_id)}
                className="mt-1"
              />
              <span>
                <span className="font-mono text-xs text-slate-500">{chunk.chunk_id}</span>
                <span className="block text-slate-600 dark:text-slate-400">
                  {chunk.text.length > 160 ? `${chunk.text.slice(0, 160)}…` : chunk.text}
                </span>
              </span>
            </label>
          </li>
        ))}
      </ul>

      <label className="block">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
          Hint — searches for more passages; the model never sees it
        </span>
        <input
          type="text"
          value={hint}
          onChange={(e) => setHint(e.target.value)}
          maxLength={500}
          placeholder="e.g. vector clocks"
          className="mt-1 w-full rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm dark:border-slate-700"
        />
      </label>

      <button
        type="submit"
        disabled={!canSubmit}
        className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
      >
        {isRunning ? "Answering…" : "Get final answer"}
      </button>
    </form>
  );
}
