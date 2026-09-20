"use client";

import { useState } from "react";
import { ApiError, deleteDocuments, uploadDocuments, type UploadedCorpus } from "@/lib/api";

/**
 * Choose the corpus: the curated demo one, or documents the reader uploads.
 *
 * Shared by the playground and the compare view because the two need exactly the
 * same control and the same warning, and a second copy is a second place for the
 * honest sentence below to drift.
 *
 * The demo corpus stays the default and the labelled first option. It is curated
 * so the techniques visibly diverge; an arbitrary upload is not, and the most
 * likely outcome on a short document is that every technique returns the same
 * passages. That is a true result rather than a broken feature, and saying so
 * before the upload is cheaper than explaining it afterwards.
 */
export function CorpusPicker({
  corpus,
  onChange,
}: {
  corpus: UploadedCorpus | null;
  onChange: (corpus: UploadedCorpus | null) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload() {
    if (files.length === 0 || isUploading) return;
    setIsUploading(true);
    setError(null);
    try {
      onChange(await uploadDocuments(files));
      setFiles([]);
    } catch (thrown) {
      setError(thrown instanceof ApiError ? thrown.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleReset() {
    if (!corpus) return;
    const { session_id } = corpus;
    onChange(null);
    await deleteDocuments(session_id);
  }

  return (
    <section className="rounded-lg border border-slate-200 p-4 dark:border-slate-800">
      <h2 className="text-xs font-medium uppercase tracking-wide text-slate-500">Corpus</h2>

      {corpus ? (
        <div className="mt-3 space-y-2">
          <p className="text-sm">
            Running against <span className="font-medium">{corpus.label}</span> —{" "}
            <span className="font-mono text-xs">{corpus.chunks}</span> chunks from{" "}
            {corpus.documents} document{corpus.documents === 1 ? "" : "s"}.
          </p>
          <p className="text-xs text-slate-500">
            Three techniques are unavailable here and say why in the selector. This corpus
            stops working an hour after upload, and is deleted the next time the server sweeps
            — on the next upload, the next restart, or the moment you try to use it again.
          </p>
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs transition hover:border-slate-500 dark:border-slate-700"
          >
            Back to the demo corpus
          </button>
        </div>
      ) : (
        <div className="mt-3 space-y-2">
          <p className="text-sm">
            Running against the <span className="font-medium">demo corpus</span> — nine distributed
            systems papers, curated so the techniques visibly disagree.
          </p>
          <p className="text-xs text-slate-500">
            Or bring your own: .txt, .md or .pdf. Expect less divergence — the demo corpus was
            chosen for questions where dense and keyword retrieval pull different passages, and
            most documents have no such question in them. The compare view reports what it
            measured either way.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="file"
              multiple
              accept=".txt,.md,.pdf"
              onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
              className="text-xs file:mr-3 file:rounded-lg file:border file:border-slate-300 file:bg-transparent file:px-3 file:py-1.5 file:text-xs dark:file:border-slate-700 dark:file:text-slate-300"
            />
            <button
              type="button"
              onClick={handleUpload}
              disabled={files.length === 0 || isUploading}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700"
            >
              {isUploading ? "Indexing…" : "Index these"}
            </button>
          </div>
        </div>
      )}

      {error && (
        <p className="mt-3 text-xs text-amber-700 dark:text-amber-400" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
