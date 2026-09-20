/**
 * Display formatters shared by the result and comparison views.
 *
 * These lived as identical copies in three components. Two copies is a
 * coincidence; three is a rule, and a rule that is stated three times drifts —
 * the moment one of them rounds differently, the trace and the diff row disagree
 * about the same run.
 */

/** Milliseconds as a human duration: `840ms`, `1.2s`. */
export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

/** USD, or the word that matters more than the number on the local backend. */
export function money(usd: number): string {
  return usd > 0 ? `$${usd.toFixed(4)}` : "free";
}

/**
 * Why a technique cannot be picked here, or null when it can.
 *
 * Four different facts get collapsed into one disabled `<option>`, and for two
 * phases the selector called all of them "not built yet". That is a roadmap
 * promise, and REALM is a pre-training method — no amount of building delivers
 * it. Both sides of that `if` typecheck, which is why this is a function with a
 * test rather than a ternary repeated in two components.
 *
 * `context` exists because the same technique is not blocked in both places:
 * Interactive RAG runs fine in the playground and only the compare view has to
 * refuse it, since a side whose human is stubbed out would misreport the run.
 *
 * `onUpload` is the Phase 13 fact, and it is last for the same branch-order
 * reason the first two are first: REALM is unavailable everywhere, and saying
 * "not on your documents" about it would imply it runs on the demo corpus. The
 * sentence itself comes from the API (`upload_note`), never from here — the
 * server owns both the gate and the explanation, so a technique cannot be
 * offered here and refused there.
 */
export function unavailableReason(
  technique: {
    implemented: boolean;
    docs_only: boolean;
    needs_human: boolean;
    upload_note?: string;
  },
  context: "playground" | "compare",
  onUpload = false,
): string | null {
  // Checked before `implemented`: a docs-only technique is also unimplemented,
  // and answering "not built yet" about REALM is the bug this function exists for.
  if (technique.docs_only) return "cannot run here";
  if (!technique.implemented) return "not built yet";
  if (technique.needs_human && context === "compare") return "needs a human";
  if (onUpload && technique.upload_note) return technique.upload_note;
  return null;
}
