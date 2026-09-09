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
