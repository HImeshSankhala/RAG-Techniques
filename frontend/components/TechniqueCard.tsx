import type { Technique } from "@/lib/api";

/**
 * One technique on the home grid.
 *
 * Phase 0 renders it as a static card. It becomes a link to /learn/[slug] in Phase 3,
 * once those pages exist — linking to routes that 404 would be worse than not linking.
 */
export function TechniqueCard({ technique }: { technique: Technique }) {
  return (
    <article className="flex flex-col rounded-lg border border-slate-200 p-5 dark:border-slate-800">
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-semibold tracking-tight">{technique.display_name}</h3>
        <StatusBadge implemented={technique.implemented} />
      </div>
      <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
        {technique.tagline}
      </p>
    </article>
  );
}

function StatusBadge({ implemented }: { implemented: boolean }) {
  const label = implemented ? "Runnable" : "Docs only";
  const classes = implemented
    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
    : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400";

  return (
    <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${classes}`}>
      {label}
    </span>
  );
}
