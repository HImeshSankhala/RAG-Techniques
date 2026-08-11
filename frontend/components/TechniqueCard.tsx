import Link from "next/link";
import type { Technique } from "@/lib/api";

/**
 * One technique on the home grid, linking to its learn page.
 *
 * Phase 0 rendered this as a static card because /learn/[slug] did not exist yet
 * and linking to a 404 is worse than not linking. Those pages exist now.
 */
export function TechniqueCard({ technique }: { technique: Technique }) {
  return (
    <Link
      href={`/learn/${technique.name}`}
      className="group flex flex-col rounded-lg border border-slate-200 p-5 transition hover:border-slate-400 hover:shadow-sm dark:border-slate-800 dark:hover:border-slate-600"
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-semibold tracking-tight group-hover:underline">
          {technique.display_name}
        </h3>
        <StatusBadge implemented={technique.implemented} />
      </div>
      <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
        {technique.tagline}
      </p>
    </Link>
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
