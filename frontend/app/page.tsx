import Link from "next/link";
import { TechniqueCard } from "@/components/TechniqueCard";
import { API_BASE_URL, getTechniques, type Technique } from "@/lib/api";

// Server Component: the fetch happens on the server, so the browser never sees a
// loading flash and the API URL stays a server concern.
export default async function HomePage() {
  let techniques: Technique[] = [];
  let error: string | null = null;

  try {
    techniques = await getTechniques();
  } catch (cause) {
    // Deliberately no hardcoded fallback list — a silent fallback would hide a dead
    // backend behind a page that looks fine.
    error = cause instanceof Error ? cause.message : "Unknown error loading techniques.";
  }

  return (
    <>
      <section className="max-w-2xl">
        <h1 className="text-3xl font-bold tracking-tight">
          Nine ways to do retrieval-augmented generation
        </h1>
        <p className="mt-4 leading-relaxed text-slate-600 dark:text-slate-400">
          RAG gives a language model access to documents it was never trained on: retrieve
          the relevant passages first, then answer from them. How you retrieve is where the
          designs diverge — and where the trade-offs in latency, cost, and accuracy live.
          Read about each technique, run it, and compare any two on the same query.
        </p>
      </section>

      {error ? (
        <ApiErrorNotice message={error} />
      ) : (
        <>
          <TechniqueGrid techniques={techniques} />
          <ComparisonTable techniques={techniques} />
        </>
      )}
    </>
  );
}

function TechniqueGrid({ techniques }: { techniques: Technique[] }) {
  return (
    <section className="mt-10">
      <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">
        Techniques ({techniques.length})
      </h2>
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {techniques.map((technique) => (
          <TechniqueCard key={technique.name} technique={technique} />
        ))}
      </div>
    </section>
  );
}

/**
 * The same nine techniques as the grid, as a table you can scan down a column.
 *
 * Every cell comes from GET /api/techniques — the registry is the one place these
 * facts live, so there is no second list here to fall out of step with it. The two
 * range columns are editorial prose (`llm_calls_range`), not measurements: a run's
 * actual `Metadata.llm_calls` depends on the route Auto RAG picked or how early
 * Agentic's planner stopped, which is exactly why a single number would be a lie.
 *
 * It shares the grid's fetch and therefore the grid's error branch: when the API is
 * down both disappear together rather than the table rendering nine empty rows.
 */
function ComparisonTable({ techniques }: { techniques: Technique[] }) {
  return (
    <section className="mt-12">
      <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">
        Side by side
      </h2>
      <p className="mt-2 max-w-2xl text-sm text-slate-600 dark:text-slate-400">
        What each technique costs per query. Ranges, not measurements — the count for a
        given run depends on the path that run takes.
      </p>

      {/* Native table in a scroll container: five columns do not fit a phone, and a
          card-per-row rewrite would lose the column-scanning that is the point. */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[36rem] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
              <th scope="col" className="py-2 pr-4 font-medium">Technique</th>
              <th scope="col" className="py-2 pr-4 font-medium">LLM calls</th>
              <th scope="col" className="py-2 pr-4 font-medium">Retrieval passes</th>
              <th scope="col" className="py-2 pr-4 font-medium">Needs a human</th>
              <th scope="col" className="py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {techniques.map((t) => (
              <tr
                key={t.name}
                className="border-b border-slate-100 last:border-0 dark:border-slate-900"
              >
                <th scope="row" className="py-2.5 pr-4 font-medium">
                  <Link href={`/learn/${t.name}`} className="hover:underline">
                    {t.display_name}
                  </Link>
                </th>
                <td className="py-2.5 pr-4 tabular-nums text-slate-600 dark:text-slate-400">
                  {t.llm_calls_range}
                </td>
                <td className="py-2.5 pr-4 tabular-nums text-slate-600 dark:text-slate-400">
                  {t.retrieval_passes_range}
                </td>
                <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-400">
                  {t.needs_human ? "yes" : "no"}
                </td>
                <td className="py-2.5 text-slate-600 dark:text-slate-400">
                  {t.docs_only ? "docs only" : t.implemented ? "runnable" : "not built yet"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ApiErrorNotice({ message }: { message: string }) {
  return (
    <section className="mt-10 rounded-lg border border-amber-300 bg-amber-50 p-5 dark:border-amber-800 dark:bg-amber-950/40">
      <h2 className="font-semibold text-amber-900 dark:text-amber-200">
        Could not load techniques
      </h2>
      <p className="mt-2 text-sm text-amber-800 dark:text-amber-300">{message}</p>
      <p className="mt-3 text-sm text-amber-800 dark:text-amber-300">
        Start the backend with <code className="font-mono">make dev</code>, then reload. The
        API should answer at <code className="font-mono">{API_BASE_URL}/api/techniques</code>.
      </p>
    </section>
  );
}
