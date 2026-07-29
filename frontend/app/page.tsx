import { TechniqueCard } from "@/components/TechniqueCard";
import { API_BASE_URL, ApiError, getTechniques, type Technique } from "@/lib/api";

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
    error = cause instanceof ApiError ? cause.message : "Unknown error loading techniques.";
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

      {error ? <ApiErrorNotice message={error} /> : <TechniqueGrid techniques={techniques} />}
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
