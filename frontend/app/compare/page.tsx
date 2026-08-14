import { CompareView } from "@/components/CompareView";
import { API_BASE_URL, getModels, getTechniques, type ModelInfo, type Technique } from "@/lib/api";

export const metadata = {
  title: "Compare — RAG Lab",
  description: "Run two RAG techniques on the same query and see exactly what differed.",
};

export default async function ComparePage() {
  let techniques: Technique[] = [];
  let models: ModelInfo[] = [];
  let error: string | null = null;

  try {
    [techniques, models] = await Promise.all([getTechniques(), getModels()]);
  } catch (cause) {
    error = cause instanceof Error ? cause.message : "Unknown error loading the compare page.";
  }

  return (
    <>
      <section className="max-w-2xl">
        <h1 className="text-3xl font-bold tracking-tight">Compare</h1>
        <p className="mt-4 leading-relaxed text-slate-600 dark:text-slate-400">
          Two techniques, one query, side by side. Vary the technique to see whether
          retrieval changes what the model was given; vary the model to see whether the
          same evidence produces a different answer.
        </p>
        <p className="mt-3 leading-relaxed text-slate-600 dark:text-slate-400">
          The answers usually look similar. The interesting differences are in the
          evidence and the cost — which is what the summary row measures.
        </p>
      </section>

      <div className="mt-8">
        {error ? (
          <SetupNotice message={error} />
        ) : (
          <CompareView techniques={techniques} models={models} />
        )}
      </div>
    </>
  );
}

function SetupNotice({ message }: { message: string }) {
  return (
    <section className="rounded-lg border border-amber-300 bg-amber-50 p-5 dark:border-amber-800 dark:bg-amber-950/40">
      <h2 className="font-semibold text-amber-900 dark:text-amber-200">
        Could not load the compare page
      </h2>
      <p className="mt-2 text-sm text-amber-800 dark:text-amber-300">{message}</p>
      <p className="mt-3 text-sm text-amber-800 dark:text-amber-300">
        Start the backend with <code className="font-mono">make dev</code>. The API should
        answer at <code className="font-mono">{API_BASE_URL}/api/techniques</code>.
      </p>
    </section>
  );
}
