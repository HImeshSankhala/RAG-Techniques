import { Playground } from "@/components/Playground";
import { API_BASE_URL, getModels, getTechniques, type ModelInfo, type Technique } from "@/lib/api";

export const metadata = {
  title: "Playground — RAG Lab",
  description: "Run a RAG technique against the indexed corpus and inspect how it answered.",
};

/**
 * Server Component for the initial lists, Client Component for the interaction.
 *
 * The technique and model lists are fetched here so the controls render populated
 * on first paint rather than flashing empty. The run itself has to happen in the
 * browser — it is a response to a click — which is why `Playground` is a client
 * component and this one only hands it data.
 */
export default async function PlaygroundPage() {
  let techniques: Technique[] = [];
  let models: ModelInfo[] = [];
  let error: string | null = null;

  try {
    [techniques, models] = await Promise.all([getTechniques(), getModels()]);
  } catch (cause) {
    error = cause instanceof Error ? cause.message : "Unknown error loading the playground.";
  }

  return (
    <>
      <section className="max-w-2xl">
        <h1 className="text-3xl font-bold tracking-tight">Playground</h1>
        <p className="mt-4 leading-relaxed text-slate-600 dark:text-slate-400">
          Ask a question against the indexed corpus — four documents on distributed systems
          (Dynamo, Bigtable, Raft/Paxos, MapReduce). The answer comes back with the passages
          that produced it and a trace of every stage, so you can see <em>why</em> it answered
          that way, not just what it said.
        </p>
      </section>

      <div className="mt-8">
        {error ? <SetupNotice message={error} /> : <Playground techniques={techniques} models={models} />}
      </div>
    </>
  );
}

function SetupNotice({ message }: { message: string }) {
  return (
    <section className="rounded-lg border border-amber-300 bg-amber-50 p-5 dark:border-amber-800 dark:bg-amber-950/40">
      <h2 className="font-semibold text-amber-900 dark:text-amber-200">
        Could not load the playground
      </h2>
      <p className="mt-2 text-sm text-amber-800 dark:text-amber-300">{message}</p>
      <p className="mt-3 text-sm text-amber-800 dark:text-amber-300">
        Start the backend with <code className="font-mono">make dev</code> and build the index
        with <code className="font-mono">make index</code>. The API should answer at{" "}
        <code className="font-mono">{API_BASE_URL}/api/techniques</code>.
      </p>
    </section>
  );
}
