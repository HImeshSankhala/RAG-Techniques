import Link from "next/link";
import { notFound } from "next/navigation";
import { getTechniques, type Technique } from "@/lib/api";

/**
 * Renders one technique's MDX document.
 *
 * The technique metadata (name, tagline, whether it runs) comes from the API so
 * the registry stays the single source of truth — the MDX file holds prose only.
 * If the backend is down the page still renders the content; only the header
 * chrome degrades, because the writing is the point of this page.
 */

// Every slug is known at build time, and an unknown one should 404 rather than
// attempt an import that cannot resolve.
export const dynamicParams = false;

const SLUGS = [
  "standard-rag",
  "fusion-rag",
  "multi-pass-rag",
  "auto-rag",
  "graph-rag",
  "agentic-rag",
  "interactive-rag",
  "feedback-rag",
  "realm",
] as const;

export function generateStaticParams() {
  return SLUGS.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const technique = await findTechnique(slug);
  return {
    title: `${technique?.display_name ?? slug} — RAG Lab`,
    description: technique?.tagline,
  };
}

async function findTechnique(slug: string): Promise<Technique | null> {
  try {
    return (await getTechniques()).find((t) => t.name === slug) ?? null;
  } catch {
    // Backend down: the prose is still worth serving.
    return null;
  }
}

export default async function LearnPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;

  if (!SLUGS.includes(slug as (typeof SLUGS)[number])) notFound();

  const { default: Content } = await import(`@/content/${slug}.mdx`);
  const technique = await findTechnique(slug);

  const index = SLUGS.indexOf(slug as (typeof SLUGS)[number]);
  const previous = SLUGS[index - 1];
  const next = SLUGS[index + 1];

  return (
    <article>
      <Link
        href="/"
        className="text-sm text-slate-500 transition hover:text-slate-900 dark:hover:text-slate-100"
      >
        ← All techniques
      </Link>

      <header className="mt-4 border-b border-slate-200 pb-6 dark:border-slate-800">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-3xl font-bold tracking-tight">
            {technique?.display_name ?? slug}
          </h1>
          {technique && <StatusBadge implemented={technique.implemented} />}
        </div>

        {technique && (
          <p className="mt-3 max-w-2xl leading-relaxed text-slate-600 dark:text-slate-400">
            {technique.tagline}
          </p>
        )}

        {technique?.implemented && (
          <Link
            href={`/playground?technique=${slug}`}
            className="mt-5 inline-block rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
          >
            Try it in the playground →
          </Link>
        )}
      </header>

      <div className="prose prose-slate mt-8 max-w-none dark:prose-invert prose-headings:tracking-tight prose-h2:mt-10 prose-h2:text-xl prose-h3:text-base prose-table:text-sm prose-pre:bg-transparent prose-pre:p-0">
        <Content />
      </div>

      <nav className="mt-12 flex justify-between gap-4 border-t border-slate-200 pt-6 text-sm dark:border-slate-800">
        {previous ? (
          <Link href={`/learn/${previous}`} className="text-slate-600 hover:underline dark:text-slate-400">
            ← {previous}
          </Link>
        ) : (
          <span />
        )}
        {next && (
          <Link href={`/learn/${next}`} className="text-slate-600 hover:underline dark:text-slate-400">
            {next} →
          </Link>
        )}
      </nav>
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
