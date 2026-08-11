"use client";

import { useEffect, useId, useState } from "react";

/**
 * Renders a ```mermaid fenced block as a diagram.
 *
 * Client-side and lazily imported: mermaid is ~1MB and pulls in a full layout
 * engine, so it must not land in the server bundle or the initial page payload.
 * The learn pages are otherwise static, and this is the one thing on them that
 * needs the browser.
 *
 * The raw fence text is kept as a fallback. If rendering fails — a syntax error
 * in the diagram, or the import failing — the reader still sees the source
 * instead of a blank space where an explanation should be.
 */
export function MermaidDiagram({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, "");
  const [svg, setSvg] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          // `base` plus explicit variables rather than a bundled theme: the site
          // is theme-aware, and mermaid's own dark theme does not follow the
          // page's light/dark state.
          theme: "base",
          themeVariables: {
            fontFamily: "inherit",
            fontSize: "14px",
            primaryColor: "#f1f5f9",
            primaryTextColor: "#0f172a",
            primaryBorderColor: "#94a3b8",
            lineColor: "#94a3b8",
            secondaryColor: "#e2e8f0",
            tertiaryColor: "#f8fafc",
          },
        });
        const { svg: rendered } = await mermaid.render(`mermaid-${id}`, chart);
        if (!cancelled) setSvg(rendered);
      } catch {
        if (!cancelled) setFailed(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [chart, id]);

  if (failed) {
    return (
      <pre className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs dark:border-slate-800 dark:bg-slate-900">
        {chart}
      </pre>
    );
  }

  return (
    <div className="my-6 overflow-x-auto rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-50">
      {svg ? (
        // Mermaid returns an SVG string it generated from the diagram source in
        // this repo — not user input — so there is no untrusted HTML here.
        <div className="flex justify-center [&_svg]:h-auto [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />
      ) : (
        <div className="h-24 animate-pulse rounded bg-slate-100" />
      )}
    </div>
  );
}
