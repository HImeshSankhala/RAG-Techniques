import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Lab",
  description:
    "Read about 9 RAG techniques, run them live, and compare any two on the same query.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white text-slate-900 antialiased dark:bg-slate-950 dark:text-slate-100">
        <header className="border-b border-slate-200 dark:border-slate-800">
          {/* Playground / Compare nav links land in Phases 2 and 5, with the pages. */}
          <div className="mx-auto max-w-5xl px-6 py-4">
            <span className="font-semibold tracking-tight">RAG Lab</span>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
