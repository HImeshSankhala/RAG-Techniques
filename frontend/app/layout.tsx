import type { Metadata } from "next";
import Link from "next/link";
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
          <nav className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-4">
            <Link href="/" className="font-semibold tracking-tight">
              RAG Lab
            </Link>
            <div className="ml-auto flex gap-5 text-sm text-slate-600 dark:text-slate-400">
              <Link href="/playground" className="transition hover:text-slate-900 dark:hover:text-slate-100">
                Playground
              </Link>
              <Link href="/compare" className="transition hover:text-slate-900 dark:hover:text-slate-100">
                Compare
              </Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
