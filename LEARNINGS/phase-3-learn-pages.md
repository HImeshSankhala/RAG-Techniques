# Phase 3 — Learn pages

Nine documents, one per technique, at `/learn/[slug]`. Only one technique runs, and the
site is now a useful learning resource anyway. That ordering is the point of this phase.

## Why content before more engines

The tempting order is to build all nine techniques and document them at the end. The plan
deliberately does the opposite, and there are two reasons.

**The writing is a design review.** Writing "when NOT to use this" for a technique you have
not built forces you to state its failure mode precisely. Several things in this phase's
content only became clear while writing them — that Fusion RAG improves *what* one pass
retrieves but cannot add passes, that Auto RAG's router is only economical because it sees
tens of tokens while the answer call sees a thousand, that Interactive RAG is the first
technique that cannot fit `run(query) -> RAGResult`. Those are the kind of constraints that
are cheap to notice now and expensive to discover mid-implementation.

**The site is useful immediately.** Eight of the nine are docs-only, and someone can still
learn the whole landscape from it today. Had the order been reversed, there would be
nothing to read until the last phase.

## The content is grounded in this project's own measurements

The docs cite failures actually observed here rather than generic textbook ones:

- **Fusion RAG** opens with the `memtable` query, which in this corpus returns Bigtable's
  *tablet location hierarchy* as its top hit while the passage defining a memtable ranks
  third. That was found in Phase 1 testing, not borrowed from a paper.
- **Graph RAG** uses the Cassandra question, whose answer is genuinely split across
  `bigtable.md` and `dynamo.md` — verifiable by reading the corpus.
- **Multi-Pass** and **Agentic** cite the cost arithmetic from the actual guardrails.

This matters because it makes the claims checkable. A reader can run the memtable query in
the playground and watch dense retrieval mis-rank it.

## Architecture: MDX with a dynamic route

```
content/<slug>.mdx   ← prose only
app/learn/[slug]/    ← one route renders all nine
implementations/registry.py  ← still the source of truth for name/tagline/implemented
```

The split that matters: **MDX holds prose, the API holds metadata.** The page fetches
`display_name`, `tagline`, and `implemented` from `/api/techniques` rather than duplicating
them in frontmatter. So the "Runnable" badge on a learn page flips automatically when a
technique is registered in Phase 4 — the same derivation that has held since Phase 0.

The route uses `await import(\`@/content/${slug}.mdx\`)` with `generateStaticParams` and
`dynamicParams = false`. All nine are prerendered at build time (13 static pages total), and
an unknown slug 404s rather than attempting an import that cannot resolve.

The learn page degrades gracefully if the backend is down: `findTechnique` catches and
returns `null`, so the prose still renders and only the header chrome is lost. The writing
is the point of the page, and it does not need an API to be worth reading.

## Keeping the markdown portable

Diagrams are ```` ```mermaid ```` fenced blocks, not imported React components. The
`mdx-components.tsx` override intercepts `<pre>` and renders those fences as diagrams:

```tsx
if (className.includes("language-mermaid")) return <MermaidDiagram chart={code} />;
```

The alternative — `<MermaidDiagram chart={...} />` written inline in each MDX file — works
but couples the content to this project's components. With fences, the nine documents are
ordinary markdown: they render correctly on GitHub, in an editor preview, and in any
markdown tool. Only the *diagram rendering* is Next-specific, not the content.

`mermaid` is ~1MB and lazily imported inside a `useEffect`, so it never enters the server
bundle or the initial payload. The raw fence text is the fallback if rendering fails —
a reader gets the diagram source rather than a blank space where an explanation should be.

## Turbopack changes how remark plugins are configured

`remark-gfm` (needed for the trade-off tables) cannot be imported and passed as a function:

```js
// Webpack-era, breaks under Turbopack
remarkPlugins: [remarkGfm]

// Turbopack
remarkPlugins: ["remark-gfm"]
```

Turbopack runs the MDX pipeline in Rust, and a JavaScript function cannot cross that
boundary — plugins are named as strings and resolved on the Rust side. The consequence is
that any plugin needing non-serializable options is currently unusable with Turbopack,
which is worth knowing before designing a content pipeline around one.

## Failure mode: two systems both trying to add quotation marks

The Graph RAG page rendered its pull quote as:

```
""Which system combined Bigtable's data model with Dynamo's replication approach?""
```

Doubled. The cause is a collision between two layers that each assumed they owned the
punctuation:

1. The MDX source contained literal quotes: `> *"Which system…?"*`
2. `@tailwindcss/typography` styles blockquotes with
   `blockquote p:first-of-type::before { content: open-quote }`

Neither is wrong in isolation. The plugin adds quotes because a blockquote is *typically*
a quotation; the author added quotes because the sentence *is* one. Together they render
twice.

The fix is to delete the literal quotes and let the stylesheet own them. The general lesson
is about **CSS that injects content**: `::before { content: … }` is invisible in the source
and invisible in the DOM inspector's element tree, so a duplicated character has no
obvious origin. It was caught by looking at the rendered page, and it would not have been
caught by a build, a type check, or a lint rule — none of which have any view of what the
stylesheet adds.

## What "done" means here

Verified in a browser rather than asserted: all nine pages return 200, prose and tables
render, both Mermaid diagrams on the Graph RAG page draw (including its `subgraph` blocks),
home cards link to the learn pages, prev/next navigation works, the "Runnable" badge
appears only on `standard-rag`, and its "Try it in the playground →" button lands on
`/playground?technique=standard-rag` with Standard RAG already selected.
