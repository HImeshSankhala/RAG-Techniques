import type { MDXComponents } from "mdx/types";
import { MermaidDiagram } from "@/components/MermaidDiagram";

/**
 * Global component overrides for every MDX file.
 *
 * Required by @next/mdx with the App Router — MDX will not compile without this
 * file at the project root.
 *
 * The interesting override is `pre`: it intercepts fenced code blocks tagged
 * ```mermaid and renders them as diagrams instead of code. That keeps the
 * technique docs as portable markdown — a ```mermaid fence still reads fine on
 * GitHub — rather than requiring an imported React component per diagram.
 */
const components: MDXComponents = {
  pre: (props) => {
    const child = props.children as
      | { props?: { className?: string; children?: string } }
      | undefined;
    const className = child?.props?.className ?? "";
    const code = child?.props?.children ?? "";

    if (className.includes("language-mermaid") && typeof code === "string") {
      return <MermaidDiagram chart={code.trim()} />;
    }

    return (
      <pre className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm dark:border-slate-800 dark:bg-slate-900">
        {props.children}
      </pre>
    );
  },
};

export function useMDXComponents(): MDXComponents {
  return components;
}
