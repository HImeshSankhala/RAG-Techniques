import { defineConfig, globalIgnores } from "eslint/config";
import nextPlugin from "@next/eslint-plugin-next";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

// Composed by hand instead of using `eslint-config-next`, because that package
// bundles eslint-plugin-react and eslint-plugin-jsx-a11y, neither of which
// supports ESLint 10 yet (both cap their peer range at ESLint 9). This is the
// "use the plugin directly" path in the Next.js ESLint docs.
//
// Revisit once eslint-plugin-react ships ESLint 10 support: switching back to
// `eslint-config-next/core-web-vitals` would restore the jsx-a11y rules this
// config currently gives up.
export default defineConfig([
  globalIgnores([".next/**", "out/**", "build/**", "next-env.d.ts"]),
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{js,jsx,ts,tsx,mjs}"],
    plugins: { "@next/next": nextPlugin },
    rules: {
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs["core-web-vitals"].rules,
    },
  },
  // `.configs.flat` — the top-level configs are still eslintrc-shaped.
  reactHooks.configs.flat["recommended-latest"],
]);
