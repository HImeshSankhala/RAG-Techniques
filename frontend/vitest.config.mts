import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * Vitest over the frontend's pure logic. See PLAN.md "Testing scope" for what
 * belongs here and what does not.
 *
 * `environment: "node"` on purpose: nothing under test touches the DOM, so a
 * jsdom dependency would be infrastructure bought for nothing. The day a test
 * genuinely needs a DOM is the day that test is out of scope.
 */
export default defineConfig({
  resolve: {
    // Mirrors the `@/*` path alias in tsconfig.json. Vitest does not read
    // tsconfig paths.
    alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
  },
  test: {
    environment: "node",
    include: ["**/*.test.ts", "**/*.test.tsx"],
    exclude: ["node_modules/**", ".next/**"],
  },
});
