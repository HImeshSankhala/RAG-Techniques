import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx,mdx}",
    "./components/**/*.{ts,tsx}",
    "./content/**/*.mdx",
  ],
  theme: {
    extend: {},
  },
  // The learn pages are nine markdown documents; `prose` is what makes headings,
  // lists, and tables readable without hand-styling every element in MDX.
  plugins: [typography],
};

export default config;
