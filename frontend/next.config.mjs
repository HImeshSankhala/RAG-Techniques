import createMDX from "@next/mdx";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  pageExtensions: ["ts", "tsx", "mdx"],
};

const withMDX = createMDX({
  options: {
    // Plugin named as a string, not imported: Turbopack runs the MDX pipeline in
    // Rust and cannot receive a JavaScript function across that boundary. Needed
    // for GFM tables, which every technique page uses for its trade-off table.
    remarkPlugins: ["remark-gfm"],
  },
});

export default withMDX(nextConfig);
