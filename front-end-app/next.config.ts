import type { NextConfig } from "next";
import createMDX from "@next/mdx";

const nextConfig: NextConfig = {
  // Cho phép .mdx vừa làm trang vừa làm module import
  pageExtensions: ["js", "jsx", "ts", "tsx", "md", "mdx"],
};

// Turbopack chỉ nhận tên plugin dạng chuỗi (không truyền được hàm JS sang Rust)
const withMDX = createMDX({
  options: {
    remarkPlugins: ["remark-gfm", "remark-math"],
    rehypePlugins: [["rehype-katex", { throwOnError: false }]],
  },
});

export default withMDX(nextConfig);
