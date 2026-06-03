import type { NextConfig } from "next";
import createMDX from "@next/mdx";

const nextConfig: NextConfig = {
  // Cho phép .mdx vừa làm trang vừa làm module import
  pageExtensions: ["js", "jsx", "ts", "tsx", "md", "mdx"],
  // Cho phép truy cập dev server qua IP LAN (nếu không, Next 16 chặn
  // các tài nguyên /_next/ cross-origin -> JS không tải -> trang không tương tác)
  allowedDevOrigins: ["192.168.1.27"],
};

// Turbopack chỉ nhận tên plugin dạng chuỗi (không truyền được hàm JS sang Rust)
const withMDX = createMDX({
  options: {
    remarkPlugins: ["remark-gfm", "remark-math"],
    rehypePlugins: [["rehype-katex", { throwOnError: false }]],
  },
});

export default withMDX(nextConfig);
