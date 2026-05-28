import type { MDXComponents } from "mdx/types";

// File bắt buộc của @next/mdx (App Router). Tùy biến cách render phần tử Markdown.
// Phần lớn style được xử lý qua CSS (.slide-body trong globals.css),
// ở đây chỉ giữ tối thiểu để mọi file MDX dùng chung.
const components: MDXComponents = {};

export function useMDXComponents(): MDXComponents {
  return components;
}
