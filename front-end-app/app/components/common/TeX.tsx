"use client";

import katex from "katex";
import { useMemo } from "react";

interface TeXProps {
  math: string;
  block?: boolean;
  style?: React.CSSProperties;
  className?: string;
}

export default function TeX({ math, block = false, style, className }: TeXProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: block,
        throwOnError: false,
        output: "html",
      });
    } catch {
      return math;
    }
  }, [math, block]);

  const Tag = block ? "div" : "span";
  return (
    <Tag
      className={className}
      style={style}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
