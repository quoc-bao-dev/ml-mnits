"use client";

import React, { useState, useEffect } from "react";
import { Spin } from "antd";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default function DocumentPage({ params }: PageProps) {
  const { slug } = React.use(params);
  const [docContent, setDocContent] = useState<string>("");
  const [loadingDoc, setLoadingDoc] = useState<boolean>(true);

  useEffect(() => {
    if (slug) {
      setLoadingDoc(true);
      fetch(`/api/documents/${slug}`)
        .then((res) => res.json())
        .then((data) => {
          setDocContent(data.content || "Tài liệu trống");
          setLoadingDoc(false);
        })
        .catch((err) => {
          setDocContent(`Lỗi khi tải tài liệu: ${err.message}`);
          setLoadingDoc(false);
        });
    }
  }, [slug]);

  return (
    <div className="glass-card markdown-container" style={{ height: "auto", minHeight: "100%" }}>

      {loadingDoc ? (
        <div style={{ textAlign: "center", padding: "100px 0" }}>
          <Spin size="large" tip="Đang tải tài liệu..." />
        </div>
      ) : (
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
        >
          {docContent}
        </ReactMarkdown>
      )}
    </div>
  );
}
