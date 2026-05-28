"use client";

import Deck from "@/content/slides/agentic-loop.mdx";

// Import MDX trong client component để các phần tử <Slide/> được tạo phía client,
// nhờ đó SlideDeck nhận diện đúng và điều hướng/đọc props hoạt động bình thường.
export default function AgenticLoopSlidesView() {
  return <Deck />;
}
