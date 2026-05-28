import type { Metadata } from "next";
import SlidesView from "./SlidesView";

export const metadata: Metadata = {
  title: "Agentic Loop & Tối ưu Token – Slide",
  description:
    "Slide chuyên sâu về vòng lặp agentic và các kỹ thuật tối ưu token trong hệ thống AI.",
};

export default function AgenticLoopSlidesPage() {
  return <SlidesView />;
}
