import type { Metadata } from "next";
import SlidesView from "./SlidesView";

export const metadata: Metadata = {
  title: "Backpropagation trong CNN – Slide",
  description:
    "Slide chuyên sâu về cơ chế lan truyền ngược trong mạng CNN, từ Loss đến điều chỉnh ma trận, kèm ví dụ chữ số 7.",
};

export default function BackpropagationSlidesPage() {
  return <SlidesView />;
}
