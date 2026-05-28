import type { Metadata } from "next";
import SlidesView from "./SlidesView";

export const metadata: Metadata = {
  title: "Slide thuyết trình – CNN MNIST Lab",
  description:
    "Bộ slide thuyết trình về đồ án CNN nhận diện chữ số MNIST xây dựng bằng NumPy.",
};

export default function SlidesPage() {
  return <SlidesView />;
}
