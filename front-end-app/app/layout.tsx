import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { AntdRegistry } from "@ant-design/nextjs-registry";
import "./globals.css";
import DashboardLayout from "./components/DashboardLayout";

const inter = Inter({
  subsets: ["latin", "vietnamese"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CNN MNIST Lab – Đồ án Toán cho KHMT",
  description:
    "Nhận diện chữ số viết tay với mạng tích chập (CNN) thuần NumPy – Đồ án môn Toán cho Khoa học Máy tính",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body style={{ fontFamily: "var(--font-inter), -apple-system, BlinkMacSystemFont, sans-serif" }}>
        <AntdRegistry>
          <DashboardLayout>{children}</DashboardLayout>
        </AntdRegistry>
      </body>
    </html>
  );
}
