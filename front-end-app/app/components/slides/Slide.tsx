import React from "react";

export type SlideVariant = "default" | "cover" | "center";

export interface SlideProps {
  /** Nhãn nhỏ phía trên tiêu đề (vd: "01 · Giới thiệu") */
  kicker?: React.ReactNode;
  /** Tiêu đề chính của slide */
  title?: React.ReactNode;
  /** Mô tả phụ dưới tiêu đề */
  subtitle?: React.ReactNode;
  /** Kiểu bố cục: mặc định | bìa (hero) | căn giữa */
  variant?: SlideVariant;
  children?: React.ReactNode;
}

/**
 * Một slide trong bộ trình chiếu. Tự render bề mặt slide (kicker/title/body),
 * SlideDeck chịu trách nhiệm chọn slide nào đang hiển thị + điều hướng.
 */
export default function Slide({
  kicker,
  title,
  subtitle,
  variant = "default",
  children,
}: SlideProps) {
  return (
    <section className={`slide-surface slide-${variant}`}>
      {(kicker || title || subtitle) && (
        <header className="slide-head">
          {kicker && <div className="slide-kicker">{kicker}</div>}
          {title && <h1 className="slide-title">{title}</h1>}
          {subtitle && <p className="slide-subtitle">{subtitle}</p>}
        </header>
      )}
      <div className="slide-body">{children}</div>
    </section>
  );
}
