"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Button, Tooltip } from "antd";
import {
  LeftOutlined,
  RightOutlined,
  FullscreenOutlined,
  FullscreenExitOutlined,
} from "@ant-design/icons";
import Slide, { type SlideProps } from "./Slide";

type SlideElement = React.ReactElement<SlideProps>;

function isSlide(node: React.ReactNode): node is SlideElement {
  return React.isValidElement(node) && node.type === Slide;
}

export default function SlideDeck({ children }: { children: React.ReactNode }) {
  // Chỉ giữ các phần tử <Slide/> làm đơn vị trình chiếu
  const slides = useMemo(() => {
    const all = React.Children.toArray(children);
    const onlySlides = all.filter(isSlide);
    return onlySlides.length > 0 ? onlySlides : all.filter(React.isValidElement);
  }, [children]);

  const total = slides.length;
  const [index, setIndex] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const deckRef = useRef<HTMLDivElement>(null);

  const go = useCallback(
    (next: number) => {
      setIndex((cur) => Math.min(Math.max(next, 0), Math.max(total - 1, 0)));
    },
    [total]
  );
  const prev = useCallback(() => go(index - 1), [go, index]);
  const next = useCallback(() => go(index + 1), [go, index]);

  const toggleFullscreen = useCallback(() => {
    const el = deckRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      el.requestFullscreen?.();
    }
  }, []);

  // Phím tắt điều hướng
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      switch (e.key) {
        case "ArrowRight":
        case "ArrowDown":
        case "PageDown":
        case " ":
          e.preventDefault();
          next();
          break;
        case "ArrowLeft":
        case "ArrowUp":
        case "PageUp":
        case "Backspace":
          e.preventDefault();
          prev();
          break;
        case "Home":
          e.preventDefault();
          go(0);
          break;
        case "End":
          e.preventDefault();
          go(total - 1);
          break;
        case "f":
        case "F":
          e.preventDefault();
          toggleFullscreen();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, go, total, toggleFullscreen]);

  // Đồng bộ trạng thái fullscreen
  useEffect(() => {
    const onFsChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  if (total === 0) {
    return <div className="slide-deck-empty">Chưa có slide nào.</div>;
  }

  const current = slides[index];
  const currentTitle =
    (isSlide(current) && (current.props.title as React.ReactNode)) || null;

  return (
    <div
      ref={deckRef}
      className={`slide-deck${isFullscreen ? " is-fullscreen" : ""}`}
    >
      {/* Thanh tiến trình */}
      <div className="slide-progress">
        <div
          className="slide-progress-fill"
          style={{ width: `${((index + 1) / total) * 100}%` }}
        />
      </div>

      {/* Sân khấu hiển thị slide hiện tại */}
      <div className="slide-stage">
        <div key={index} className="slide-fade">
          {current}
        </div>
      </div>

      {/* Thanh điều khiển */}
      <div className="slide-controls">
        <Button
          icon={<LeftOutlined />}
          onClick={prev}
          disabled={index === 0}
          aria-label="Slide trước"
        >
          Trước
        </Button>

        <div className="slide-meta">
          <span className="slide-counter">
            {index + 1} <span className="sep">/</span> {total}
          </span>
          {currentTitle && (
            <span className="slide-now" title={String(currentTitle)}>
              {currentTitle}
            </span>
          )}
        </div>

        <div className="slide-dots">
          {slides.map((_, i) => (
            <button
              key={i}
              className={`slide-dot${i === index ? " active" : ""}`}
              onClick={() => go(i)}
              aria-label={`Tới slide ${i + 1}`}
            />
          ))}
        </div>

        <div className="slide-actions">
          <Tooltip title={isFullscreen ? "Thoát toàn màn hình (F)" : "Toàn màn hình (F)"}>
            <Button
              icon={isFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
              onClick={toggleFullscreen}
              aria-label="Toàn màn hình"
            />
          </Tooltip>
          <Button
            type="primary"
            icon={<RightOutlined />}
            iconPosition="end"
            onClick={next}
            disabled={index === total - 1}
            aria-label="Slide tiếp theo"
          >
            Tiếp
          </Button>
        </div>
      </div>
    </div>
  );
}
