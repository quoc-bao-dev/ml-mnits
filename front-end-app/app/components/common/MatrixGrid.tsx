"use client";

import React from "react";

interface MatrixGridProps {
  data: number[][];
  cellSize?: number;
  highlightRegion?: { row: number; col: number; size: number };
  colorScheme?: "grayscale" | "heatmap" | "gradient" | "diverging";
  showValues?: boolean;
  maxVal?: number;
  minVal?: number;
  label?: string;
  borderColor?: string;
  onCellClick?: (row: number, col: number, value: number) => void;
  selectedCell?: { row: number; col: number } | null;
}

function getColor(value: number, min: number, max: number, scheme: string): string {
  const norm = max === min ? 0.5 : (value - min) / (max - min);
  if (scheme === "grayscale") {
    const g = Math.round(norm * 255);
    return `rgb(${g},${g},${g})`;
  }
  if (scheme === "heatmap") {
    // Blue → Cyan → Green → Yellow → Red
    const r = norm < 0.5 ? 0 : Math.round((norm - 0.5) * 2 * 255);
    const g = norm < 0.5 ? Math.round(norm * 2 * 255) : Math.round((1 - norm) * 2 * 255);
    const b = norm < 0.5 ? Math.round((1 - norm * 2) * 255) : 0;
    return `rgb(${r},${g},${b})`;
  }
  if (scheme === "diverging") {
    // Red (negative, norm=0) -> Dark Slate (neutral, norm=0.5) -> Green (positive, norm=1)
    if (norm < 0.5) {
      const t = norm * 2; // 0 to 1
      const r = Math.round(239 * (1 - t) + 30 * t);
      const g = Math.round(68 * (1 - t) + 41 * t);
      const b = Math.round(68 * (1 - t) + 59 * t);
      return `rgb(${r},${g},${b})`;
    } else {
      const t = (norm - 0.5) * 2; // 0 to 1
      const r = Math.round(30 * (1 - t) + 16 * t);
      const g = Math.round(41 * (1 - t) + 185 * t);
      const b = Math.round(59 * (1 - t) + 129 * t);
      return `rgb(${r},${g},${b})`;
    }
  }
  // gradient: purple to green
  const r = Math.round((1 - norm) * 99);
  const g = Math.round(norm * 185);
  const b = Math.round((1 - norm) * 241);
  return `rgb(${r},${g},${b})`;
}

export default function MatrixGrid({
  data,
  cellSize = 18,
  highlightRegion,
  colorScheme = "grayscale",
  showValues = false,
  maxVal,
  minVal,
  label,
  borderColor,
  onCellClick,
  selectedCell,
}: MatrixGridProps) {
  if (!data || data.length === 0) return null;

  const rows = data.length;
  const cols = data[0].length;

  // Compute min/max from data if not provided
  const flatVals = data.flat();
  const computedMin = minVal ?? Math.min(...flatVals);
  const computedMax = maxVal ?? Math.max(...flatVals);

  const isHighlighted = (r: number, c: number) => {
    if (!highlightRegion) return false;
    const { row, col, size } = highlightRegion;
    return r >= row && r < row + size && c >= col && c < col + size;
  };

  const isSelected = (r: number, c: number) => {
    if (!selectedCell) return false;
    return selectedCell.row === r && selectedCell.col === c;
  };

  return (
    <div style={{ display: "inline-block" }}>
      {label && (
        <div style={{
          fontSize: 11,
          color: "var(--text-muted)",
          marginBottom: 4,
          textAlign: "center",
          fontWeight: 600,
        }}>
          {label}
        </div>
      )}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
          gridTemplateRows: `repeat(${rows}, ${cellSize}px)`,
          gap: 1,
          border: borderColor ? `2px solid ${borderColor}` : "1px solid var(--border)",
          borderRadius: 4,
          padding: 1,
          background: "var(--bg-primary)",
        }}
      >
        {data.map((row, ri) =>
          row.map((val, ci) => {
            const highlighted = isHighlighted(ri, ci);
            const selected = isSelected(ri, ci);
            return (
              <div
                key={`${ri}-${ci}`}
                title={`[${ri},${ci}] = ${val.toFixed(4)}`}
                onClick={() => onCellClick?.(ri, ci, val)}
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: getColor(val, computedMin, computedMax, colorScheme),
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: showValues ? Math.min(cellSize * 0.45, 10) : 0,
                  color: val > (computedMax + computedMin) / 2 ? "#000" : "#fff",
                  fontFamily: "var(--font-jetbrains-mono), monospace",
                  fontWeight: 500,
                  outline: selected ? "2px solid var(--accent)" : highlighted ? "2px solid #f59e0b" : "none",
                  outlineOffset: -1,
                  zIndex: selected || highlighted ? 2 : 1,
                  position: "relative",
                  transition: "outline 0.15s ease",
                  cursor: onCellClick ? "pointer" : "default",
                }}
              >
                {showValues ? val.toFixed(2) : ""}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
