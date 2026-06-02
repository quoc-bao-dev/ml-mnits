"use client";

import { InfoCircleOutlined } from "@ant-design/icons";
import { Popover } from "antd";
import TeX from "./TeX";

interface FormulaPopoverProps {
  /** Tiêu đề ngắn cho popover */
  title: string;
  /** Công thức tổng quát (LaTeX) */
  generalFormula: string;
  /** Công thức đã thay số cụ thể (LaTeX), nếu có */
  substitutedFormula?: string;
  /** Kết quả cuối cùng (LaTeX hoặc text), nếu có */
  result?: string;
  /** Ghi chú thêm */
  note?: string;
  /** Element trigger; nếu không có sẽ dùng icon mặc định */
  children?: React.ReactNode;
  /** Vị trí popover */
  placement?: "top" | "bottom" | "left" | "right" | "topLeft" | "topRight" | "bottomLeft" | "bottomRight";
}

export default function FormulaPopover({
  title,
  generalFormula,
  substitutedFormula,
  result,
  note,
  children,
  placement = "right",
}: FormulaPopoverProps) {
  const content = (
    <div style={{ maxWidth: 460, padding: 4, display: "flex", flexDirection: "column", gap: 12 }}>
      <div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
          Công thức tổng quát
        </div>
        <div style={{ padding: "10px 12px", background: "rgba(99,102,241,0.08)", borderRadius: 6, border: "1px solid var(--border)" }}>
          <TeX math={generalFormula} block />
        </div>
      </div>

      {substitutedFormula && (
        <div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
            Thay số vào
          </div>
          <div style={{ padding: "10px 12px", background: "rgba(245, 158, 11, 0.08)", borderRadius: 6, border: "1px solid rgba(245, 158, 11, 0.3)" }}>
            <TeX math={substitutedFormula} block />
          </div>
        </div>
      )}

      {result && (
        <div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
            Kết quả
          </div>
          <div style={{ padding: "10px 12px", background: "rgba(16, 185, 129, 0.08)", borderRadius: 6, border: "1px solid rgba(16, 185, 129, 0.3)" }}>
            <TeX math={result} block />
          </div>
        </div>
      )}

      {note && (
        <div style={{ fontSize: 11, color: "var(--text-secondary)", fontStyle: "italic", borderTop: "1px solid var(--border)", paddingTop: 8 }}>
          {note}
        </div>
      )}
    </div>
  );

  return (
    <Popover
      title={<span style={{ fontSize: 13 }}>{title}</span>}
      content={content}
      placement={placement}
      trigger={["click", "hover"]}
      mouseEnterDelay={0.3}
    >
      {children || (
        <InfoCircleOutlined
          style={{
            color: "var(--accent)",
            cursor: "pointer",
            fontSize: 14,
            verticalAlign: "middle",
          }}
        />
      )}
    </Popover>
  );
}
