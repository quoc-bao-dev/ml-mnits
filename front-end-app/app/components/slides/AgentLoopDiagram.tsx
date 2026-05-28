import React from "react";

/**
 * Sơ đồ trực quan vòng lặp của một AI agent (ReAct):
 * Nhiệm vụ → [Suy luận → Hành động → Quan sát] ↺ lặp lại → Trả lời.
 * Thuần CSS, không tương tác — dùng trong slide để minh họa.
 */
export default function AgentLoopDiagram() {
  return (
    <div className="agent-diagram">
      {/* Đầu vào */}
      <div className="agent-entry">
        <span className="agent-chip agent-chip-task">📥 Nhiệm vụ</span>
        <span className="agent-conn">→</span>
      </div>

      {/* Vòng lặp 3 bước */}
      <div className="agent-cycle-wrap">
        <div className="agent-arc">
          <span className="agent-arc-label">↺ Lặp lại đến khi xong</span>
        </div>

        <div className="agent-cycle">
          <div className="agent-node agent-node-reason">
            <div className="agent-node-icon">🧠</div>
            <div className="agent-node-name">Suy luận</div>
            <div className="agent-node-desc">Mô hình quyết định bước tiếp theo</div>
          </div>

          <span className="agent-conn">→</span>

          <div className="agent-node agent-node-act">
            <div className="agent-node-icon">🔧</div>
            <div className="agent-node-name">Hành động</div>
            <div className="agent-node-desc">Gọi công cụ: tìm kiếm, đọc file…</div>
          </div>

          <span className="agent-conn">→</span>

          <div className="agent-node agent-node-observe">
            <div className="agent-node-icon">👁️</div>
            <div className="agent-node-name">Quan sát</div>
            <div className="agent-node-desc">Đưa kết quả trở lại ngữ cảnh</div>
          </div>
        </div>

        {/* Lối ra */}
        <div className="agent-exit">
          <span className="agent-conn agent-conn-down">↓</span>
          <span className="agent-chip agent-chip-answer">
            ✅ Trả lời cuối — khi không cần gọi tool nữa
          </span>
        </div>
      </div>
    </div>
  );
}
