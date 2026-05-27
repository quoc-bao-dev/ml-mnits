import { Select } from "antd";
import { useState } from "react";
import MatrixGrid from "../common/MatrixGrid";

interface MaxPoolBackwardProps {
  data: any;
  step: any;
}

export default function MaxPoolBackward({ data, step }: MaxPoolBackwardProps) {
  const [backwardFilter, setBackwardFilter] = useState(0);
  const [selectedPoolBackCell, setSelectedPoolBackCell] = useState<{ row: number, col: number } | null>({ row: 0, col: 0 });

  const gradPool = data.steps[4]?.grad_pool; // 13x13x8
  const gradConv = step.grad_conv; // 26x26x8
  const convForward = data.steps[0]?.output; // 26x26x8

  if (!gradPool || !gradConv || !convForward) {
    return <div>Thiếu dữ liệu log để trực quan hóa MaxPool backward.</div>;
  }

  const r = selectedPoolBackCell?.row ?? 0;
  const c = selectedPoolBackCell?.col ?? 0;

  const poolGradSlice = gradPool.map((row: number[][]) => row.map(cell => cell[backwardFilter]));
  const convGradSlice = gradConv.map((row: number[][]) => row.map(cell => cell[backwardFilter]));
  const convForwardSlice = convForward.map((row: number[][]) => row.map(cell => cell[backwardFilter]));

  // Get 2x2 forward values
  const forwardVals = [
    [convForwardSlice[r * 2]?.[c * 2] ?? 0, convForwardSlice[r * 2]?.[c * 2 + 1] ?? 0],
    [convForwardSlice[r * 2 + 1]?.[c * 2] ?? 0, convForwardSlice[r * 2 + 1]?.[c * 2 + 1] ?? 0]
  ];
  const maxForwardVal = Math.max(...forwardVals.flat());

  // Find which coordinate of the 2x2 region was the max (winner)
  let maxOffsetRow = 0;
  let maxOffsetCol = 0;
  let found = false;
  for (let i2 = 0; i2 < 2; i2++) {
    for (let j2 = 0; j2 < 2; j2++) {
      if (forwardVals[i2][j2] === maxForwardVal) {
        maxOffsetRow = i2;
        maxOffsetCol = j2;
        found = true;
        break;
      }
    }
    if (found) break;
  }

  const poolGradVal = poolGradSlice[r][c];

  // 2x2 routing gradient grid
  const routedGrads = [
    [0, 0],
    [0, 0]
  ];
  routedGrads[maxOffsetRow][maxOffsetCol] = poolGradVal;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Filter selection */}
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>Chọn bộ lọc trực quan hóa:</span>
        <Select value={backwardFilter} onChange={setBackwardFilter} style={{ width: 140 }}
          options={Array.from({ length: 8 }, (_, i) => ({ value: i, label: `Filter ${i}` }))} />
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
        {/* Left: Input Gradient 13x13 */}
        <div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>1. Gradient từ Softmax (13×13)</div>
          <MatrixGrid
            data={poolGradSlice}
            cellSize={22}
            colorScheme="heatmap"
            selectedCell={selectedPoolBackCell}
            onCellClick={(row, col) => setSelectedPoolBackCell({ row, col })}
            label={`grad_pool (F${backwardFilter})`}
            highlightRegion={{ row: r, col: c, size: 1 }}
          />
          <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", marginTop: 4 }}>
            Click chọn ô để xem định tuyến gradient
          </div>
        </div>

        {/* Middle: Winner-take-all Routing Viz */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16, flex: 1, minWidth: 300, alignItems: "center" }}>
          <div className="glass-card" style={{ padding: 16, width: "100%", height: "auto" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 12, textAlign: "center" }}>
              Định tuyến ô [{r}, {c}] (Winner-take-all)
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: "center" }}>
              {/* Visual value transition */}
              <div style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: "center" }}>
                <div style={{ padding: "8px 12px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", borderRadius: 6, textAlign: "center" }}>
                  <div style={{ fontSize: 9, color: "var(--text-muted)" }}>Gradient nguồn</div>
                  <div style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 700, color: "var(--warning)" }}>{poolGradVal.toFixed(6)}</div>
                </div>
                <span style={{ fontSize: 18, color: "var(--text-muted)" }}>➔</span>
                <div style={{ padding: "8px 12px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", borderRadius: 6, textAlign: "center" }}>
                  <div style={{ fontSize: 9, color: "var(--text-muted)" }}>Vị trí max forward</div>
                  <div style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 700, color: "var(--success)" }}>[{r * 2 + maxOffsetRow}, {c * 2 + maxOffsetCol}]</div>
                </div>
              </div>

              {/* 2x2 forward values vs 2x2 gradient routing */}
              <div style={{ display: "flex", gap: 24, justifyContent: "center", marginTop: 8, flexWrap: "wrap" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>Vùng 2×2 (Forward)</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 40px)", gap: 4 }}>
                    {forwardVals.map((rowVals, ri) =>
                      rowVals.map((val, ci) => {
                        const isMax = ri === maxOffsetRow && ci === maxOffsetCol;
                        return (
                          <div key={`${ri}-${ci}`} style={{
                            width: 40,
                            height: 40,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: isMax ? "rgba(16, 185, 129, 0.2)" : "rgba(255, 255, 255, 0.02)",
                            border: `1px solid ${isMax ? "var(--success)" : "var(--border)"}`,
                            borderRadius: 4,
                            fontFamily: "monospace",
                            fontSize: 10,
                            color: isMax ? "var(--success)" : "var(--text-secondary)",
                            fontWeight: isMax ? 700 : 400
                          }}>
                            {val.toFixed(3)}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>Định tuyến Gradient</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 50px)", gap: 4 }}>
                    {routedGrads.map((rowVals, ri) =>
                      rowVals.map((val, ci) => {
                        const isWinner = ri === maxOffsetRow && ci === maxOffsetCol;
                        return (
                          <div key={`${ri}-${ci}`} style={{
                            width: 50,
                            height: 40,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: isWinner ? "rgba(245, 158, 11, 0.2)" : "rgba(255,255,255,0.01)",
                            border: `1px solid ${isWinner ? "var(--warning)" : "rgba(255,255,255,0.05)"}`,
                            borderRadius: 4,
                            fontFamily: "monospace",
                            fontSize: 9,
                            color: isWinner ? "var(--warning)" : "var(--text-muted)",
                            fontWeight: isWinner ? 700 : 400
                          }}>
                            {val.toFixed(4)}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>

              <div style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5, textAlign: "left", width: "100%", marginTop: 8 }}>
                <strong>Quy tắc Winner-take-all:</strong>
                <br />
                • Chỉ phần tử lớn nhất trong vùng 2x2 lúc forward (giá trị <strong>{maxForwardVal.toFixed(4)}</strong> tại chỉ số [{r * 2 + maxOffsetRow}, {c * 2 + maxOffsetCol}]) có đóng góp vào đầu ra.
                <br />
                • Do đó, gradient lỗi nhận được {poolGradVal.toFixed(6)} sẽ chạy <strong>nguyên vẹn</strong> về vị trí này. Ba vị trí còn lại nhận gradient bằng 0.
              </div>
            </div>
          </div>
        </div>

        {/* Right: Output Gradient 26x26 */}
        <div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>2. Gradient truyền ngược về Conv (26×26)</div>
          <MatrixGrid
            data={convGradSlice}
            cellSize={11}
            colorScheme="heatmap"
            label={`grad_conv (F${backwardFilter})`}
            highlightRegion={{ row: r * 2, col: c * 2, size: 2 }}
          />
        </div>
      </div>
    </div>
  );
}
