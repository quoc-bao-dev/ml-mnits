import { Select, Tabs } from "antd";
import { useState } from "react";
import MatrixGrid from "../common/MatrixGrid";

interface ConvBackwardProps {
  data: any;
  step: any;
}

export default function ConvBackward({ data, step }: ConvBackwardProps) {
  const [convBackTab, setConvBackTab] = useState("interactive");
  const [backwardFilter, setBackwardFilter] = useState(0);
  const [selectedConvBackWeight, setSelectedConvBackWeight] = useState<{ row: number, col: number } | null>({ row: 0, col: 0 });

  const filtersBefore = step.filters_before; // 8x3x3
  const filtersAfter = step.filters_after; // 8x3x3
  const filterDeltas = step.filter_deltas; // 8x3x3
  const lr = step.learning_rate || data.learning_rate;
  const gradConv = data.steps[5]?.grad_conv; // 26x26x8

  if (!filtersBefore || !filtersAfter || !filterDeltas || !gradConv) {
    return <div>Thiếu dữ liệu log để trực quan hóa Conv backward.</div>;
  }

  const fIdx = backwardFilter;
  const m = selectedConvBackWeight?.row ?? 0;
  const n = selectedConvBackWeight?.col ?? 0;

  const filterBefore = filtersBefore[fIdx];
  const filterAfter = filtersAfter[fIdx];
  const delta = filterDeltas[fIdx];

  // Gradient is -delta / lr
  const filterGrad = delta.map((row: number[]) => row.map((v: number) => -v / lr));

  const gradConvSlice = gradConv.map((row: number[][]) => row.map(cell => cell[fIdx]));

  // Get 26x26 slice of input image starting at (m, n)
  const getSubImage = (img: number[][], startRow: number, startCol: number) => {
    const sub: number[][] = [];
    for (let r = 0; r < 26; r++) {
      const rowVals: number[] = [];
      for (let c = 0; c < 26; c++) {
        rowVals.push(img[startRow + r]?.[startCol + c] ?? 0);
      }
      sub.push(rowVals);
    }
    return sub;
  };

  const subImg = getSubImage(data.input_image, m, n);
  const gradVal = filterGrad[m][n];
  const wBeforeVal = filterBefore[m][n];
  const wAfterVal = filterAfter[m][n];

  const tabItems = [
    {
      key: "interactive",
      label: "1. Chi tiết & Tích chập Gradient",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Controls */}
          <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Bộ lọc đang hiển thị:</span>
            <Select value={backwardFilter} onChange={setBackwardFilter} style={{ width: 140 }}
              options={Array.from({ length: 8 }, (_, i) => ({ value: i, label: `Filter ${i}` }))} />
          </div>

          {/* 3x3 Matrices comparison */}
          <div className="glass-card" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
              So sánh bộ lọc {fIdx} trước & sau cập nhật (η = {lr})
            </div>

            <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <MatrixGrid
                  data={filterBefore}
                  cellSize={38}
                  colorScheme="heatmap"
                  showValues
                  selectedCell={selectedConvBackWeight}
                  onCellClick={(row, col) => setSelectedConvBackWeight({ row, col })}
                  label="Trọng số W (Trước)"
                />
              </div>
              <div style={{ fontSize: 20, color: "var(--text-muted)", fontWeight: 700 }}>-</div>
              <div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <MatrixGrid
                    data={filterGrad}
                    cellSize={38}
                    colorScheme="heatmap"
                    showValues
                    selectedCell={selectedConvBackWeight}
                    onCellClick={(row, col) => setSelectedConvBackWeight({ row, col })}
                    label="Gradient (∂L/∂W)"
                    borderColor="#f59e0b"
                  />
                </div>
              </div>
              <div style={{ fontSize: 20, color: "var(--text-muted)", fontWeight: 700 }}>=</div>
              <div>
                <MatrixGrid
                  data={filterAfter}
                  cellSize={38}
                  colorScheme="heatmap"
                  showValues
                  selectedCell={selectedConvBackWeight}
                  onCellClick={(row, col) => setSelectedConvBackWeight({ row, col })}
                  label="Trọng số W (Sau)"
                />
              </div>
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center" }}>
              Click vào ô bất kỳ trong bộ lọc 3×3 để xem chi tiết tích chập gradient tạo ra giá trị đó
            </div>
          </div>

          {/* Detailed cross-correlation viz */}
          {selectedConvBackWeight && (
            <div className="glass-card" style={{ padding: 20 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--warning)", marginBottom: 12 }}>
                Cách tính Gradient tại ô [{m}, {n}] của Bộ lọc {fIdx}
              </div>

              <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 16 }}>
                Giá trị gradient được tính bằng cách trượt ma trận lỗi 26 × 26 trên ảnh đầu vào, bắt đầu dịch chuyển một khoảng m = {m}, n = {n}:
                <br />
                <code style={{ color: "var(--accent-light)", fontSize: 13, fontFamily: "monospace" }}>
                  ∂L/∂W[{m}, {n}] = Σ_(i=0..25) Σ_(j=0..25) [ Ảnh[i + {m}, j + {n}] · ∂L/∂Y[i, j] ]
                </code>
              </div>

              <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
                {/* Sliced Input Image 26x26 */}
                <div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>Ảnh đầu vào dịch chuyển [{m}, {n}] (26×26)</div>
                  <MatrixGrid
                    data={subImg}
                    cellSize={10}
                    colorScheme="grayscale"
                    label={`Image [${m}:${m + 26}, ${n}:${n + 26}]`}
                  />
                </div>

                <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", alignSelf: "center" }}>
                  <span style={{ fontSize: 20, color: "var(--text-muted)" }}>×</span>
                  <span style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>Nhân chập</span>
                </div>

                {/* Gradient map 26x26 */}
                <div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>Ma trận lỗi gradient (26×26)</div>
                  <MatrixGrid
                    data={gradConvSlice}
                    cellSize={10}
                    colorScheme="heatmap"
                    label={`grad_conv (Filter ${fIdx})`}
                  />
                </div>

                <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", alignSelf: "center" }}>
                  <span style={{ fontSize: 20, color: "var(--text-muted)" }}>➔</span>
                </div>

                {/* Formula explanation and cell update result */}
                <div className="glass-card" style={{ flex: 1, minWidth: 280, padding: 16, height: "auto" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 12 }}>
                    Kết quả tính toán & Cập nhật
                  </div>

                  <div style={{ fontFamily: "monospace", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.8 }}>
                    • Gradient: <br />
                    &nbsp;&nbsp;∂L/∂W[{m},{n}] = <strong style={{ color: "var(--warning)" }}>{gradVal.toFixed(6)}</strong>
                    <br /><br />
                    • Phép toán cập nhật trọng số:<br />
                    &nbsp;&nbsp;W_new = W_old - lr × Gradient<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {wBeforeVal.toFixed(6)} - {lr} × {gradVal.toFixed(6)}<br />
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong style={{ color: "var(--success)" }}>{wAfterVal.toFixed(6)}</strong>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* All filters preview */}
          <div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>Xem nhanh thay đổi của cả 8 bộ lọc (Nhấp chọn để trực quan hóa chi tiết)</div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {filtersBefore.map((_: number[][], idx: number) => {
                const filterDelta = filterDeltas[idx];
                const maxDiff = Math.max(...filterDelta.flat().map(Math.abs));
                return (
                  <div
                    key={idx}
                    onClick={() => setBackwardFilter(idx)}
                    style={{
                      cursor: "pointer",
                      outline: idx === fIdx ? "2px solid var(--warning)" : "1px solid var(--border)",
                      borderRadius: 8,
                      padding: 8,
                      background: idx === fIdx ? "rgba(245, 158, 11, 0.05)" : "transparent",
                      transition: "all 0.2s"
                    }}
                  >
                    <div style={{ fontSize: 11, fontWeight: 600, color: idx === fIdx ? "var(--warning)" : "var(--text-primary)", marginBottom: 4 }}>Filter {idx}</div>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      <MatrixGrid data={filtersBefore[idx]} cellSize={8} colorScheme="heatmap" />
                      <span style={{ fontSize: 10, color: "var(--text-muted)" }}>➔</span>
                      <MatrixGrid data={filtersAfter[idx]} cellSize={8} colorScheme="heatmap" />
                    </div>
                    <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 4, textAlign: "center" }}>
                      |Δ_max|: {maxDiff.toFixed(6)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )
    },
    {
      key: "comparison",
      label: "2. So sánh Toàn bộ Ma trận Thay đổi (Deltas)",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Bảng so sánh chi tiết các ma trận bộ lọc trước và sau khi tối ưu hóa. Ma trận hiệu số (Delta) ΔW = W_sau - W_truoc được hiển thị với màu sắc phân kỳ:
            <br />
            • Màu <span style={{ color: "var(--success)", fontWeight: 600 }}>Xanh lá</span> đại diện cho trọng số <strong>tăng lên</strong> (ΔW &gt; 0).
            <br />
            • Màu <span style={{ color: "var(--danger)", fontWeight: 600 }}>Đỏ</span> đại diện cho trọng số <strong>giảm đi</strong> (ΔW &lt; 0).
            <br />
            • Màu <strong>Xám đen</strong> đại diện cho trọng số hầu như <strong>không đổi</strong> (ΔW ≈ 0).
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {filtersBefore.map((_: number[][], idx: number) => {
              const fBefore = filtersBefore[idx];
              const fAfter = filtersAfter[idx];
              const fDelta = filterDeltas[idx];

              // Calculate stats
              const flatDelta = fDelta.flat();
              const absDeltas = flatDelta.map(Math.abs);
              const maxAbs = Math.max(...absDeltas);
              const sumAbs = absDeltas.reduce((a: number, b: number) => a + b, 0);
              const avgAbs = sumAbs / 9;

              return (
                <div key={idx} className="glass-card" style={{ padding: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, borderBottom: "1px solid var(--border)", paddingBottom: 8 }}>
                    <span style={{ fontSize: 14, fontWeight: 700, color: "var(--warning)" }}>Bộ lọc {idx}</span>
                    <div style={{ display: "flex", gap: 16, fontSize: 11, color: "var(--text-secondary)" }}>
                      <span>L1 Norm (Tổng Δ tuyệt đối): <strong style={{ color: "var(--text-primary)" }}>{sumAbs.toFixed(6)}</strong></span>
                      <span>Độ lệch trung bình: <strong style={{ color: "var(--text-primary)" }}>{avgAbs.toFixed(6)}</strong></span>
                      <span>Độ lệch lớn nhất |Δ_max|: <strong style={{ color: "var(--warning)" }}>{maxAbs.toFixed(6)}</strong></span>
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "space-around", alignItems: "center" }}>
                    {/* Before */}
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <MatrixGrid data={fBefore} cellSize={26} colorScheme="heatmap" showValues label="Trước (W_old)" />
                    </div>

                    <div style={{ fontSize: 16, color: "var(--text-muted)", fontWeight: 700 }}>+</div>

                    {/* Delta */}
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <MatrixGrid
                        data={fDelta}
                        cellSize={26}
                        colorScheme="diverging"
                        showValues
                        minVal={-maxAbs}
                        maxVal={maxAbs}
                        label="Hiệu số (ΔW)"
                        borderColor="rgba(245, 158, 11, 0.3)"
                      />
                    </div>

                    <div style={{ fontSize: 16, color: "var(--text-muted)", fontWeight: 700 }}>➔</div>

                    {/* After */}
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <MatrixGrid data={fAfter} cellSize={26} colorScheme="heatmap" showValues label="Sau (W_new)" />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )
    }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Tabs activeKey={convBackTab} onChange={setConvBackTab} items={tabItems} type="card" />
    </div>
  );
}
