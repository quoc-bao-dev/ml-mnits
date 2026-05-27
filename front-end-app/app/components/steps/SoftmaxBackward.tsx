import { ArrowLeftOutlined } from "@ant-design/icons";
import { Select, Tabs } from "antd";
import { useState } from "react";
import MatrixGrid from "../common/MatrixGrid";

interface SoftmaxBackwardProps {
  data: any;
  step: any;
}

export default function SoftmaxBackward({ data, step }: SoftmaxBackwardProps) {
  const [softmaxBackTab, setSoftmaxBackTab] = useState("error_vector");
  const [backwardFilter, setBackwardFilter] = useState(0);
  const [selectedSoftmaxBackCell, setSelectedSoftmaxBackCell] = useState<{ row: number, col: number } | null>({ row: 0, col: 0 });

  const probs: number[] = data.steps[2]?.probabilities || [];
  const label = data.label;
  const initialGrad = step.initial_gradient || [];
  const dLdz = step.d_L_d_z || [];
  const lr = data.learning_rate;
  const gradPool = step.grad_pool; // 13x13x8

  // Tab Items for Softmax Backward
  const items = [
    {
      key: "error_vector",
      label: "1. Vector lỗi (∂L/∂z = p - y)",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Sự kết hợp giữa <strong>Softmax</strong> và <strong>Cross-Entropy</strong> cho ra một công thức đạo hàm cực kỳ tối giản đối với Logits (z):
            <br />
            <code style={{ color: "var(--accent-light)", fontSize: 14 }}>
              ∂L/∂z_i = p_i - y_i
            </code>
            &nbsp;(trong đó y là one-hot vector nhãn đúng).
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)", color: "var(--text-muted)" }}>
                  <th style={{ padding: "8px 12px" }}>Lớp (Class)</th>
                  <th style={{ padding: "8px 12px" }}>Xác suất p_i</th>
                  <th style={{ padding: "8px 12px" }}>Nhãn đúng y_i</th>
                  <th style={{ padding: "8px 12px" }}>Phép tính (p_i - y_i)</th>
                  <th style={{ padding: "8px 12px" }}>Gradient ∂L/∂z_i</th>
                  <th style={{ padding: "8px 12px" }}>Ý nghĩa toán học</th>
                </tr>
              </thead>
              <tbody>
                {dLdz.map((gradVal: number, i: number) => {
                  const isTruth = i === label;
                  let explanation = "";
                  let valColor = "var(--text-primary)";

                  if (isTruth) {
                    explanation = "Cần TĂNG logit z này (gradient < 0)";
                    valColor = "var(--danger)";
                  } else if (probs[i] > 0.05) {
                    explanation = "Cần GIẢM logit z này (gradient > 0)";
                    valColor = "var(--warning)";
                  } else {
                    explanation = "Đã gần bằng 0, ít thay đổi";
                    valColor = "var(--text-muted)";
                  }

                  return (
                    <tr key={i} style={{ borderBottom: "1px solid var(--border)", background: isTruth ? "rgba(16, 185, 129, 0.05)" : "transparent" }}>
                      <td style={{ padding: "8px 12px", fontWeight: 600 }}>{i} {isTruth && "✅"}</td>
                      <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>{probs[i]?.toFixed(6)}</td>
                      <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>{isTruth ? "1.0" : "0.0"}</td>
                      <td style={{ padding: "8px 12px", fontFamily: "monospace", color: "var(--text-muted)" }}>
                        {probs[i]?.toFixed(4)} - {isTruth ? "1.0" : "0.0"}
                      </td>
                      <td style={{ padding: "8px 12px", fontFamily: "monospace", fontWeight: 700, color: valColor }}>
                        {gradVal > 0 ? "+" : ""}{gradVal.toFixed(6)}
                      </td>
                      <td style={{ padding: "8px 12px", fontSize: 11, fontStyle: "italic", color: "var(--text-secondary)" }}>
                        {explanation}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="glass-card" style={{ padding: 12, fontSize: 12, color: "var(--text-secondary)" }}>
            <strong>Tại sao lại âm/dương?</strong>
            <br />
            • Tại nhãn đúng {label}, gradient bằng {dLdz[label]?.toFixed(4)} (âm). Theo Gradient Descent, ta cập nhật z_label ← z_label - η · gradient, do đó z_label sẽ <strong>tăng lên</strong>.
            <br />
            • Tại các nhãn sai khác, gradient mang giá trị dương. Khi trừ đi gradient dương, các logits tương ứng sẽ bị <strong>giảm đi</strong>.
          </div>
        </div>
      )
    },
    {
      key: "weights_update",
      label: "2. Cập nhật Trọng số & Bias",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Lớp Fully Connected chứa ma trận trọng số W kích thước 1352 × 10. Với vector đầu vào x (size 1352) và gradient ∂L/∂z (size 10), công thức cập nhật cho mỗi trọng số là:
            <br />
            <code style={{ color: "var(--accent-light)", fontSize: 14 }}>
              W_[j, i] = W_[j, i] - η · x_j · (∂L/∂z_i)
            </code>
          </div>

          {/* 10x10 slice simulation */}
          {(() => {
            const xFlat = data.steps[1]?.output.flat(2) || [];

            // We construct a stable mock weights matrix slice for visualization
            const getWeightVal = (j: number, i: number) => {
              const val = Math.sin(j * 0.7 + i * 0.3) * 0.1;
              return val;
            };

            // We render a grid of weight updates
            const gridData: number[][] = [];
            const updatedGridData: number[][] = [];
            const weightGrads: number[][] = [];

            for (let j = 0; j < 10; j++) {
              const rowWeights: number[] = [];
              const rowUpdated: number[] = [];
              const rowGrads: number[] = [];
              const xj = xFlat[j] || 0;
              for (let i = 0; i < 10; i++) {
                const w_old = getWeightVal(j, i);
                const dLdz_i = dLdz[i] || 0;
                const dLdW = xj * dLdz_i;
                const w_new = w_old - lr * dLdW;
                rowWeights.push(w_old);
                rowGrads.push(dLdW);
                rowUpdated.push(w_new);
              }
              gridData.push(rowWeights);
              weightGrads.push(rowGrads);
              updatedGridData.push(rowUpdated);
            }

            return (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "space-between" }}>
                  <div>
                    <MatrixGrid data={gridData} cellSize={24} colorScheme="heatmap" label="W trước (10×10 slice)" showValues />
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
                    <span style={{ fontSize: 24, color: "var(--text-muted)" }}>-</span>
                    <span style={{ fontSize: 12, color: "var(--accent)" }}>η = {lr}</span>
                  </div>
                  <div>
                    <MatrixGrid data={weightGrads} cellSize={24} colorScheme="heatmap" label="Gradient (x_j · ∂L/∂z_i)" showValues />
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
                    <span style={{ fontSize: 24, color: "var(--text-muted)" }}>=</span>
                  </div>
                  <div>
                    <MatrixGrid data={updatedGridData} cellSize={24} colorScheme="heatmap" label="W sau cập nhật" showValues />
                  </div>
                </div>

                <div className="glass-card" style={{ padding: 16 }}>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: 8, fontSize: 13 }}>Ví dụ tính toán bias (b) - kích thước 10:</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10 }}>
                    {dLdz.map((gradVal: number, i: number) => {
                      const b_old = i * 0.01;
                      const b_new = b_old - lr * gradVal;
                      return (
                        <div key={i} style={{ padding: 10, background: "rgba(255, 255, 255, 0.02)", borderRadius: 6, border: "1px solid var(--border)", fontSize: 11 }}>
                          <strong>b<sub>{i}</sub>:</strong> {b_old.toFixed(3)} → <strong>{b_new.toFixed(4)}</strong>
                          <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 4 }}>
                            Δ = -{lr} × {gradVal.toFixed(3)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )
    },
    {
      key: "reshape_grad",
      label: "3. Reshape Gradient (∂L/∂x)",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Để truyền ngược gradient về MaxPool, ta tính gradient đầu vào lớp Linear: ∂L/∂x = W^T · ∂L/∂z (vector 1D kích thước <strong>1352</strong>).
            Sau đó, vector này được <strong>Reshape ngược (fold)</strong> thành ma trận 3D (13 × 13 × 8).
          </div>

          {gradPool ? (() => {
            const flatGrad = gradPool.flat(2);
            const row = selectedSoftmaxBackCell?.row ?? 0;
            const col = selectedSoftmaxBackCell?.col ?? 0;
            const flatIndex = row * 104 + col * 8 + backwardFilter;

            const render1DGradSnippet = (idx: number, arr: number[]) => {
              const startIdx = Math.max(0, idx - 4);
              const endIdx = Math.min(arr.length - 1, idx + 4);

              const elements = [];
              if (startIdx > 0) elements.push({ isEllipsis: true, key: "start-ell", text: `... ${startIdx} phần tử` });

              for (let i = startIdx; i <= endIdx; i++) {
                elements.push({ index: i, val: arr[i], isSelected: i === idx, key: `elem-${i}` });
              }

              if (endIdx < arr.length - 1) elements.push({ isEllipsis: true, key: "end-ell", text: `${arr.length - 1 - endIdx} phần tử ...` });

              return (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, overflowX: "auto", padding: "8px 0" }}>
                  {elements.map((el: any) => {
                    if (el.isEllipsis) {
                      return <div key={el.key} style={{ padding: "0 10px", color: "var(--text-muted)", fontSize: 11, fontStyle: "italic" }}>{el.text}</div>;
                    }
                    return (
                      <div key={el.key} style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: 70 }}>
                        <div style={{ fontSize: 10, color: el.isSelected ? "var(--warning)" : "var(--text-muted)", marginBottom: 4, fontFamily: "monospace" }}>
                          [{el.index}]
                        </div>
                        <div style={{
                          padding: "8px 4px",
                          background: el.isSelected ? "rgba(245, 158, 11, 0.15)" : "rgba(255, 255, 255, 0.03)",
                          border: `1px solid ${el.isSelected ? "var(--warning)" : "var(--border)"}`,
                          borderRadius: 6,
                          fontFamily: "monospace",
                          fontSize: 11,
                          color: el.isSelected ? "var(--warning)" : "var(--text-primary)",
                          textAlign: "center",
                          width: "100%"
                        }}>
                          {el.val?.toFixed(5)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            };

            const currentFilterSlice = gradPool.map((r: number[][]) => r.map(c => c[backwardFilter]));

            return (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                <div className="glass-card" style={{ padding: 20, display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ fontWeight: 600, marginBottom: 12 }}>1. Vector 1D Gradient (1352 phần tử)</div>
                  <div style={{ width: "100%", maxWidth: 800 }}>
                    {render1DGradSnippet(flatIndex, flatGrad)}
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", color: "var(--warning)", margin: "16px 0" }}>
                    <ArrowLeftOutlined style={{ fontSize: 20, transform: "rotate(90deg)" }} />
                    <span style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, marginTop: 4 }}>Reshape ngược về 3D (13×13×8)</span>
                  </div>

                  <div style={{ fontWeight: 600, marginBottom: 12 }}>2. Ma trận Gradient 2D theo bộ lọc (13×13)</div>
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
                    {Array.from({ length: 8 }).map((_, f) => (
                      <div
                        key={f}
                        onClick={() => setBackwardFilter(f)}
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          gap: 4,
                          cursor: "pointer",
                          padding: 6,
                          borderRadius: 8,
                          background: backwardFilter === f ? "rgba(245, 158, 11, 0.1)" : "transparent",
                          border: backwardFilter === f ? "1px solid var(--warning)" : "1px solid transparent",
                          transition: "all 0.2s"
                        }}
                      >
                        <div style={{ fontSize: 10, fontWeight: 600, color: backwardFilter === f ? "var(--warning)" : "var(--text-muted)" }}>Filter {f}</div>
                        <MatrixGrid
                          data={gradPool.map((r: number[][]) => r.map(c => c[f]))}
                          cellSize={4}
                          colorScheme="heatmap"
                          selectedCell={selectedSoftmaxBackCell && backwardFilter === f ? selectedSoftmaxBackCell : null}
                        />
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
                  <div>
                    <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 600 }}>Chi tiết Filter {backwardFilter}:</span>
                      <Select value={backwardFilter} onChange={setBackwardFilter} size="small" style={{ width: 100 }}
                        options={Array.from({ length: 8 }, (_, i) => ({ value: i, label: `F${i}` }))} />
                    </div>
                    <MatrixGrid
                      data={currentFilterSlice}
                      cellSize={24}
                      colorScheme="heatmap"
                      selectedCell={selectedSoftmaxBackCell}
                      onCellClick={(r, c) => setSelectedSoftmaxBackCell({ row: r, col: c })}
                      label={`Gradient 13×13 – Filter ${backwardFilter}`}
                    />
                  </div>
                  {selectedSoftmaxBackCell && (
                    <div className="glass-card" style={{ flex: 1, minWidth: 260, padding: 16 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 12 }}>
                        Ánh xạ phần tử được chọn
                      </div>
                      <div style={{ fontFamily: "monospace", fontSize: 12, color: "var(--text-secondary)", lineHeight: 2 }}>
                        Tọa độ 3D: [Hàng <strong>{row}</strong>, Cột <strong>{col}</strong>, Filter <strong>{backwardFilter}</strong>]
                        <br />
                        Giá trị Gradient: <strong style={{ color: "var(--warning)" }}>{gradPool[row][col][backwardFilter]?.toFixed(6)}</strong>
                        <br /><br />
                        Công thức giải nén vị trí index 1D:<br />
                        Index = (Hàng × 104) + (Cột × 8) + Filter<br />
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ({row} × 104) + ({col} × 8) + {backwardFilter}<br />
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong style={{ color: "var(--warning)", fontSize: 14 }}>{flatIndex}</strong>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })() : (
            <div>Không tìm thấy dữ liệu `grad_pool` trong log.</div>
          )}
        </div>
      )
    }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Tabs activeKey={softmaxBackTab} onChange={setSoftmaxBackTab} items={items} type="card" />
    </div>
  );
}
