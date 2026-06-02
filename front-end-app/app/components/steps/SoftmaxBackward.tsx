"use client";

import { ArrowLeftOutlined, ArrowRightOutlined } from "@ant-design/icons";
import { Select, Tabs } from "antd";
import { useState } from "react";
import FormulaPopover from "../common/FormulaPopover";
import MatrixGrid from "../common/MatrixGrid";
import TeX from "../common/TeX";

interface SoftmaxBackwardProps {
  data: any;
  step: any;
}

export default function SoftmaxBackward({ data, step }: SoftmaxBackwardProps) {
  const [softmaxBackTab, setSoftmaxBackTab] = useState("error_vector");
  const [backwardFilter, setBackwardFilter] = useState(0);
  const [selectedSoftmaxBackCell, setSelectedSoftmaxBackCell] = useState<{ row: number, col: number } | null>({ row: 0, col: 0 });
  const [selectedWeightCell, setSelectedWeightCell] = useState<{ row: number; col: number }>({ row: 0, col: 0 });

  const probs: number[] = data.steps[2]?.probabilities || [];
  const label = data.label;
  const dLdz: number[] = step.d_L_d_z || [];
  const lr: number = step.learning_rate ?? data.learning_rate;
  const gradPool = step.grad_pool;

  // Dữ liệu thật về 3 thành phần thay đổi
  const wBefore: number[][] = step.sm_weights_before || [];
  const wAfter: number[][] = step.sm_weights_after || [];
  const wDelta: number[][] = step.sm_weights_delta || [];
  const xFlat: number[] = step.sm_last_input || [];
  const sliceRows: number = step.sm_weights_slice_rows || 20;
  const totalRows: number = step.sm_weights_total_rows || 1352;

  const bBefore: number[] = step.sm_biases_before || [];
  const bAfter: number[] = step.sm_biases_after || [];
  const bDelta: number[] = step.sm_biases_delta || [];

  // ════════════════════════════════════════════════════════════
  // TAB 1 — Vector lỗi (∂L/∂z = p − y)
  // ════════════════════════════════════════════════════════════
  const renderErrorVector = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span>Khi ghép Softmax + Cross-Entropy, đạo hàm Loss theo logits có công thức cực gọn:</span>
        <TeX math={"\\dfrac{\\partial L}{\\partial z_i} = p_i - y_i"} />
        <FormulaPopover
          title="Chứng minh ∂L/∂z = p − y"
          generalFormula={"\\frac{\\partial L}{\\partial z_j} = \\sum_i \\frac{\\partial L}{\\partial p_i} \\cdot \\frac{\\partial p_i}{\\partial z_j}"}
          substitutedFormula={"= -\\frac{1}{p_{label}} \\cdot \\frac{\\partial p_{label}}{\\partial z_j}"}
          result={"= \\begin{cases} p_j - 1 & j = label \\\\ p_j & j \\neq label \\end{cases} = p_j - y_j"}
          note="Đây là lý do PyTorch/TF luôn ghép Softmax + CE thành 1 hàm — giúp tránh tính Jacobian đầy đủ."
        />
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, textAlign: "left" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", color: "var(--text-muted)" }}>
              <th style={{ padding: "8px 12px" }}>Lớp</th>
              <th style={{ padding: "8px 12px" }}>p<sub>i</sub></th>
              <th style={{ padding: "8px 12px" }}>y<sub>i</sub></th>
              <th style={{ padding: "8px 12px" }}>Phép tính p<sub>i</sub> − y<sub>i</sub></th>
              <th style={{ padding: "8px 12px" }}>∂L/∂z<sub>i</sub></th>
              <th style={{ padding: "8px 12px" }}>Hiệu ứng cập nhật</th>
            </tr>
          </thead>
          <tbody>
            {dLdz.map((gradVal, i) => {
              const isTruth = i === label;
              let valColor = "var(--text-primary)";
              let explanation = "";
              if (isTruth) {
                valColor = "var(--danger)";
                explanation = "Gradient âm → bias tăng → tăng xác suất lớp này";
              } else if (probs[i] > 0.05) {
                valColor = "var(--warning)";
                explanation = "Gradient dương → bias giảm → giảm xác suất lớp này";
              } else {
                valColor = "var(--text-muted)";
                explanation = "Đã gần 0, ít thay đổi";
              }
              return (
                <tr key={i} style={{ borderBottom: "1px solid var(--border)", background: isTruth ? "rgba(16,185,129,0.05)" : "transparent" }}>
                  <td style={{ padding: "8px 12px", fontWeight: 600 }}>{i} {isTruth && "✅"}</td>
                  <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>{probs[i]?.toFixed(6)}</td>
                  <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>{isTruth ? "1.0" : "0.0"}</td>
                  <td style={{ padding: "8px 12px", fontFamily: "monospace", color: "var(--text-muted)" }}>
                    {probs[i]?.toFixed(4)} − {isTruth ? "1.0" : "0.0"}
                  </td>
                  <td style={{ padding: "8px 12px", fontFamily: "monospace", fontWeight: 700, color: valColor }}>
                    {gradVal > 0 ? "+" : ""}{gradVal.toFixed(6)}
                  </td>
                  <td style={{ padding: "8px 12px", fontSize: 11, fontStyle: "italic", color: "var(--text-secondary)" }}>{explanation}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );

  // ════════════════════════════════════════════════════════════
  // TAB 2 — Bias: ma trận thật trước → sau
  // ════════════════════════════════════════════════════════════
  const renderBiasUpdate = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span>Bias là vector 10 phần tử (1 cho mỗi lớp). Công thức cập nhật:</span>
        <TeX math={"\\mathbf{b} \\leftarrow \\mathbf{b} - \\eta \\cdot \\dfrac{\\partial L}{\\partial \\mathbf{z}}"} />
        <FormulaPopover
          title="Vì sao bias gradient = ∂L/∂z?"
          generalFormula={"\\frac{\\partial z_c}{\\partial b_c} = 1 \\quad \\Rightarrow \\quad \\frac{\\partial L}{\\partial b_c} = \\frac{\\partial L}{\\partial z_c}"}
          note="Bias chỉ cộng thẳng vào logit nên đạo hàm theo bias bằng 1 → Chain Rule chỉ là 'nhân với 1' → giữ nguyên gradient logits."
        />
      </div>

      {/* Hàng ngang: Bias trước → Δ → Bias sau */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text-primary)" }}>
          So sánh bias trước & sau cập nhật (η = {lr})
        </div>

        {[
          { title: "b_trước", values: bBefore, color: "#6366f1" },
          { title: "Δb = −η · ∂L/∂z", values: bDelta, color: "#f59e0b" },
          { title: "b_sau = b_trước + Δb", values: bAfter, color: "#10b981" },
        ].map((row, idx) => (
          <div key={idx} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 200, fontSize: 12, color: row.color, fontWeight: 600, fontFamily: "monospace" }}>
              {row.title}
            </div>
            <div style={{ display: "flex", gap: 4 }}>
              {row.values.map((v, i) => {
                const isTruth = i === label;
                const isPositive = v > 0;
                const isNegative = v < 0;
                let bg = "rgba(255,255,255,0.04)";
                if (idx === 1) {
                  bg = isPositive ? "rgba(16,185,129,0.15)" : isNegative ? "rgba(239,68,68,0.15)" : "rgba(255,255,255,0.04)";
                }
                return (
                  <FormulaPopover
                    key={i}
                    title={`b[${i}] cập nhật`}
                    placement="top"
                    generalFormula={`b_{${i}} \\leftarrow b_{${i}} - \\eta \\cdot \\dfrac{\\partial L}{\\partial z_{${i}}}`}
                    substitutedFormula={`= ${bBefore[i]?.toFixed(6)} - ${lr} \\times ${dLdz[i]?.toFixed(6)}`}
                    result={`= ${bAfter[i]?.toFixed(6)}`}
                    note={isTruth ? `Lớp ${i} là nhãn đúng → bias tăng để mạng "nghiêng" về lớp này hơn ở lần forward sau.` : `Lớp ${i} không phải nhãn → bias giảm nhẹ để mạng "nghiêng đi" khỏi lớp này.`}
                  >
                    <div style={{
                      minWidth: 70,
                      padding: "6px 8px",
                      background: bg,
                      border: `1px solid ${isTruth ? "var(--accent)" : "var(--border)"}`,
                      borderRadius: 6,
                      fontFamily: "monospace",
                      fontSize: 11,
                      textAlign: "center",
                      cursor: "pointer",
                      position: "relative",
                    }}>
                      <div style={{ fontSize: 9, color: "var(--text-muted)", marginBottom: 2 }}>
                        b[{i}]{isTruth && " ⭐"}
                      </div>
                      <div style={{ color: idx === 1 ? row.color : "var(--text-primary)", fontWeight: 600 }}>
                        {v >= 0 ? "+" : ""}{v.toFixed(4)}
                      </div>
                    </div>
                  </FormulaPopover>
                );
              })}
            </div>
          </div>
        ))}

        <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", marginTop: 4 }}>
          Bấm vào mỗi ô để xem công thức + thay số cụ thể tại vị trí đó
        </div>
      </div>

      {/* Highlight ô bias của lớp đúng */}
      <div className="glass-card" style={{ padding: 14, background: "rgba(16,185,129,0.04)" }}>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.8 }}>
          <strong style={{ color: "var(--success)" }}>Quan sát:</strong> bias của lớp đúng (số {label}) có
          Δ = <strong style={{ color: "var(--success)", fontFamily: "monospace" }}>{bDelta[label] >= 0 ? "+" : ""}{bDelta[label]?.toFixed(6)}</strong>
          {" "}(dương, lớn nhất). Các bias khác đều âm với độ lớn nhỏ (≈ −0.0005). Đây là minh chứng:
          <strong> mạng đã &ldquo;học&rdquo; được hướng đúng chỉ sau 1 lần cập nhật.</strong>
        </div>
      </div>
    </div>
  );

  // ════════════════════════════════════════════════════════════
  // TAB 3 — Weights: 3D thật (slice 20×10) trước → sau
  // ════════════════════════════════════════════════════════════
  const renderWeightsUpdate = () => {
    const j = selectedWeightCell.row;   // pixel index
    const i = selectedWeightCell.col;   // class index
    const xj = xFlat[j] ?? 0;
    const dLdz_i = dLdz[i] ?? 0;
    const gradWji = xj * dLdz_i;
    const wOld = wBefore[j]?.[i] ?? 0;
    const wNew = wAfter[j]?.[i] ?? 0;
    const deltaW = wDelta[j]?.[i] ?? 0;

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span>Ma trận W kích thước {totalRows} × 10. Công thức cập nhật outer product:</span>
          <TeX math={"\\dfrac{\\partial L}{\\partial W} = \\mathbf{x} \\otimes \\dfrac{\\partial L}{\\partial \\mathbf{z}}"} />
          <FormulaPopover
            title="Vì sao outer product?"
            generalFormula={"\\frac{\\partial L}{\\partial W_{j,i}} = \\frac{\\partial L}{\\partial z_i} \\cdot \\frac{\\partial z_i}{\\partial W_{j,i}}"}
            substitutedFormula={"\\frac{\\partial z_i}{\\partial W_{j,i}} = x_j \\quad (\\text{vì } z_i = \\sum_j W_{j,i} x_j + b_i)"}
            result={"\\frac{\\partial L}{\\partial W_{j,i}} = x_j \\cdot (p_i - y_i)"}
            note="Mỗi phần tử của ma trận gradient là tích của 1 pixel input với 1 phần tử của vector (p − y) — đây là định nghĩa outer product."
          />
        </div>

        <div style={{ fontSize: 11, color: "var(--warning)", padding: 8, background: "rgba(245,158,11,0.08)", borderRadius: 4, border: "1px solid rgba(245,158,11,0.2)" }}>
          ⚠ Ma trận thật có {totalRows} hàng. Để xem được, ta đang hiển thị <strong>{sliceRows} hàng đầu</strong> (ứng với {sliceRows} pixel đầu của vector x đã flatten).
        </div>

        {/* 3 ma trận: trước - Δ - sau */}
        <div className="glass-card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
            Ma trận W slice {sliceRows} × 10 — thật từ training (η = {lr})
          </div>
          <div style={{ display: "flex", gap: 16, alignItems: "center", justifyContent: "center", flexWrap: "wrap" }}>
            <div>
              <MatrixGrid
                data={wBefore}
                cellSize={28}
                colorScheme="heatmap"
                showValues
                selectedCell={selectedWeightCell}
                onCellClick={(r, c) => setSelectedWeightCell({ row: r, col: c })}
                label="W trước (W_old)"
              />
            </div>
            <div style={{ fontSize: 22, color: "var(--text-muted)", fontWeight: 700 }}>+</div>
            <div>
              <MatrixGrid
                data={wDelta}
                cellSize={28}
                colorScheme="diverging"
                showValues
                selectedCell={selectedWeightCell}
                onCellClick={(r, c) => setSelectedWeightCell({ row: r, col: c })}
                label="ΔW = −η · ∂L/∂W"
                borderColor="rgba(245,158,11,0.5)"
              />
            </div>
            <div style={{ fontSize: 22, color: "var(--text-muted)", fontWeight: 700 }}>=</div>
            <div>
              <MatrixGrid
                data={wAfter}
                cellSize={28}
                colorScheme="heatmap"
                showValues
                selectedCell={selectedWeightCell}
                onCellClick={(r, c) => setSelectedWeightCell({ row: r, col: c })}
                label="W sau (W_new)"
              />
            </div>
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", marginTop: 8 }}>
            Click vào bất kỳ ô nào để xem công thức tính chi tiết tại vị trí đó
          </div>
        </div>

        {/* Chi tiết phép tính tại ô được chọn */}
        <div className="glass-card" style={{ padding: 16, background: "rgba(245,158,11,0.03)" }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--warning)", marginBottom: 12 }}>
            Phép tính tại W[{j}, {i}]  &nbsp;
            <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 400 }}>
              (pixel index j={j}, class i={i})
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Trái: dữ liệu đầu vào */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 12, fontFamily: "monospace" }}>
              <div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>x[{j}] (giá trị pixel input):</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "var(--accent-light)" }}>{xj.toFixed(6)}</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>∂L/∂z[{i}] = p[{i}] − y[{i}]:</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: i === label ? "var(--danger)" : "var(--warning)" }}>
                  {dLdz_i >= 0 ? "+" : ""}{dLdz_i.toFixed(6)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>W[{j}, {i}] cũ:</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>{wOld.toFixed(6)}</div>
              </div>
            </div>

            {/* Phải: công thức LaTeX với số */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>Bước 1 — Tính gradient:</div>
                <div style={{ padding: "8px 10px", background: "rgba(99,102,241,0.08)", borderRadius: 4 }}>
                  <TeX math={`\\frac{\\partial L}{\\partial W_{${j},${i}}} = x_{${j}} \\cdot \\frac{\\partial L}{\\partial z_{${i}}} = ${xj.toFixed(4)} \\times ${dLdz_i.toFixed(4)} = ${gradWji.toFixed(6)}`} block />
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>Bước 2 — Cập nhật:</div>
                <div style={{ padding: "8px 10px", background: "rgba(16,185,129,0.08)", borderRadius: 4 }}>
                  <TeX math={`W_{${j},${i}}^{new} = ${wOld.toFixed(6)} - ${lr} \\times ${gradWji.toFixed(6)} = ${wNew.toFixed(6)}`} block />
                </div>
              </div>
              <div style={{ fontSize: 11, color: "var(--text-secondary)", fontStyle: "italic" }}>
                ΔW = {deltaW >= 0 ? "+" : ""}{deltaW.toFixed(8)}
                {Math.abs(deltaW) < 1e-7 && " (≈ 0 vì x[j] = 0 — pixel đen)"}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ════════════════════════════════════════════════════════════
  // TAB 4 — Reshape Gradient ∂L/∂x → 13×13×8
  // ════════════════════════════════════════════════════════════
  const renderReshapeGrad = () => {
    if (!gradPool) return <div>Thiếu dữ liệu grad_pool.</div>;
    const flatGrad = gradPool.flat(2);
    const row = selectedSoftmaxBackCell?.row ?? 0;
    const col = selectedSoftmaxBackCell?.col ?? 0;
    const flatIndex = row * 104 + col * 8 + backwardFilter;
    const currentFilterSlice = gradPool.map((r: number[][]) => r.map((c: number[]) => c[backwardFilter]));

    const render1DSnippet = (idx: number, arr: number[]) => {
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
            if (el.isEllipsis) return <div key={el.key} style={{ padding: "0 10px", color: "var(--text-muted)", fontSize: 11, fontStyle: "italic" }}>{el.text}</div>;
            return (
              <div key={el.key} style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: 70 }}>
                <div style={{ fontSize: 10, color: el.isSelected ? "var(--warning)" : "var(--text-muted)", marginBottom: 4, fontFamily: "monospace" }}>[{el.index}]</div>
                <div style={{
                  padding: "8px 4px",
                  background: el.isSelected ? "rgba(245,158,11,0.15)" : "rgba(255,255,255,0.03)",
                  border: `1px solid ${el.isSelected ? "var(--warning)" : "var(--border)"}`,
                  borderRadius: 6, fontFamily: "monospace", fontSize: 11,
                  color: el.isSelected ? "var(--warning)" : "var(--text-primary)",
                  textAlign: "center", width: "100%"
                }}>
                  {el.val?.toFixed(5)}
                </div>
              </div>
            );
          })}
        </div>
      );
    };

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span>Tính gradient theo input để truyền tiếp về Pool:</span>
          <TeX math={"\\dfrac{\\partial L}{\\partial \\mathbf{x}} = W \\cdot \\dfrac{\\partial L}{\\partial \\mathbf{z}}"} />
          <FormulaPopover
            title="Vì sao W (không phải W^T)?"
            generalFormula={"\\frac{\\partial L}{\\partial x_p} = \\sum_c \\frac{\\partial L}{\\partial z_c} \\cdot \\frac{\\partial z_c}{\\partial x_p} = \\sum_c \\frac{\\partial L}{\\partial z_c} \\cdot W_{p,c}"}
            note={`Shape của W là (1352, 10). Khi nhân với vector ∂L/∂z (10,) → vector kết quả có shape (1352,) — khớp với shape của x.`}
          />
        </div>

        <div className="glass-card" style={{ padding: 16, display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>1. Vector 1D Gradient (1352 phần tử)</div>
          <div style={{ width: "100%", maxWidth: 800 }}>{render1DSnippet(flatIndex, flatGrad)}</div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", color: "var(--warning)", margin: "12px 0" }}>
            <ArrowLeftOutlined style={{ fontSize: 18, transform: "rotate(90deg)" }} />
            <span style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, marginTop: 4 }}>Reshape (fold) về 13×13×8</span>
          </div>

          <div style={{ fontWeight: 600, marginBottom: 8 }}>2. Ma trận Gradient 2D theo bộ lọc (13×13)</div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
            {Array.from({ length: 8 }).map((_, f) => (
              <div key={f} onClick={() => setBackwardFilter(f)} style={{
                display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
                cursor: "pointer", padding: 6, borderRadius: 8,
                background: backwardFilter === f ? "rgba(245,158,11,0.1)" : "transparent",
                border: backwardFilter === f ? "1px solid var(--warning)" : "1px solid transparent",
                transition: "all 0.2s"
              }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: backwardFilter === f ? "var(--warning)" : "var(--text-muted)" }}>Filter {f}</div>
                <MatrixGrid
                  data={gradPool.map((r: number[][]) => r.map((c: number[]) => c[f]))}
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
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 8 }}>
                Ánh xạ 1D ↔ 3D
                {" "}
                <FormulaPopover
                  title="Công thức ánh xạ index"
                  generalFormula={"\\text{idx}_{1D} = r \\cdot 104 + c \\cdot 8 + f"}
                  substitutedFormula={`= ${row} \\times 104 + ${col} \\times 8 + ${backwardFilter}`}
                  result={`= ${flatIndex}`}
                  note="Hệ số 104 = 13 (chiều rộng) × 8 (số filter). Đây là cách NumPy lưu mảng 3D dưới dạng dải bộ nhớ tuyến tính (row-major)."
                />
              </div>
              <div style={{ fontFamily: "monospace", fontSize: 12, color: "var(--text-secondary)", lineHeight: 2 }}>
                Toạ độ 3D: [r={row}, c={col}, f={backwardFilter}]<br />
                Giá trị: <strong style={{ color: "var(--warning)" }}>{gradPool[row]?.[col]?.[backwardFilter]?.toFixed(6)}</strong><br />
                Index 1D: <strong style={{ color: "var(--warning)" }}>{flatIndex}</strong>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const items = [
    { key: "error_vector", label: "1. Vector lỗi (∂L/∂z)", children: renderErrorVector() },
    { key: "biases_update", label: "2. Cập nhật Bias (10 số)", children: renderBiasUpdate() },
    { key: "weights_update", label: `3. Cập nhật Weights (${sliceRows}×10 slice của ${totalRows}×10)`, children: renderWeightsUpdate() },
    { key: "reshape_grad", label: "4. Reshape gradient → Pool", children: renderReshapeGrad() },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Pipeline gợi nhớ 3 thành phần thay đổi */}
      <div className="glass-card" style={{ padding: 12, background: "rgba(99,102,241,0.04)", border: "1px solid var(--border)" }}>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>
          <strong style={{ color: "var(--accent)" }}>Tại bước Softmax Backward này, có 2 ma trận thay đổi:</strong>
          <br />
          <ArrowRightOutlined style={{ color: "var(--success)" }} /> <strong>Bias</strong> (vector 10) — tab 2
          {" • "}
          <ArrowRightOutlined style={{ color: "var(--success)" }} /> <strong>Weights</strong> (ma trận {totalRows}×10) — tab 3
          <br />
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            (Ma trận thứ 3 — Conv Filters — sẽ thay đổi ở bước Conv Backward sau)
          </span>
        </div>
      </div>

      <Tabs activeKey={softmaxBackTab} onChange={setSoftmaxBackTab} items={items} type="card" />
    </div>
  );
}
