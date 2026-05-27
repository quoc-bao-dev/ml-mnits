import { AimOutlined, ArrowRightOutlined, CheckCircleFilled } from "@ant-design/icons";
import { Badge, Select, Tabs } from "antd";
import { useState } from "react";
import MatrixGrid from "../common/MatrixGrid";

interface SoftmaxForwardProps {
  data: any;
  step: any;
}

export default function SoftmaxForward({ data, step }: SoftmaxForwardProps) {
  const [softmaxTab, setSoftmaxTab] = useState("flatten");
  const [flattenFilter, setFlattenFilter] = useState(0);
  const [flattenSelected, setFlattenSelected] = useState<{ row: number, col: number, val: number } | null>({ row: 0, col: 0, val: 0 });

  const probs: number[] = step.probabilities;
  const logits: number[] = step.logits;
  const maxLogit = Math.max(...logits);

  // Compute intermediate math for Softmax tab
  const exps = logits.map((z: number) => Math.exp(z - maxLogit));
  const sumExps = exps.reduce((a: number, b: number) => a + b, 0);

  // Retrieve pool output for Flatten tab from previous step
  const poolOutput = data.steps[1]?.output; // 13x13x8

  const render1DArraySnippet = (selectedIndex: number, flatArray: number[]) => {
    const startIdx = Math.max(0, selectedIndex - 4);
    const endIdx = Math.min(flatArray.length - 1, selectedIndex + 4);

    const elements = [];
    if (startIdx > 0) elements.push({ isEllipsis: true, key: "start-ell", text: `... ${startIdx} phần tử` });

    for (let i = startIdx; i <= endIdx; i++) {
      elements.push({ index: i, val: flatArray[i], isSelected: i === selectedIndex, key: `elem-${i}` });
    }

    if (endIdx < flatArray.length - 1) elements.push({ isEllipsis: true, key: "end-ell", text: `${flatArray.length - 1 - endIdx} phần tử ...` });

    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, overflowX: "auto", padding: "8px 0" }}>
        {elements.map((el: any) => {
          if (el.isEllipsis) {
            return (
              <div key={el.key} style={{ padding: "0 12px", color: "var(--text-muted)", fontSize: 12, fontStyle: "italic" }}>
                {el.text}
              </div>
            );
          }
          return (
            <div key={el.key} style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              minWidth: 70,
            }}>
              <div style={{ fontSize: 11, color: el.isSelected ? "var(--success)" : "var(--text-muted)", marginBottom: 6, fontFamily: "var(--font-jetbrains-mono), monospace", fontWeight: el.isSelected ? 700 : 500 }}>
                [{el.index}]
              </div>
              <div style={{
                padding: "10px 6px",
                background: el.isSelected ? "rgba(16, 185, 129, 0.15)" : "rgba(255, 255, 255, 0.03)",
                border: `1px solid ${el.isSelected ? "var(--success)" : "var(--border)"}`,
                borderRadius: 6,
                fontFamily: "var(--font-jetbrains-mono), monospace",
                fontSize: 12,
                color: el.isSelected ? "var(--success)" : "var(--text-primary)",
                textAlign: "center",
                width: "100%",
                boxShadow: el.isSelected ? "0 0 12px rgba(16, 185, 129, 0.3)" : "none",
                transition: "all 0.3s",
                fontWeight: el.isSelected ? 700 : 500
              }}>
                {el.val?.toFixed(4)}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Tab Items
  const items = [
    {
      key: "flatten",
      label: "1. Flatten (Duỗi thẳng)",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Toàn bộ dữ liệu từ 8 bộ lọc của MaxPool (mỗi bộ lọc kích thước <strong>13×13</strong>) sẽ được nối và duỗi thẳng thành một vector 1D (<strong>1352</strong> phần tử).
          </div>

          {poolOutput ? (() => {
            const flatArray = poolOutput.flat(2);
            const selectedIndex = flattenSelected ? (flattenSelected.row * 104 + flattenSelected.col * 8 + flattenFilter) : 0;
            return (
              <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                {/* Sơ đồ tổng quan 8 ma trận -> 1D Array */}
                <div className="glass-card" style={{ padding: 24, display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ fontWeight: 600, marginBottom: 16, color: "var(--text-primary)" }}>
                    1. Từ 8 Đặc trưng 2D (13×13)
                  </div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
                    {Array.from({ length: 8 }).map((_, f) => (
                      <div
                        key={f}
                        onClick={() => {
                          setFlattenFilter(f);
                          if (flattenSelected) setFlattenSelected({ ...flattenSelected, val: poolOutput[flattenSelected.row][flattenSelected.col][f] });
                        }}
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          gap: 6,
                          cursor: "pointer",
                          padding: 8,
                          borderRadius: 8,
                          background: flattenFilter === f ? "rgba(99, 102, 241, 0.1)" : "transparent",
                          border: flattenFilter === f ? "1px solid var(--accent)" : "1px solid transparent",
                          transition: "all 0.2s"
                        }}
                      >
                        <div style={{ fontSize: 11, fontWeight: 600, color: flattenFilter === f ? "var(--accent)" : "var(--text-muted)" }}>F{f}</div>
                        <MatrixGrid
                          data={poolOutput.map((r: number[][]) => r.map(c => c[f]))}
                          cellSize={4}
                          colorScheme="heatmap"
                          selectedCell={flattenSelected && flattenFilter === f ? { row: flattenSelected.row, col: flattenSelected.col } : null}
                        />
                      </div>
                    ))}
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", color: "var(--accent)", margin: "20px 0" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ height: 1, width: 60, background: "var(--accent)", opacity: 0.5 }} />
                      <span style={{ fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Duỗi thẳng (Flatten)</span>
                      <div style={{ height: 1, width: 60, background: "var(--accent)", opacity: 0.5 }} />
                    </div>
                    <ArrowRightOutlined style={{ fontSize: 24, transform: "rotate(90deg)", marginTop: 8 }} />
                  </div>

                  <div style={{ fontWeight: 600, marginBottom: 16, color: "var(--text-primary)" }}>
                    2. Thành Vector 1D (1352 phần tử)
                  </div>

                  {/* Visual 1D Array */}
                  <div style={{ width: "100%", maxWidth: 800 }}>
                    {render1DArraySnippet(selectedIndex, flatArray)}
                  </div>
                </div>

                {/* Detailed inspector */}
                <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
                        Phóng to Filter {flattenFilter}:
                      </span>
                      <Select value={flattenFilter} onChange={(f) => {
                        setFlattenFilter(f);
                        if (flattenSelected) setFlattenSelected({ ...flattenSelected, val: poolOutput[flattenSelected.row][flattenSelected.col][f] });
                      }} style={{ width: 100 }} size="small" options={Array.from({ length: 8 }, (_, i) => ({ value: i, label: `F${i}` }))} />
                    </div>
                    <MatrixGrid
                      data={poolOutput.map((r: number[][]) => r.map(c => c[flattenFilter]))}
                      cellSize={22}
                      colorScheme="heatmap"
                      selectedCell={flattenSelected ? { row: flattenSelected.row, col: flattenSelected.col } : null}
                      onCellClick={(row, col, val) => setFlattenSelected({ row, col, val })}
                    />
                    <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center" }}>
                      Click vào một ô để kiểm tra ánh xạ
                    </div>
                  </div>

                  {flattenSelected && (
                    <div className="glass-card" style={{ flex: 1, minWidth: 280, padding: 20 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 12 }}>
                        Ánh xạ phần tử được chọn
                      </div>
                      <div style={{ fontFamily: "monospace", fontSize: 13, color: "var(--text-secondary)", lineHeight: 2.2 }}>
                        Tọa độ 3D: [Hàng <strong>{flattenSelected.row}</strong>, Cột <strong>{flattenSelected.col}</strong>, Filter <strong>{flattenFilter}</strong>]<br />
                        Giá trị: <strong style={{ color: "var(--warning)" }}>{flattenSelected.val.toFixed(6)}</strong><br />
                        <br />
                        Công thức ánh xạ sang 1D (C-contiguous):<br />
                        Index = (Hàng × 104) + (Cột × 8) + Filter<br />
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ({flattenSelected.row} × 104) + ({flattenSelected.col} × 8) + {flattenFilter}<br />
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <strong style={{ color: "var(--success)", fontSize: 16 }}>{selectedIndex}</strong>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })() : (
            <div>Không tìm thấy dữ liệu MaxPool trước đó.</div>
          )}
        </div>
      )
    },
    {
      key: "linear",
      label: "2. Linear (z = Wx + b)",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Vector 1D (<strong>1352</strong> phần tử) được nhân ma trận với bộ trọng số <strong>W (1352×10)</strong> và cộng bias <strong>b (10)</strong> để tạo ra 10 Logits (z).<br />
            Công thức: <code style={{ color: "var(--accent-light)" }}>z_i = Σ (W_[j, i] × x_j) + b_i</code>
          </div>

          <div className="glass-card" style={{ padding: 16 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-muted)", marginBottom: 12 }}>Logits (z) — 10 giá trị Raw Scores</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
              {logits.map((v, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(255,255,255,0.03)", padding: "8px 12px", borderRadius: 6, border: "1px solid var(--border)" }}>
                  <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>z<sub>{i}</sub></span>
                  <span style={{ fontFamily: "monospace", color: v === maxLogit ? "var(--warning)" : "var(--text-secondary)" }}>{v.toFixed(4)}</span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
              <em>* Điểm Logit cao nhất: <strong>{maxLogit.toFixed(4)}</strong> (tại z<sub>{logits.indexOf(maxLogit)}</sub>)</em>
            </div>
          </div>
        </div>
      )
    },
    {
      key: "softmax",
      label: "3. Softmax (Tính Xác Suất)",
      children: (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            Chuyển đổi Logits thành phân phối xác suất (tổng = 100%). Sử dụng Trick trừ đi max(z) để tránh tràn số (ổn định số học).<br />
            Công thức: <code style={{ color: "var(--accent-light)" }}>p_i = exp(z_i - z_max) / Σ exp(z_k - z_max)</code>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)", color: "var(--text-muted)" }}>
                  <th style={{ padding: 8 }}>Class</th>
                  <th style={{ padding: 8 }}>Logit (z)</th>
                  <th style={{ padding: 8 }}>Shift (z - max)</th>
                  <th style={{ padding: 8 }}>Tử số e^(shift)</th>
                  <th style={{ padding: 8 }}>Xác suất (p)</th>
                </tr>
              </thead>
              <tbody>
                {probs.map((p, i) => {
                  const isPred = i === step.prediction;
                  const isTruth = i === step.correct_label;
                  let bg = "transparent";
                  if (isPred && isTruth) bg = "rgba(16, 185, 129, 0.15)";
                  else if (isPred) bg = "rgba(239, 68, 68, 0.15)";
                  else if (isTruth) bg = "rgba(245, 158, 11, 0.1)";

                  return (
                    <tr key={i} style={{ borderBottom: "1px solid var(--border)", background: bg }}>
                      <td style={{ padding: 8, fontWeight: 600 }}>
                        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                          <span>{i}</span>
                          {isPred && <AimOutlined style={{ color: "#ef4444" }} title="Dự đoán" />}
                          {isTruth && <CheckCircleFilled style={{ color: "#10b981" }} title="Nhãn đúng" />}
                        </div>
                      </td>
                      <td style={{ padding: 8, fontFamily: "monospace" }}>{logits[i].toFixed(4)}</td>
                      <td style={{ padding: 8, fontFamily: "monospace" }}>{(logits[i] - maxLogit).toFixed(4)}</td>
                      <td style={{ padding: 8, fontFamily: "monospace", color: "var(--accent-light)" }}>{exps[i].toFixed(6)}</td>
                      <td style={{ padding: 8 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <div style={{ flex: 1, height: 8, background: "#1e293b", borderRadius: 4, minWidth: 60 }}>
                            <div style={{ width: `${p * 100}%`, height: "100%", background: isPred ? "var(--success)" : "var(--accent)", borderRadius: 4 }} />
                          </div>
                          <span style={{ fontFamily: "monospace", width: 50 }}>{(p * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="glass-card" style={{ padding: 12, display: "flex", gap: 20, flexWrap: "wrap" }}>
            <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              <strong>Max Logit (z_max):</strong> {maxLogit.toFixed(4)}
            </div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              <strong>Tổng tử số (S):</strong> {sumExps.toFixed(6)}
            </div>
          </div>
        </div>
      )
    }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <Tabs activeKey={softmaxTab} onChange={setSoftmaxTab} items={items} type="card" />
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 8 }}>
        <Badge status={step.is_correct ? "success" : "error"} text={<span style={{ fontSize: 14 }}>Dự đoán: <strong>{step.prediction}</strong> | Nhãn đúng: <strong>{step.correct_label}</strong> | Độ tin cậy: <strong>{(step.confidence * 100).toFixed(1)}%</strong></span>} />
      </div>
    </div>
  );
}
