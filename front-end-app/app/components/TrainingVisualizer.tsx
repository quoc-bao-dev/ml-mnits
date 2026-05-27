"use client";
import React, { useState, useEffect } from "react";
import { Button, Steps, Slider, Select, Spin, Alert, Badge, Tabs } from "antd";
import {
  PlayCircleOutlined,
  EyeOutlined,
  ArrowRightOutlined,
  ArrowLeftOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
  AimOutlined,
  CheckCircleFilled,
} from "@ant-design/icons";
import MatrixGrid from "./MatrixGrid";

interface VisualData {
  input_image: number[][];
  label: number;
  learning_rate: number;
  total_steps: number;
  steps: any[];
}

export default function TrainingVisualizer() {
  const [data, setData] = useState<VisualData | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  // Conv visualizer state
  const [convRow, setConvRow] = useState(0);
  const [convCol, setConvCol] = useState(0);
  const [convFilter, setConvFilter] = useState(0);
  // MaxPool visualizer state
  const [poolRow, setPoolRow] = useState(0);
  const [poolCol, setPoolCol] = useState(0);
  const [poolFilter, setPoolFilter] = useState(0);
  // Softmax visualizer state
  const [softmaxTab, setSoftmaxTab] = useState("flatten");
  const [flattenFilter, setFlattenFilter] = useState(0);
  const [flattenSelected, setFlattenSelected] = useState<{row: number, col: number, val: number} | null>({row: 0, col: 0, val: 0});


  useEffect(() => { fetchData(); }, []);

  const fetchData = () => {
    setLoading(true);
    fetch("/api/train/visual")
      .then(r => r.json())
      .then(d => { if (d.exists && d.data) setData(d.data); setLoading(false); })
      .catch(() => setLoading(false));
  };

  const generate = () => {
    setGenerating(true);
    fetch("/api/train/visual", { method: "POST" })
      .then(r => r.json())
      .then(() => { fetchData(); setGenerating(false); setCurrentStep(0); })
      .catch(() => setGenerating(false));
  };

  if (loading) return <div style={{ textAlign: "center", padding: 60 }}><Spin tip="Đang tải dữ liệu..." /></div>;

  if (!data) {
    return (
      <div style={{ textAlign: "center", padding: 40 }}>
        <Alert message="Chưa có dữ liệu trực quan" description="Bấm nút bên dưới để chạy 1 vòng forward+backward trên 1 ảnh mẫu MNIST và visualize từng bước toán học." type="info" showIcon style={{ marginBottom: 20 }} />
        <Button type="primary" icon={<PlayCircleOutlined />} loading={generating} onClick={generate} size="large">
          Tạo dữ liệu trực quan
        </Button>
      </div>
    );
  }

  const step = data.steps[currentStep];

  // Compute conv result for current position
  const computeConvAt = (row: number, col: number, filterIdx: number) => {
    if (!data || step?.type !== "conv_forward") return null;
    const img = data.input_image;
    const filters = step.filters;
    const filter = filters[filterIdx];
    const region: number[][] = [];
    const products: number[][] = [];
    let sum = 0;
    for (let m = 0; m < 3; m++) {
      const rRow: number[] = [];
      const pRow: number[] = [];
      for (let n = 0; n < 3; n++) {
        const pixelVal = img[row + m]?.[col + n] ?? 0;
        const filterVal = filter[m][n];
        const prod = pixelVal * filterVal;
        rRow.push(pixelVal);
        pRow.push(prod);
        sum += prod;
      }
      region.push(rRow);
      products.push(pRow);
    }
    return { region, filter, products, sum };
  };

  const renderConvForward = () => {
    const comp = computeConvAt(convRow, convCol, convFilter);
    const outputSlice = step.output.map((row: number[][]) => row.map((cell: number[]) => cell[convFilter]));

    const handleConvNext = () => {
      if (convCol < 25) {
        setConvCol(convCol + 1);
      } else if (convRow < 25) {
        setConvCol(0);
        setConvRow(convRow + 1);
      } else {
        setConvCol(0);
        setConvRow(0);
      }
    };

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Controls */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
          <Button icon={<ArrowRightOutlined />} onClick={handleConvNext} type="primary" ghost>Trượt Tiếp</Button>
          <div style={{ flex: 1, minWidth: 200 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Vị trí kernel (hàng): {convRow}</span>
            <Slider min={0} max={25} value={convRow} onChange={setConvRow} />
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Vị trí kernel (cột): {convCol}</span>
            <Slider min={0} max={25} value={convCol} onChange={setConvCol} />
          </div>
          <Select value={convFilter} onChange={setConvFilter} style={{ width: 140 }}
            options={Array.from({ length: 8 }, (_, i) => ({ value: i, label: `Filter ${i}` }))} />
        </div>
        {/* Visualization */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
          <div>
            <MatrixGrid data={data.input_image} cellSize={14} colorScheme="grayscale" label="Ảnh đầu vào 28×28" highlightRegion={{ row: convRow, col: convCol, size: 3 }} />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: "center" }}>
            {comp && (
              <>
                <MatrixGrid data={comp.region} cellSize={36} colorScheme="grayscale" showValues label={`Vùng ảnh [${convRow}:${convRow+3}, ${convCol}:${convCol+3}]`} borderColor="#f59e0b" />
                <span style={{ fontSize: 18, color: "var(--text-muted)" }}>×</span>
                <MatrixGrid data={comp.filter} cellSize={36} colorScheme="heatmap" showValues label={`Filter ${convFilter} (3×3)`} borderColor="#6366f1" />
                <span style={{ fontSize: 18, color: "var(--text-muted)" }}>=</span>
                <MatrixGrid data={comp.products} cellSize={36} colorScheme="gradient" showValues label="Tích từng phần tử" borderColor="#10b981" />
              </>
            )}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {comp && (
              <div className="glass-card" style={{ padding: 16, height: "auto" }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "var(--accent)", marginBottom: 8 }}>Kết quả tại [{convRow}, {convCol}]</div>
                <div style={{ fontFamily: "var(--font-jetbrains-mono), monospace", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.8 }}>
                  Y[{convRow},{convCol},{convFilter}] = Σ (X · W)<br />
                  = {comp.products.flat().map(v => v.toFixed(4)).join(" + ")}<br />
                  = <strong style={{ color: "var(--success)", fontSize: 14 }}>{comp.sum.toFixed(6)}</strong>
                </div>
              </div>
            )}
            <div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>Feature Map (Filter {convFilter})</div>
              <MatrixGrid data={outputSlice} cellSize={12} colorScheme="heatmap" label={`Output 26×26 – Filter ${convFilter}`} />
            </div>
          </div>
        </div>
        {/* All 8 filters */}
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 8 }}>Tất cả 8 bộ lọc (Filters)</div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {step.filters.map((f: number[][], idx: number) => (
              <div key={idx} onClick={() => setConvFilter(idx)} style={{ cursor: "pointer", outline: idx === convFilter ? "2px solid var(--accent)" : "none", borderRadius: 6, padding: 2 }}>
                <MatrixGrid data={f} cellSize={28} colorScheme="heatmap" showValues label={`F${idx}`} />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderMaxPoolForward = () => {
    // Show input (conv output for selected filter) and output
    const prevStep = data.steps[0]; // conv_forward
    const inputSlice = prevStep.output.map((row: number[][]) => row.map((cell: number[]) => cell[poolFilter]));
    const outputSlice = step.output.map((row: number[][]) => row.map((cell: number[]) => cell[poolFilter]));
    // Highlight 2x2 region
    const highlightRow = poolRow * 2;
    const highlightCol = poolCol * 2;
    // Get the 2x2 region values
    const regionVals = [
      [inputSlice[highlightRow]?.[highlightCol] ?? 0, inputSlice[highlightRow]?.[highlightCol + 1] ?? 0],
      [inputSlice[highlightRow + 1]?.[highlightCol] ?? 0, inputSlice[highlightRow + 1]?.[highlightCol + 1] ?? 0],
    ];
    const maxVal = Math.max(...regionVals.flat());

    const handlePoolNext = () => {
      if (poolCol < 12) {
        setPoolCol(poolCol + 1);
      } else if (poolRow < 12) {
        setPoolCol(0);
        setPoolRow(poolRow + 1);
      } else {
        setPoolCol(0);
        setPoolRow(0);
      }
    };

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
          <Button icon={<ArrowRightOutlined />} onClick={handlePoolNext} type="primary" ghost>Trượt Tiếp</Button>
          <div style={{ flex: 1, minWidth: 150 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Pool hàng: {poolRow}</span>
            <Slider min={0} max={12} value={poolRow} onChange={setPoolRow} />
          </div>
          <div style={{ flex: 1, minWidth: 150 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Pool cột: {poolCol}</span>
            <Slider min={0} max={12} value={poolCol} onChange={setPoolCol} />
          </div>
          <Select value={poolFilter} onChange={setPoolFilter} style={{ width: 140 }}
            options={Array.from({ length: 8 }, (_, i) => ({ value: i, label: `Filter ${i}` }))} />
        </div>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
          <div>
            <MatrixGrid data={inputSlice} cellSize={14} colorScheme="heatmap" label={`Input 26×26 (F${poolFilter})`} highlightRegion={{ row: highlightRow, col: highlightCol, size: 2 }} />
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
            <MatrixGrid data={regionVals} cellSize={48} colorScheme="heatmap" showValues label={`Vùng 2×2 tại [${highlightRow},${highlightCol}]`} borderColor="#f59e0b" />
            <div className="glass-card" style={{ padding: 12, height: "auto", textAlign: "center" }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>max({regionVals.flat().map(v => v.toFixed(3)).join(", ")})</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--success)" }}> = {maxVal.toFixed(6)}</div>
            </div>
            <span style={{ fontSize: 18, color: "var(--text-muted)" }}>↓</span>
            <MatrixGrid data={outputSlice} cellSize={18} colorScheme="heatmap" label={`Output 13×13 (F${poolFilter})`} highlightRegion={{ row: poolRow, col: poolCol, size: 1 }} />
          </div>
        </div>
      </div>
    );
  };

  const renderSoftmaxForward = () => {
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
                        onCellClick={(row, col, val) => setFlattenSelected({row, col, val})}
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
                          Tọa độ 3D: [Hàng <strong>{flattenSelected.row}</strong>, Cột <strong>{flattenSelected.col}</strong>, Filter <strong>{flattenFilter}</strong>]<br/>
                          Giá trị: <strong style={{ color: "var(--warning)" }}>{flattenSelected.val.toFixed(6)}</strong><br/>
                          <br/>
                          Công thức ánh xạ sang 1D (C-contiguous):<br/>
                          Index = (Hàng × 104) + (Cột × 8) + Filter<br/>
                          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ({flattenSelected.row} × 104) + ({flattenSelected.col} × 8) + {flattenFilter}<br/>
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
              Vector 1D (<strong>1352</strong> phần tử) được nhân ma trận với bộ trọng số <strong>W (1352×10)</strong> và cộng bias <strong>b (10)</strong> để tạo ra 10 Logits (z).<br/>
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
              Chuyển đổi Logits thành phân phối xác suất (tổng = 100%). Sử dụng Trick trừ đi max(z) để tránh tràn số (ổn định số học).<br/>
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
  };

  const renderLoss = () => {
    const pCorrect = step.correct_prob;
    const label = step.correct_label;
    const loss = step.loss_value;
    const probs = step.probabilities || [];
    
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Explanation of Cross-Entropy */}
        <div className="glass-card" style={{ padding: 20 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", marginBottom: 12 }}>
            Hàm mất mát Cross-Entropy (Categorical Cross-Entropy)
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 16 }}>
            Đo lường độ sai lệch giữa phân phối xác suất dự đoán <code style={{ color: "var(--accent-light)" }}>p</code> và nhãn thực tế <code style={{ color: "var(--success)" }}>y</code> (ở dạng one-hot vector).<br/>
            Công thức tổng quát: <strong>L = - Σ y<sub>i</sub> · ln(p<sub>i</sub>)</strong><br/>
            Vì y<sub>i</sub> = 1 tại nhãn đúng và y<sub>i</sub> = 0 tại các nhãn sai, công thức rút gọn thành:
          </div>
          <div style={{ padding: "12px 20px", background: "rgba(255, 255, 255, 0.03)", border: "1px solid var(--border)", borderRadius: 8, display: "inline-block" }}>
            <span style={{ fontSize: 18, fontWeight: 600, color: "var(--warning)", fontFamily: "var(--font-jetbrains-mono), monospace" }}>
              L = -ln(p<sub>{label}</sub>)
            </span>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
          {/* Comparison table */}
          <div className="glass-card" style={{ padding: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 16 }}>
              So sánh Phân phối Xác suất
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", fontSize: 12, fontWeight: 600, color: "var(--text-muted)", borderBottom: "1px solid var(--border)", paddingBottom: 8 }}>
                <div style={{ width: 60 }}>Class</div>
                <div style={{ flex: 1 }}>Dự đoán (p)</div>
                <div style={{ width: 80, textAlign: "right" }}>Thực tế (y)</div>
              </div>
              {probs.map((p: number, i: number) => {
                const isTruth = i === label;
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", fontSize: 13, padding: "4px 0", background: isTruth ? "rgba(16, 185, 129, 0.1)" : "transparent", borderRadius: 4 }}>
                    <div style={{ width: 60, fontWeight: isTruth ? 600 : 400, color: isTruth ? "var(--success)" : "var(--text-primary)", paddingLeft: 4 }}>
                      {i} {isTruth && "✅"}
                    </div>
                    <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ flex: 1, height: 6, background: "#1e293b", borderRadius: 3 }}>
                        <div style={{ width: `${p * 100}%`, height: "100%", background: isTruth ? "var(--success)" : "var(--accent)", borderRadius: 3 }} />
                      </div>
                      <span style={{ fontFamily: "monospace", width: 50, color: isTruth ? "var(--success)" : "var(--text-secondary)" }}>{(p * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ width: 80, textAlign: "right", fontFamily: "monospace", fontWeight: isTruth ? 600 : 400, color: isTruth ? "var(--success)" : "var(--text-muted)", paddingRight: 4 }}>
                      {isTruth ? "1.0" : "0.0"}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Calculation result */}
          <div className="glass-card" style={{ padding: 20, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", textAlign: "center" }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 24 }}>
              Kết quả tính Loss
            </div>
            
            <div style={{ fontSize: 16, color: "var(--text-secondary)", marginBottom: 12, fontFamily: "var(--font-jetbrains-mono), monospace" }}>
              L = -ln({pCorrect.toFixed(6)})
            </div>
            
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 24 }}>
              <span style={{ fontSize: 24, fontWeight: 600 }}>L =</span>
              <span style={{ fontSize: 48, fontWeight: 700, color: "var(--warning)", lineHeight: 1 }}>{loss.toFixed(6)}</span>
            </div>
            
            <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6, background: "rgba(0,0,0,0.2)", padding: 12, borderRadius: 8, textAlign: "left", width: "100%" }}>
              <strong>Tính chất của hàm -ln(x):</strong><br/>
              • Nếu mô hình chắc chắn đúng (p<sub>{label}</sub> → 1), Loss → 0.<br/>
              • Nếu dự đoán sai (p<sub>{label}</sub> → 0), Loss tăng rất lớn.
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSoftmaxBackward = () => (
    <div className="glass-card" style={{ padding: 20, height: "auto" }}>
      <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8 }}>
        <p><strong>1.</strong> Tính gradient ban đầu: <code>∂L/∂p[{data.label}] = -1/p[{data.label}]</code></p>
        <p><strong>2.</strong> Nhân với Jacobian của Softmax: <code>∂L/∂z = ∂L/∂p · J(Softmax)</code></p>
        <p><strong>3.</strong> Cập nhật trọng số: <code>W ← W - {data.learning_rate} · (∂L/∂z ⊗ xᵀ)</code></p>
        <p><strong>4.</strong> Cập nhật bias: <code>b ← b - {data.learning_rate} · ∂L/∂z</code></p>
        <p><strong>5.</strong> Truyền gradient về MaxPool: <code>∂L/∂x = Wᵀ · ∂L/∂z</code> → reshape {step.output_gradient_shape?.join("×")}</p>
      </div>
    </div>
  );

  const renderMaxPoolBackward = () => {
    const heatmap = step.gradient_heatmap_f0;
    return (
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div className="glass-card" style={{ padding: 16, height: "auto", flex: 1, minWidth: 280 }}>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8 }}>
            <p>Gradient chỉ chảy về vị trí có giá trị max trong mỗi vùng 2×2.</p>
            <p>Các vị trí không phải max nhận gradient = 0 (<strong>winner-take-all</strong>).</p>
          </div>
        </div>
        {heatmap && (
          <div>
            <MatrixGrid data={heatmap} cellSize={12} colorScheme="heatmap" label="Gradient heatmap (Filter 0) — 26×26" />
          </div>
        )}
      </div>
    );
  };

  const renderConvBackward = () => (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div className="glass-card" style={{ padding: 16, height: "auto" }}>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8 }}>
          <p><strong>Gradient bộ lọc:</strong> <code>∂L/∂W_f[m,n] = Σ X[i+m, j+n] · ∂L/∂Y[i, j, f]</code></p>
          <p><strong>Cập nhật:</strong> <code>W_f ← W_f - {data.learning_rate} · ∂L/∂W_f</code></p>
        </div>
      </div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 12 }}>So sánh bộ lọc Trước vs Sau cập nhật</div>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          {step.filters_before?.map((_: number[][], idx: number) => (
            <div key={idx} style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "center" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--accent)" }}>Filter {idx}</div>
              <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                <MatrixGrid data={step.filters_before[idx]} cellSize={28} colorScheme="heatmap" showValues label="Trước" />
                <span style={{ fontSize: 14, color: "var(--text-muted)" }}>→</span>
                <MatrixGrid data={step.filters_after[idx]} cellSize={28} colorScheme="heatmap" showValues label="Sau" />
              </div>
              <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                Δ max: {Math.max(...step.filter_deltas[idx].flat().map(Math.abs)).toFixed(6)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderStep = () => {
    switch (step.type) {
      case "conv_forward": return renderConvForward();
      case "maxpool_forward": return renderMaxPoolForward();
      case "softmax_forward": return renderSoftmaxForward();
      case "loss": return renderLoss();
      case "softmax_backward": return renderSoftmaxBackward();
      case "maxpool_backward": return renderMaxPoolBackward();
      case "conv_backward": return renderConvBackward();
      default: return <div>Unknown step type: {step.type}</div>;
    }
  };

  const stepItems = data.steps.map((s, i) => ({
    title: s.title,
    description: s.phase === "forward" ? "Forward" : "Backward",
    status: (i === currentStep ? "process" : i < currentStep ? "finish" : "wait") as "process" | "finish" | "wait",
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 56, height: 56, background: "#000", borderRadius: 8, border: "2px solid var(--accent)", overflow: "hidden" }}>
            <MatrixGrid data={data.input_image} cellSize={2} colorScheme="grayscale" />
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>Ảnh mẫu: Chữ số <strong style={{ color: "var(--accent)" }}>{data.label}</strong></div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>28×28 pixel | lr={data.learning_rate}</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Button icon={<ArrowLeftOutlined />} disabled={currentStep === 0} onClick={() => setCurrentStep(c => c - 1)}>Trước</Button>
          <Button icon={<ArrowRightOutlined />} disabled={currentStep === data.total_steps - 1} onClick={() => setCurrentStep(c => c + 1)} type="primary">Tiếp</Button>
          <Button icon={<ReloadOutlined />} onClick={generate} loading={generating}>Tạo mới</Button>
        </div>
      </div>
      {/* Main layout */}
      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 20, alignItems: "start" }}>
        {/* Timeline sidebar */}
        <div className="glass-card" style={{ padding: 16, position: "sticky", top: 24, zIndex: 10, height: "fit-content" }}>
          <Steps direction="vertical" size="small" current={currentStep} onChange={setCurrentStep} items={stepItems} />
        </div>
        {/* Visualization panel */}
        <div className="glass-card" style={{ height: "auto", minHeight: 400 }}>
          <div className="glass-card-header">
            <EyeOutlined style={{ color: step.phase === "forward" ? "var(--accent)" : "var(--warning)" }} />
            <span>{step.title}</span>
            <Badge count={step.phase === "forward" ? "Forward" : "Backward"} style={{ backgroundColor: step.phase === "forward" ? "var(--accent)" : "var(--warning)", marginLeft: 8 }} />
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>{step.description}</div>
          {step.formula && (
            <div style={{ padding: "8px 12px", background: "rgba(99,102,241,0.08)", borderRadius: 8, fontFamily: "var(--font-jetbrains-mono), monospace", fontSize: 13, color: "var(--accent-light)", marginBottom: 16, border: "1px solid var(--border)" }}>
              {step.formula}
            </div>
          )}
          {renderStep()}
        </div>
      </div>
    </div>
  );
}
