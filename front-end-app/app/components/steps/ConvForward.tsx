import { ArrowRightOutlined } from "@ant-design/icons";
import { Button, Select, Slider } from "antd";
import { useState } from "react";
import MatrixGrid from "../common/MatrixGrid";

interface ConvForwardProps {
  data: any;
  step: any;
}

export default function ConvForward({ data, step }: ConvForwardProps) {
  const [convRow, setConvRow] = useState(0);
  const [convCol, setConvCol] = useState(0);
  const [convFilter, setConvFilter] = useState(0);

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
          <MatrixGrid 
            data={data.input_image} 
            cellSize={14} 
            colorScheme="grayscale" 
            label="Ảnh đầu vào 28×28" 
            highlightRegion={{ row: convRow, col: convCol, size: 3 }} 
            onCellClick={(r, c) => {
              setConvRow(Math.min(25, Math.max(0, r - 1)));
              setConvCol(Math.min(25, Math.max(0, c - 1)));
            }}
            onRegionSelect={(region) => {
              setConvRow(Math.min(25, Math.max(0, region.startRow)));
              setConvCol(Math.min(25, Math.max(0, region.startCol)));
            }}
          />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: "center" }}>
          {comp && (
            <>
              <MatrixGrid data={comp.region} cellSize={36} colorScheme="grayscale" showValues label={`Vùng ảnh [${convRow}:${convRow + 3}, ${convCol}:${convCol + 3}]`} borderColor="#f59e0b" />
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
            <MatrixGrid data={outputSlice} cellSize={12} colorScheme="heatmap" label={`Output 26×26 – Filter ${convFilter}`} selectedCell={{ row: convRow, col: convCol }} />
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
}
