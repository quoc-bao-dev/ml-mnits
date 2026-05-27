import { ArrowRightOutlined } from "@ant-design/icons";
import { Button, Select, Slider } from "antd";
import { useState } from "react";
import MatrixGrid from "../common/MatrixGrid";

interface MaxPoolForwardProps {
  data: any;
  step: any;
}

export default function MaxPoolForward({ data, step }: MaxPoolForwardProps) {
  const [poolRow, setPoolRow] = useState(0);
  const [poolCol, setPoolCol] = useState(0);
  const [poolFilter, setPoolFilter] = useState(0);

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
}
