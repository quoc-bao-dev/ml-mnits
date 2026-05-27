"use client";
import {
  ArrowLeftOutlined,
  ArrowRightOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  ReloadOutlined
} from "@ant-design/icons";
import { Alert, Badge, Button, Spin, Steps } from "antd";
import { useEffect, useState } from "react";
import MatrixGrid from "./common/MatrixGrid";

// Import step components
import ConvBackward from "./steps/ConvBackward";
import ConvForward from "./steps/ConvForward";
import Loss from "./steps/Loss";
import MaxPoolBackward from "./steps/MaxPoolBackward";
import MaxPoolForward from "./steps/MaxPoolForward";
import SoftmaxBackward from "./steps/SoftmaxBackward";
import SoftmaxForward from "./steps/SoftmaxForward";

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

  const renderStep = () => {
    switch (step.type) {
      case "conv_forward": return <ConvForward data={data} step={step} />;
      case "maxpool_forward": return <MaxPoolForward data={data} step={step} />;
      case "softmax_forward": return <SoftmaxForward data={data} step={step} />;
      case "loss": return <Loss step={step} />;
      case "softmax_backward": return <SoftmaxBackward data={data} step={step} />;
      case "maxpool_backward": return <MaxPoolBackward data={data} step={step} />;
      case "conv_backward": return <ConvBackward data={data} step={step} />;
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
