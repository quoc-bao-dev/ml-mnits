"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, Tabs, Steps, Badge, Spin } from "antd";
import {
  RocketOutlined,
  ExperimentOutlined,
  ClearOutlined,
  ReloadOutlined,
  EditOutlined,
  FileImageOutlined,
  WarningOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import ConvForward from "../../components/steps/ConvForward";
import MaxPoolForward from "../../components/steps/MaxPoolForward";
import SoftmaxForward from "../../components/steps/SoftmaxForward";

export default function PredictPage() {
  const [predictMode, setPredictMode] = useState<"draw" | "sample">("draw");
  const [predicting, setPredicting] = useState<boolean>(false);
  const [predictStep, setPredictStep] = useState<number>(0);
  const [viewStep, setViewStep] = useState<number>(3);
  const [predictResult, setPredictResult] = useState<any>(null);
  
  // State lưu ảnh mẫu ngẫu nhiên từ MNIST được chọn
  const [selectedSample, setSelectedSample] = useState<{
    inputImage: string;
    trueLabel: number | null;
  } | null>(null);
  const [loadingSample, setLoadingSample] = useState<boolean>(false);

  // State phục vụ tinh chỉnh sửa lỗi trực tuyến (Active Learning)
  const [finetuning, setFinetuning] = useState<boolean>(false);
  const [finetuneCorrectLabel, setFinetuneCorrectLabel] = useState<number | null>(null);
  const [finetuneResult, setFinetuneResult] = useState<any>(null);
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDrawingRef = useRef<boolean>(false);

  const submitFinetune = () => {
    if (finetuneCorrectLabel === null) return;
    setFinetuning(true);
    setFinetuneResult(null);

    fetch("/api/predict/finetune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ correctLabel: finetuneCorrectLabel }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          console.error("Finetune error:", data.error);
        } else {
          setFinetuneResult(data);
        }
        setFinetuning(false);
      })
      .catch((err) => {
        console.error("Finetune API error:", err);
        setFinetuning(false);
      });
  };


  const selectRandomSample = () => {
    setLoadingSample(true);
    setPredictResult(null);
    setPredictStep(0);
    
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "select_sample" }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          console.error("Error selecting sample:", data.error);
        } else {
          setSelectedSample({
            inputImage: data.inputImage,
            trueLabel: data.trueLabel,
          });
        }
        setLoadingSample(false);
      })
      .catch((err) => {
        console.error("Error selecting sample:", err);
        setLoadingSample(false);
      });
  };


  useEffect(() => {
    if (predictMode === "draw") {
      initCanvas();
    }
  }, [predictMode]);

  const initCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  };

  const clearCanvas = () => {
    initCanvas();
    setPredictResult(null);
    setPredictStep(0);
    setFinetuneCorrectLabel(null);
    setFinetuneResult(null);
  };

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    isDrawingRef.current = true;
    const { x, y } = getCanvasCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { x, y } = getCanvasCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const handleMouseUpOrLeave = () => {
    isDrawingRef.current = false;
  };

  const runPrediction = async () => {
    setPredicting(true);
    setPredictResult(null);
    setFinetuneCorrectLabel(null);
    setFinetuneResult(null);
    
    setPredictStep(1); 
    await new Promise((r) => setTimeout(r, 1200));
    
    setPredictStep(2); 
    await new Promise((r) => setTimeout(r, 1500));
    
    setPredictStep(3); 
    await new Promise((r) => setTimeout(r, 1200));

    let bodyData: any = { useSample: true };

    if (predictMode === "draw") {
      const canvas = canvasRef.current;
      if (canvas) {
        const base64Image = canvas.toDataURL("image/png");
        bodyData = { useSample: false, image: base64Image };
      }
    } else {
      bodyData = { useSample: true, trueLabel: selectedSample?.trueLabel };
    }

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyData),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          setPredictStep(0);
        } else {
          setPredictResult(data);
          setPredictStep(4);
          setViewStep(3);
        }
        setPredicting(false);
      })
      .catch((err) => {
        console.error("Prediction API error:", err);
        setPredictStep(0);
        setPredicting(false);
      });
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24, alignItems: "start" }}>
      {/* Khối công cụ chọn/vẽ */}
      <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: 20, position: "sticky", top: 24, zIndex: 10, height: "fit-content" }}>
        <div className="glass-card-header">
          <RocketOutlined style={{ color: "var(--accent)" }} />
          <span>Vùng nhập Ảnh đầu vào</span>
        </div>

        <Tabs
          activeKey={predictMode}
          onChange={(key) => {
            if (predicting) return;
            const mode = key as "draw" | "sample";
            setPredictMode(mode);
            setPredictResult(null);
            setPredictStep(0);
            // Tự động load ảnh ngẫu nhiên nếu chưa có
            if (mode === "sample" && !selectedSample) {
              selectRandomSample();
            }
          }}
          centered
          items={[
            {
              key: "draw",
              label: "Vẽ tay số (Canvas)",
              icon: <EditOutlined />,
              disabled: predicting,
            },
            {
              key: "sample",
              label: "Lấy ảnh ngẫu nhiên từ MNIST",
              icon: <FileImageOutlined />,
              disabled: predicting,
            },
          ]}
        />

        {predictMode === "draw" ? (
          <div className="canvas-container">
            <canvas
              ref={canvasRef}
              width={280}
              height={280}
              className="canvas-element"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUpOrLeave}
              onMouseLeave={handleMouseUpOrLeave}
            />
            <div style={{ marginTop: 16, display: "flex", gap: 12, width: "100%" }}>
              <Button
                icon={<ClearOutlined />}
                onClick={clearCanvas}
                style={{ width: "50%" }}
                disabled={predicting}
              >
                Xóa Canvas
              </Button>
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={runPrediction}
                loading={predicting}
                style={{ width: "50%" }}
              >
                Dự đoán số
              </Button>
            </div>
          </div>
        ) : (
          <div style={{ textAlign: "center", padding: "30px 20px" }}>
            <div
              style={{
                width: 140,
                height: 140,
                background: "#000",
                margin: "0 auto 20px",
                border: "2px dashed var(--accent)",
                borderRadius: 12,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
                position: "relative",
              }}
            >
              {loadingSample ? (
                <Spin tip="Đang tải mẫu..." size="small" />
              ) : selectedSample ? (
                <img
                  src={selectedSample.inputImage}
                  alt="Selected MNIST Sample"
                  style={{
                    width: "100%",
                    height: "100%",
                    imageRendering: "pixelated",
                    objectFit: "contain",
                  }}
                />
              ) : (
                <span style={{ fontSize: 32 }}>📊</span>
              )}
            </div>

            {selectedSample && !loadingSample && (
              <div style={{ marginBottom: 16 }}>
                <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Nhãn thật sự của ảnh mẫu: </span>
                <strong style={{ color: "var(--success)", fontSize: 15 }}>{selectedSample.trueLabel}</strong>
              </div>
            )}

            <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 20 }}>
              Hệ thống sẽ lấy ngẫu nhiên 1 ảnh chữ số từ tập kiểm thử 10.000 ảnh MNIST trên server để truyền qua mạng CNN.
            </p>
            
            <div style={{ display: "flex", gap: 12, width: "100%" }}>
              <Button
                icon={<ReloadOutlined />}
                onClick={selectRandomSample}
                disabled={predicting}
                loading={loadingSample}
                style={{ width: "40%" }}
              >
                Lấy ảnh khác
              </Button>
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={runPrediction}
                loading={predicting}
                disabled={!selectedSample || loadingSample}
                style={{ width: "60%" }}
              >
                Dự đoán số này
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Khối hiển thị phân tích step-by-step & kết quả */}
      <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div className="glass-card-header">
          <ExperimentOutlined style={{ color: "var(--success)" }} />
          <span>Luồng Phân tích Toán học của CNN</span>
        </div>

        <Steps
          direction="vertical"
          size="small"
          current={predictStep === 4 ? viewStep : predictStep - 1}
          onChange={(current) => {
            if (predictStep === 4 && predictResult?.visualData) {
              setViewStep(current);
            }
          }}
          items={[
            {
              title: "Bước 1: Phép toán tích chập Conv3x3",
              description: predictStep >= 1 ? "Trượt 8 bộ lọc 3x3 để trích xuất đặc trưng" : "",
            },
            {
              title: "Bước 2: Lớp gộp cực đại (MaxPooling)",
              description: predictStep >= 2 ? "Gộp 2x2 để giảm kích thước 50%" : "",
            },
            {
              title: "Bước 3: Lớp Linear & Softmax",
              description: predictStep >= 3 ? "Phẳng hóa và tính phân phối xác suất 10 lớp" : "",
            },
            {
              title: "Bước 4: Kết xuất Kết quả",
              description: predictStep >= 4 ? "Hiển thị chữ số có độ tin cậy cao nhất" : "",
            },
          ]}
        />

        {predictResult && predictStep === 4 && (
          <div style={{ marginTop: 12, borderTop: "1px solid var(--border-light)", paddingTop: 16 }}>
            {viewStep === 0 && predictResult.visualData && (
              <ConvForward data={predictResult.visualData} step={predictResult.visualData.steps[0]} />
            )}
            
            {viewStep === 1 && predictResult.visualData && (
              <MaxPoolForward data={predictResult.visualData} step={predictResult.visualData.steps[1]} />
            )}
            
            {viewStep === 2 && predictResult.visualData && (
              <SoftmaxForward data={predictResult.visualData} step={predictResult.visualData.steps[2]} />
            )}
            
            {viewStep === 3 && (
              <>
                <div style={{ display: "flex", alignItems: "center", gap: 24, marginBottom: 20 }}>
                  <div
                    style={{
                      width: 72,
                      height: 72,
                      borderRadius: 16,
                      background: "linear-gradient(135deg, var(--accent), #a855f7)",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      boxShadow: "0 0 15px var(--accent-glow)",
                    }}
                  >
                    <span style={{ fontSize: 11, color: "rgba(255,255,255,0.7)", fontWeight: 500 }}>DỰ ĐOÁN</span>
                    <strong style={{ fontSize: 32, color: "#fff", lineHeight: 1, marginTop: -2 }}>
                      {predictResult.prediction}
                    </strong>
                  </div>
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)" }}>
                      Chữ số dự đoán: {predictResult.prediction}
                    </div>
                    <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                      Độ tin cậy: {(predictResult.confidence * 100).toFixed(1)}%
                    </div>
                    {predictResult.trueLabel !== null && (
                      <div style={{ fontSize: 13, marginTop: 4 }}>
                        Nhãn thực tế: <strong>{predictResult.trueLabel}</strong>{" "}
                        {predictResult.prediction === predictResult.trueLabel ? (
                          <Badge status="success" text="Chính xác" />
                        ) : (
                          <Badge status="error" text="Sai lệch" />
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Phân phối xác suất các số */}
                <div style={{ marginBottom: 24 }}>
                  <span style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 12, color: "var(--text-primary)" }}>
                    Phân phối xác suất đầu ra (Softmax):
                  </span>
                  {predictResult.probabilities &&
                    predictResult.probabilities.map((prob: number, num: number) => {
                      const isActive = num === predictResult.prediction;
                      return (
                        <div key={num} className="probability-bar-container">
                          <span className="probability-label">{num}</span>
                          <div className="probability-bar-bg">
                            <div
                              className={`probability-bar-fill ${isActive ? "active" : ""}`}
                              style={{ width: `${(prob * 100).toFixed(1)}%` }}
                            />
                          </div>
                          <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      );
                    })}
                </div>

                {/* Khu vực sửa lỗi cho Canvas (chỉ áp dụng ở mode draw) */}
                {predictMode === "draw" && (
                  <div style={{
                    marginTop: 20,
                    padding: 16,
                    background: "rgba(244, 63, 94, 0.05)",
                    border: "1px dashed rgba(244, 63, 94, 0.3)",
                    borderRadius: 12,
                    marginBottom: 24,
                    textAlign: "left"
                  }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, fontWeight: 600, color: "var(--error)", marginBottom: 8 }}>
                      <WarningOutlined />
                      <span>Máy nhận dạng sai nét vẽ của bạn?</span>
                    </span>
                    <p style={{ color: "var(--text-secondary)", fontSize: 12, marginBottom: 12, lineHeight: 1.4 }}>
                      Mạng nơ-ron có thể tự cập nhật trọng số thông qua lan truyền ngược (backpropagation) để ghi nhớ nét vẽ này. Hãy chọn số bạn thực sự vẽ:
                    </p>
                    
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12, justifyContent: "center" }}>
                      {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((num) => (
                        <Button
                          key={num}
                          size="small"
                          type={finetuneCorrectLabel === num ? "primary" : "default"}
                          danger={finetuneCorrectLabel === num}
                          onClick={() => setFinetuneCorrectLabel(num)}
                          disabled={finetuning}
                          style={{ width: 32, padding: 0 }}
                        >
                          {num}
                        </Button>
                      ))}
                    </div>

                    <Button
                      type="primary"
                      danger
                      icon={<SyncOutlined spin={finetuning} />}
                      disabled={finetuneCorrectLabel === null || finetuning}
                      loading={finetuning}
                      onClick={submitFinetune}
                      style={{ width: "100%", height: 36 }}
                    >
                      Cập nhật Trọng số & Sửa lỗi
                    </Button>

                    {finetuneResult && (
                      <div style={{
                        marginTop: 12,
                        padding: 10,
                        background: "rgba(16, 185, 129, 0.08)",
                        border: "1px solid rgba(16, 185, 129, 0.2)",
                        borderRadius: 8,
                        fontSize: 12,
                        lineHeight: 1.5
                      }}>
                        <div style={{ color: "var(--success)", fontWeight: 600, marginBottom: 4 }}>
                          ✅ Tinh chỉnh mô hình thành công!
                        </div>
                        <div>
                          • Số dự đoán cũ: <strong style={{ color: "var(--error)" }}>{finetuneResult.before.prediction}</strong> (Loss: {finetuneResult.before.loss.toFixed(4)})
                        </div>
                        <div>
                          • Số dự đoán mới: <strong style={{ color: "var(--success)" }}>{finetuneResult.after.prediction}</strong> (Loss: {finetuneResult.after.loss.toFixed(4)})
                        </div>
                        <p style={{ margin: "6px 0 0 0", fontStyle: "italic", fontSize: 11, color: "var(--text-muted)", lineHeight: 1.4 }}>
                          *Trọng số mạng nơ-ron tại các bộ lọc Conv & Softmax đã được cập nhật cục bộ qua Gradient Descent. Hãy bấm nút <strong>"Dự đoán số"</strong> ở trên một lần nữa để cập nhật biểu đồ!
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Ảnh vẽ biểu đồ phân phối xác suất từ python */}
                {predictResult.chartImage && (
                  <div>
                    <span style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 8, color: "var(--text-primary)" }}>
                      Biểu đồ phân tích từ Python Engine:
                    </span>
                    <img
                      src={predictResult.chartImage}
                      alt="Prediction Probability Chart"
                      style={{ width: "100%", borderRadius: 10, border: "1px solid var(--border)" }}
                    />
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
