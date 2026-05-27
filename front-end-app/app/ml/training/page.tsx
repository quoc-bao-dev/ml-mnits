"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, Spin, Alert, Tabs } from "antd";
import {
  ThunderboltOutlined,
  ExperimentOutlined,
  ReloadOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import TrainingVisualizer from "@/app/components/TrainingVisualizer";

export default function TrainingPage() {
  const [activeTab, setActiveTab] = useState<string>("train");
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [charts, setCharts] = useState<Record<string, string>>({});
  const [loadingCharts, setLoadingCharts] = useState<boolean>(false);
  const [numTrain, setNumTrain] = useState<number>(1000);
  const [numTest, setNumTest] = useState<number>(1000);
  const terminalBodyRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Kiểm tra trạng thái training ban đầu & tải charts hiện có
    checkTrainingStatus();
    loadCharts();

    // Thiết lập polling trạng thái training định kỳ
    const interval = setInterval(() => {
      checkTrainingStatus();
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Polling xem logs & status
  const checkTrainingStatus = () => {
    fetch("/api/train")
      .then((res) => res.json())
      .then((data) => {
        setIsTraining(data.isTraining);
        if (data.numTrain) setNumTrain(data.numTrain);
        if (data.numTest) setNumTest(data.numTest);
        if (data.isTraining) {
          setTrainingLogs(data.logs || []);
        } else if (trainingLogs.length > 0 && data.logs && data.logs.length > trainingLogs.length) {
          setTrainingLogs(data.logs);
          loadCharts();
        }
      })
      .catch((err) => console.error("Error checking training status:", err));
  };

  const loadCharts = () => {
    setLoadingCharts(true);
    fetch("/api/charts")
      .then((res) => res.json())
      .then((data) => {
        setCharts(data);
        setLoadingCharts(false);
      })
      .catch((err) => {
        console.error("Error loading charts:", err);
        setLoadingCharts(false);
      });
  };

  // Cuộn log terminal xuống cuối mà không giật màn hình
  useEffect(() => {
    const terminalBody = terminalBodyRef.current;
    if (terminalBody) {
      terminalBody.scrollTop = terminalBody.scrollHeight;
    }
  }, [trainingLogs]);

  const startTraining = () => {
    if (isTraining) return;
    setIsTraining(true);
    setTrainingLogs(["🚀 Gửi yêu cầu bắt đầu huấn luyện lên server..."]);

    fetch("/api/train", { method: "POST" })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          setTrainingLogs((prev) => [...prev, `❌ Lỗi: ${data.error}`]);
          setIsTraining(false);
        } else {
          checkTrainingStatus();
        }
      })
      .catch((err) => {
        setTrainingLogs((prev) => [...prev, `❌ Kết nối thất bại: ${err.message}`]);
        setIsTraining(false);
      });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        centered
        items={[
          {
            key: "train",
            label: "Huấn luyện Model",
            icon: <ThunderboltOutlined />,
          },
          {
            key: "visual",
            label: "Trực quan hóa CNN",
            icon: <EyeOutlined />,
          },
        ]}
      />

      {activeTab === "visual" ? (
        <TrainingVisualizer />
      ) : (
        <>
          {/* Khối cấu hình & Trạng thái chạy */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 24 }}>
            <div className="glass-card">
              <div className="glass-card-header">
                <ThunderboltOutlined style={{ color: "var(--accent)" }} />
                <span>Tham số Huấn luyện CNN</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Kích thước Epochs:</span>
                  <strong style={{ color: "var(--text-primary)" }}>3 Epochs</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Tốc độ học (Learning Rate):</span>
                  <strong style={{ color: "var(--text-primary)" }}>0.005</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Kích thước Bộ lọc:</span>
                  <strong style={{ color: "var(--text-primary)" }}>3 × 3 (8 Filters)</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Tập dữ liệu trích xuất:</span>
                  <strong style={{ color: "var(--text-primary)" }}>
                    {numTrain.toLocaleString()} Train + {numTest.toLocaleString()} Test
                  </strong>
                </div>


                <div style={{ marginTop: 10, display: "flex", gap: 12 }}>
                  <Button
                    type="primary"
                    icon={<ThunderboltOutlined />}
                    loading={isTraining}
                    onClick={startTraining}
                    style={{ flexGrow: 1, height: 40 }}
                  >
                    {isTraining ? "Đang huấn luyện..." : "Bắt đầu Huấn luyện"}
                  </Button>
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={loadCharts}
                    style={{ height: 40 }}
                    title="Tải lại charts"
                  />
                </div>
              </div>
            </div>

            {/* Console Log Terminal */}
            <div className="terminal-window">
              <div className="terminal-header">
                <div className="terminal-dots">
                  <div className="terminal-dot red" />
                  <div className="terminal-dot yellow" />
                  <div className="terminal-dot green" />
                </div>
                <div className="terminal-title">NUMPY CNN ENGINE - LIVE TERMINAL</div>
                <div />
              </div>
              <div className="terminal-body" ref={terminalBodyRef}>
                {trainingLogs.length === 0 ? (
                  <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
                    Chưa có tiến trình huấn luyện nào được chạy hoặc ghi log.
                  </div>
                ) : (
                  trainingLogs.map((log, index) => (
                    <div key={index} className="terminal-line">
                      {log}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Output kết quả trực quan */}
          <div className="glass-card">
            <div className="glass-card-header">
              <ExperimentOutlined style={{ color: "var(--success)" }} />
              <span>Biểu đồ & Phân tích Kết quả Huấn luyện</span>
            </div>

            {loadingCharts ? (
              <div style={{ textAlign: "center", padding: "60px 0" }}>
                <Spin tip="Đang tải các biểu đồ kết quả..." />
              </div>
            ) : Object.keys(charts).length === 0 ? (
              <Alert
                message="Chưa có kết quả"
                description="Hãy bấm nút 'Bắt đầu Huấn luyện' để huấn luyện mô hình. Sau khi huấn luyện thành công, các biểu đồ toán học (Loss, Accuracy) và minh hoạ dự đoán MNIST sẽ hiển thị tại đây."
                type="info"
                showIcon
              />
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
                <div>
                  <h3 style={{ marginBottom: 16, color: "var(--text-primary)" }}>1. Biểu đồ Lịch sử Huấn luyện (Loss & Accuracy)</h3>
                  {charts.training_history && (
                    <img
                      src={charts.training_history}
                      alt="Training History"
                      style={{ width: "100%", borderRadius: 12, border: "1px solid var(--border)" }}
                    />
                  )}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 24 }}>
                  <div>
                    <h3 style={{ marginBottom: 12, color: "var(--text-primary)" }}>2. Ảnh mẫu MNIST và chuẩn hóa pixel</h3>
                    {charts.pixel_matrix && (
                      <img
                        src={charts.pixel_matrix}
                        alt="Pixel Matrix"
                        style={{ width: "100%", borderRadius: 12, border: "1px solid var(--border)" }}
                      />
                    )}
                  </div>
                  <div>
                    <h3 style={{ marginBottom: 12, color: "var(--text-primary)" }}>3. Dự đoán minh họa trên 20 ảnh tập test</h3>
                    {charts.predictions && (
                      <img
                        src={charts.predictions}
                        alt="Predictions Grid"
                        style={{ width: "100%", borderRadius: 12, border: "1px solid var(--border)" }}
                      />
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
