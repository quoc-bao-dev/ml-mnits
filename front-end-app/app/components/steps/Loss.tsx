interface LossProps {
  step: any;
}

export default function Loss({ step }: LossProps) {
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
          Đo lường độ sai lệch giữa phân phối xác suất dự đoán <code style={{ color: "var(--accent-light)" }}>p</code> và nhãn thực tế <code style={{ color: "var(--success)" }}>y</code> (ở dạng one-hot vector).<br />
          Công thức tổng quát: <strong>L = - Σ y<sub>i</sub> · ln(p<sub>i</sub>)</strong><br />
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
            <strong>Tính chất của hàm -ln(x):</strong><br />
            • Nếu mô hình chắc chắn đúng (p<sub>{label}</sub> → 1), Loss → 0.<br />
            • Nếu dự đoán sai (p<sub>{label}</sub> → 0), Loss tăng rất lớn.
          </div>
        </div>
      </div>
    </div>
  );
}
