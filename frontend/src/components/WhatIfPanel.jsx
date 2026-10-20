import React from 'react';

const actionLabels = {
  groupwork: "Thảo luận nhóm",
  quiz: "Kiểm tra ngắn",
  lecture: "Giảng bài tiếp",
  discussion: "Hỏi đáp mở",
  break: "Nghỉ giải lao"
};

const actionColors = {
  groupwork: "linear-gradient(90deg, #8b5cf6, #d946ef)",
  quiz: "linear-gradient(90deg, #38bdf8, #2563eb)",
  lecture: "linear-gradient(90deg, #64748b, #94a3b8)",
  discussion: "linear-gradient(90deg, #06b6d4, #10b981)",
  break: "linear-gradient(90deg, #10b981, #34d399)"
};

export default function WhatIfPanel({ whatIfSims }) {
  return (
    <div className="glass-box">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <h3 style={{ fontSize: "0.95rem", fontWeight: 700, fontFamily: "'Outfit', sans-serif" }}>
          Dự Báo What-If (Monte Carlo M2)
        </h3>
        <span style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>10 rollouts</span>
      </div>

      <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.4 }}>
        Mô phỏng đa kịch bản nếu giáo viên chọn các hành động sư phạm khác nhau:
      </p>

      <div className="whatif-chart">
        {whatIfSims.map(sim => {
          const act = sim.candidate_action;
          const deltaMVal = (sim.mean_delta_mastery || 0) * 100;
          const deltaMStr = (deltaMVal >= 0 ? "+" : "") + deltaMVal.toFixed(1) + "%";
          const rawRew = sim.median_reward !== undefined ? sim.median_reward : 0.0;
          const rewStr = rawRew.toFixed(2);
          const isNegative = rawRew < 0;

          // Chuẩn hóa chiều dài thanh tiến trình
          let widthPct = 20;
          if (rawRew >= 0) {
            widthPct = Math.min(100, Math.max(15, (rawRew / 0.6) * 100));
          } else {
            widthPct = Math.min(100, Math.max(20, (Math.abs(rawRew) / 0.4) * 100));
          }

          const barColor = isNegative
            ? "linear-gradient(90deg, #ef4444, #f97316)"
            : (actionColors[act] || "#38bdf8");

          return (
            <div key={act} className="whatif-bar-item">
              <div className="whatif-bar-header">
                <span>{actionLabels[act] || act}</span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", color: "var(--text-muted)", fontSize: "0.75rem" }}>
                  Reward: <strong style={{ color: isNegative ? "#f87171" : "#fff" }}>{rewStr}</strong>
                  {sim.p10_reward !== undefined && (
                    <span style={{ fontSize: "0.68rem", color: "var(--text-dim)", marginLeft: "4px" }}>
                      [{sim.p10_reward.toFixed(2)}~{sim.p90_reward.toFixed(2)}]
                    </span>
                  )}
                  {" | "}ΔMastery: <span style={{ color: deltaMVal >= 0 ? "var(--color-success)" : "#f87171" }}>{deltaMStr}</span>
                </span>
              </div>
              <div className="whatif-bar-track" style={{ background: isNegative ? "rgba(239,68,68,0.15)" : "rgba(255,255,255,0.05)" }}>
                <div className="whatif-bar-fill" style={{ width: `${widthPct}%`, background: barColor }}></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
