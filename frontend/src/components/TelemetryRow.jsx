import React from 'react';

export default function TelemetryRow({ telemetry }) {
  return (
    <section className="telemetry-row">
      {/* 1. Mastery Card */}
      <div className="telemetry-card" style={{ "--card-accent": "#10b981" }}>
        <div className="telemetry-header">
          <span>NĂNG LỰC LỚP (MASTERY)</span>
          <span>{telemetry.meanMastery > 0.6 ? "Tốt" : "Cần lưu ý"}</span>
        </div>
        <div className="telemetry-val">
          {(telemetry.meanMastery * 100).toFixed(1)}%
        </div>
        <div className="telemetry-sub">
          Nhóm yếu (Q25): <strong style={{ color: "var(--color-danger)" }}>{(telemetry.q25Mastery * 100).toFixed(1)}%</strong> • Q75: {(telemetry.q75Mastery * 100).toFixed(1)}%
        </div>
      </div>

      {/* 2. Attention Card */}
      <div className="telemetry-card" style={{ "--card-accent": "#38bdf8" }}>
        <div className="telemetry-header">
          <span>ĐỘ TẬP TRUNG (ATTENTION)</span>
          <span>{telemetry.meanAttention > 0.7 ? "Tốt" : "Giảm sút"}</span>
        </div>
        <div className="telemetry-val">
          {(telemetry.meanAttention * 100).toFixed(1)}%
        </div>
        <div className="telemetry-sub">
          Phản ứng nhạy bén • Kích hoạt học tập
        </div>
      </div>

      {/* 3. Fatigue Card */}
      <div className="telemetry-card" style={{ "--card-accent": "#f59e0b" }}>
        <div className="telemetry-header">
          <span>MỆT MỎI NHẬN THỨC</span>
          <span>{telemetry.meanFatigue > 0.35 ? "Cảnh báo cao" : "Bình thường"}</span>
        </div>
        <div className="telemetry-val">
          {(telemetry.meanFatigue * 100).toFixed(1)}%
        </div>
        <div className="telemetry-sub">
          Tải nhận thức tích lũy: <strong>{telemetry.workload.toFixed(1)}</strong> / 15.0
        </div>
      </div>

      {/* 4. Progress & Environment Card */}
      <div className="telemetry-card" style={{ "--card-accent": "#a855f7" }}>
        <div className="telemetry-header">
          <span>TIẾN ĐỘ & MÔI TRƯỜNG</span>
          <span style={{ color: "var(--primary-glow)", fontWeight: "600" }}>
            Đã học: {Math.round(telemetry.timeElapsedMinutes || 0)}' / Còn: {Math.round(telemetry.timeRemainingMinutes)}'
          </span>
        </div>
        <div className="telemetry-val">
          {Math.round(telemetry.co2)} <small style={{ fontSize: "0.9rem", color: "var(--text-dim)" }}>ppm CO₂</small>
        </div>
        <div className="telemetry-sub">
          Nhiệt độ: {telemetry.temp.toFixed(1)}°C • Độ ồn: {Math.round(telemetry.noise)} dB
        </div>
      </div>
    </section>
  );
}
