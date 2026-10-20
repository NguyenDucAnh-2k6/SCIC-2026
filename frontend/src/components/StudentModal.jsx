import React from 'react';

export default function StudentModal({ student, onClose }) {
  if (!student) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-card">
        <div className="modal-header">
          <div className="modal-title">Sinh viên {student.id}</div>
          <button className="modal-close-btn" onClick={onClose}>&times;</button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "20px" }}>
          <div className="meta-item">
            <div className="label">Mức độ hiểu bài (Mastery)</div>
            <div className="val" style={{ fontSize: "1.2rem", color: "var(--color-success)" }}>
              {(student.mastery * 100).toFixed(1)}%
            </div>
          </div>
          <div className="meta-item">
            <div className="label">Độ tập trung (Attention)</div>
            <div className="val" style={{ fontSize: "1.2rem", color: "var(--primary-glow)" }}>
              {(student.attention * 100).toFixed(1)}%
            </div>
          </div>
          <div className="meta-item">
            <div className="label">Mức độ mệt mỏi (Fatigue)</div>
            <div className="val" style={{ fontSize: "1.2rem", color: "var(--color-warning)" }}>
              {(student.fatigue * 100).toFixed(1)}%
            </div>
          </div>
          <div className="meta-item">
            <div className="label">Vị trí chỗ ngồi</div>
            <div className="val">Hàng {student.row} - Cột {student.col}</div>
          </div>
        </div>

        <h4 style={{ fontSize: "0.85rem", marginBottom: "10px" }}>Nhật Ký Tương Tác Gần Nhất (EdNet KT1 Log)</h4>
        <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: "8px", padding: "10px", fontFamily: "'JetBrains Mono', monospace", fontSize: "0.72rem", lineHeight: 1.6, color: "#94a3b8", maxHeight: "120px", overflowY: "auto" }}>
          {(student.recentLogs || [
            { q: "q10428 (Algebra)", ans: "Correct (b)", elapsed: "18s" },
            { q: "q10429 (Functions)", ans: student.mastery > 0.4 ? "Correct (a)" : "Incorrect (c)", elapsed: "35s" }
          ]).map((log, idx) => (
            <div key={idx}>
              &gt; [{log.q}] • Kết quả: <span style={{ color: log.ans.includes('Correct') ? '#10b981' : '#ef4444' }}>{log.ans}</span> ({log.elapsed})
            </div>
          ))}
        </div>

        <div style={{ marginTop: "20px", display: "flex", justifyContent: "flex-end" }}>
          <button className="btn btn-primary" onClick={onClose}>Đóng</button>
        </div>
      </div>
    </div>
  );
}
