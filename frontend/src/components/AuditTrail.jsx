import React from 'react';

export default function AuditTrail({ auditTrail }) {
  return (
    <div className="glass-box">
      <h3 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: "10px", fontFamily: "'Outfit', sans-serif" }}>
        Nhật Ký Quyết Định Sư Phạm
      </h3>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "180px", overflowY: "auto", fontSize: "0.75rem" }}>
        {auditTrail.length === 0 ? (
          <div style={{ color: "var(--text-dim)", textAlign: "center", padding: "10px" }}>Chưa có quyết định nào được ghi lại.</div>
        ) : (
          auditTrail.map((item, idx) => {
            const isAcc = item.status === "ACCEPTED";
            const isMod = item.status === "MODIFIED";
            const statusColor = isAcc ? "var(--color-success)" : (isMod ? "var(--primary-glow)" : "var(--color-danger)");

            return (
              <div key={idx} style={{ background: "rgba(255,255,255,0.03)", padding: "8px 10px", borderRadius: "8px", borderLeft: `3px solid ${statusColor}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "2px" }}>
                  <strong style={{ color: statusColor }}>{item.status}: {item.action.toUpperCase()}</strong>
                  <span style={{ color: "var(--text-dim)" }}>{item.time}</span>
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>{item.reason}</div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
