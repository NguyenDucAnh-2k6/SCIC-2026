import React from 'react';

export default function Header({
  isAutoPlay,
  setIsAutoPlay,
  policyMode,
  setPolicyMode,
  sessionDuration,
  onDurationChange,
  onInitClassroom,
  onStepClassroom,
  onRunWhatIf,
  addToast,
  telemetryAction
}) {
  return (
    <header className="app-header">
      <div className="brand-wrapper">
        <div className="brand-logo-icon">CT</div>
        <div className="brand-text">
          <h1>ClassTwin Digital Twin</h1>
          <p>Offline RL & Inverse RL Pedagogical Decision Support • SCIC 2026</p>
        </div>
      </div>

      <div className="header-actions" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        {/* Status Badge */}
        <div className="status-badge">
          <span className="status-dot"></span>
          <span>Digital Twin Online</span>
        </div>

        {/* Session Duration Selector: 45m | 90m | 180m */}
        <div
          className="duration-selector"
          style={{
            display: 'flex',
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.65)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '10px',
            padding: '3px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.25)'
          }}
          title="Chọn thời lượng buổi học"
        >
          <span style={{ fontSize: '0.75rem', color: '#94a3b8', padding: '0 6px', fontWeight: 600 }}>⏱️</span>
          {[
            { value: 45, label: "45'" },
            { value: 90, label: "90'" },
            { value: 180, label: "180'" }
          ].map(opt => (
            <button
              key={opt.value}
              type="button"
              className={`btn-mode ${sessionDuration === opt.value ? 'active' : ''}`}
              onClick={() => onDurationChange(opt.value)}
              style={{
                padding: '5px 9px',
                fontSize: '0.78rem',
                fontWeight: sessionDuration === opt.value ? '700' : '500',
                borderRadius: '7px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                color: sessionDuration === opt.value ? '#fff' : '#94a3b8',
                background: sessionDuration === opt.value
                  ? 'linear-gradient(135deg, #0ea5e9, #0284c7)'
                  : 'transparent',
                boxShadow: sessionDuration === opt.value ? '0 2px 8px rgba(14, 165, 233, 0.4)' : 'none'
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* Policy Mode Selector: CQL Explicit vs Inverse RL */}
        <div
          className="policy-selector"
          style={{
            display: 'flex',
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.65)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '10px',
            padding: '3px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.25)'
          }}
          title="Chọn giải thuật sinh khuyến nghị sư phạm"
        >
          <button
            type="button"
            className={`btn-mode ${policyMode === 'cql_explicit' ? 'active' : ''}`}
            onClick={() => {
              setPolicyMode('cql_explicit');
              addToast('Đã chuyển sang chế độ CQL (Hàm thưởng đa mục tiêu cải tiến)', 'info');
            }}
            style={{
              padding: '6px 12px',
              fontSize: '0.8rem',
              fontWeight: policyMode === 'cql_explicit' ? '700' : '500',
              borderRadius: '7px',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              color: policyMode === 'cql_explicit' ? '#fff' : '#94a3b8',
              background: policyMode === 'cql_explicit'
                ? 'linear-gradient(135deg, #3b82f6, #1d4ed8)'
                : 'transparent',
              boxShadow: policyMode === 'cql_explicit' ? '0 2px 8px rgba(59, 130, 246, 0.4)' : 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <span style={{ fontSize: '0.9rem' }}>🎯</span>
            CQL Đa Mục Tiêu
          </button>

          <button
            type="button"
            className={`btn-mode ${policyMode === 'cql_irl' ? 'active' : ''}`}
            onClick={() => {
              setPolicyMode('cql_irl');
              addToast('Đã chuyển sang chế độ Inverse RL (Phần thưởng ẩn từ chuyên gia sư phạm)', 'info');
            }}
            style={{
              padding: '6px 12px',
              fontSize: '0.8rem',
              fontWeight: policyMode === 'cql_irl' ? '700' : '500',
              borderRadius: '7px',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              color: policyMode === 'cql_irl' ? '#fff' : '#94a3b8',
              background: policyMode === 'cql_irl'
                ? 'linear-gradient(135deg, #8b5cf6, #6d28d9)'
                : 'transparent',
              boxShadow: policyMode === 'cql_irl' ? '0 2px 8px rgba(139, 92, 246, 0.4)' : 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <span style={{ fontSize: '0.9rem' }}>🧠</span>
            Inverse RL (Ẩn)
          </button>
        </div>

        {/* Nạp Lớp Button */}
        <button className="btn btn-glass" onClick={() => onInitClassroom(true)}>
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Nạp Lớp (53 SV)
        </button>

        {/* Auto-Play Toggle Button */}
        <button
          className={`btn ${isAutoPlay ? 'btn-danger' : 'btn-glass'}`}
          style={isAutoPlay ? { backgroundColor: '#ef4444', color: '#fff', border: 'none' } : {}}
          onClick={() => {
            setIsAutoPlay(!isAutoPlay);
            addToast(isAutoPlay ? "Đã dừng Auto-Play thời gian!" : "Đã bật Auto-Play: Lớp học tự động tiến triển 5 phút / 3 giây!", "info");
          }}
        >
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            {isAutoPlay ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            )}
          </svg>
          {isAutoPlay ? "Tạm Dừng Giả Lập" : "Tự Động Tiến Triển"}
        </button>

        {/* Step +5 min Button */}
        <button
          className="btn btn-primary"
          onClick={() => onStepClassroom(telemetryAction || "lecture", 5.0)}
          title={`Tiến triển tiếp 5 phút cho hoạt động ${(telemetryAction || "lecture").toUpperCase()}`}
        >
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          +5' {(telemetryAction || "lecture").toUpperCase()}
        </button>

        {/* What-If Button */}
        <button className="btn btn-accent" onClick={onRunWhatIf}>
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Mô Phỏng What-If
        </button>
      </div>
    </header>
  );
}
