import React from 'react';

export default function RecommendationCard({
  recommendation,
  activeDecision,
  onTeacherDecision,
  onOpenModifyModal,
  onRequestNextRecommendation
}) {
  const getPolicyBadge = () => {
    if (recommendation.evidence_level === "heuristic_fallback") {
      return { text: "Safe Fallback Rule", color: "#f59e0b" };
    }
    if (recommendation.evidence_level === "inverse_rl_policy") {
      return { text: "Inverse RL (MaxEnt Expert)", color: "#a855f7" };
    }
    return { text: "CQL Offline Policy", color: "var(--primary-glow)" };
  };

  const badge = getPolicyBadge();

  // Khi buổi học (3 giờ) đã hoàn tất
  if (recommendation.is_completed || recommendation.evidence_level === "session_completed") {
    return (
      <div className="glass-box ai-card" style={{ borderColor: "rgba(168, 85, 247, 0.5)" }}>
        <div className="ai-card-badge" style={{ background: "rgba(168, 85, 247, 0.2)", color: "#c084fc" }}>
          <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
          </svg>
          Buổi Học Đã Hoàn Tất
        </div>
        <div className="ai-action-hero">
          <div className="ai-action-name" style={{ color: "#c084fc", fontSize: "1.4rem" }}>
            KẾT THÚC PHIÊN HỌC (3H)
          </div>
          <div className="ai-action-duration">
            180 / 180 Phút
          </div>
        </div>
        <div className="ai-explanation" style={{ borderLeft: "3px solid #a855f7", background: "rgba(168, 85, 247, 0.06)" }}>
          {recommendation.explanation || "Buổi học 3 giờ đã hoàn tất trọn vẹn! Mô hình số đã lưu lại toàn bộ diễn biến năng lực và độ mệt mỏi của học sinh."}
        </div>
      </div>
    );
  }

  // Nếu giáo viên đã phê duyệt và hành động đang được thực thi
  if (activeDecision && activeDecision.isExecuting) {
    const isAccepted = activeDecision.status === "ACCEPTED";
    return (
      <div className="glass-box ai-card" style={{ borderColor: isAccepted ? "rgba(16, 185, 129, 0.5)" : "rgba(59, 130, 246, 0.5)" }}>
        <div className="ai-card-badge" style={{ background: isAccepted ? "rgba(16, 185, 129, 0.2)" : "rgba(59, 130, 246, 0.2)", color: isAccepted ? "#34d399" : "#60a5fa" }}>
          <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
          </svg>
          {isAccepted ? "Đã Phê Duyệt & Đang Thực Thi" : "Đã Điều Chỉnh & Đang Thực Thi"}
        </div>

        <div className="ai-action-hero">
          <div className="ai-action-name" style={{ color: isAccepted ? "#34d399" : "#60a5fa" }}>
            {activeDecision.action.toUpperCase()}
          </div>
          <div className="ai-action-duration">
            {activeDecision.elapsed !== undefined && activeDecision.elapsed > 0 ? `${activeDecision.elapsed} / ${activeDecision.duration} Phút` : `${activeDecision.duration} Phút`}
          </div>
        </div>

        <div className="ai-explanation" style={{ borderLeft: `3px solid ${isAccepted ? '#10b981' : '#3b82f6'}`, background: isAccepted ? 'rgba(16, 185, 129, 0.06)' : 'rgba(59, 130, 246, 0.06)' }}>
          Lớp học đang thực thi hoạt động <strong>{activeDecision.action.toUpperCase()}</strong> ({activeDecision.duration} phút) theo quyết định của giáo viên. Các mô hình mạng nơ-ron đang theo dõi phản ứng năng lực của sinh viên.
        </div>

        <div className="evidence-meta">
          <div className="meta-item">
            <div className="label">Trạng thái lớp</div>
            <div className="val" style={{ color: "var(--color-success)", fontWeight: 700 }}>Đang diễn ra</div>
          </div>
          <div className="meta-item">
            <div className="label">Cơ chế quyết định</div>
            <div className="val" style={{ color: badge.color, fontWeight: 600 }}>{badge.text}</div>
          </div>
          <div className="meta-item">
            <div className="label">Độ phủ quan sát</div>
            <div className="val">{((recommendation.coverage || 0.92) * 100).toFixed(1)}% (Tin cậy)</div>
          </div>
          <div className="meta-item">
            <div className="label">Độ trễ phản hồi</div>
            <div className="val">{recommendation.latency_ms || 1.8} ms</div>
          </div>
        </div>

        <div className="teacher-actions-group">
          <button
            className="btn btn-primary"
            style={{ width: "100%", justifyContent: "center", padding: "12px", background: "linear-gradient(135deg, #0ea5e9, #3b82f6)" }}
            onClick={onRequestNextRecommendation}
          >
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Xin Khuyến Nghị Bước Kế Tiếp
          </button>

          <div className="teacher-actions-buttons">
            <button className="btn btn-glass" style={{ width: "100%", justifyContent: "center" }} onClick={onOpenModifyModal}>
              <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              Đổi Sang Hoạt Động Khác
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Trạng thái chờ giáo viên ra quyết định (AI đề xuất mới)
  return (
    <div className="glass-box ai-card">
      <div className="ai-card-badge">
        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        Hệ Thống Trợ Lý AI Khuyến Nghị
      </div>

      <div className="ai-action-hero">
        <div className="ai-action-name">{recommendation.recommended_action || "groupwork"}</div>
        <div className="ai-action-duration">{Math.round((recommendation.planned_duration_s || 600) / 60)} Phút</div>
      </div>

      <div className="ai-explanation">
        {recommendation.explanation}
      </div>

      <div className="evidence-meta">
        <div className="meta-item">
          <div className="label">Cơ chế quyết định</div>
          <div className="val" style={{ color: badge.color, fontWeight: "600" }}>
            {badge.text}
          </div>
        </div>
        <div className="meta-item">
          <div className="label">Độ phủ quan sát</div>
          <div className="val">{((recommendation.coverage || 0.92) * 100).toFixed(1)}% (Tin cậy)</div>
        </div>
        <div className="meta-item">
          <div className="label">Độ bất định</div>
          <div className="val">{(recommendation.uncertainty || 0.08).toFixed(2)} (Thấp)</div>
        </div>
        <div className="meta-item">
          <div className="label">Độ trễ phản hồi</div>
          <div className="val">{recommendation.latency_ms || 1.8} ms</div>
        </div>
      </div>

      <div className="teacher-actions-group">
        <button
          className="btn btn-success"
          style={{ width: "100%", justifyContent: "center", padding: "12px" }}
          onClick={() => onTeacherDecision("ACCEPTED")}
        >
          <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
          </svg>
          Chấp Nhận Khuyến Nghị (Accept)
        </button>

        <div className="teacher-actions-buttons">
          <button className="btn btn-glass" style={{ justifyContent: "center" }} onClick={onOpenModifyModal}>
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
            Điều Chỉnh
          </button>

          <button
            className="btn btn-danger"
            style={{ justifyContent: "center" }}
            onClick={() => onTeacherDecision("REJECTED", null, "Giáo viên nhận thấy lớp chưa sẵn sàng.")}
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
            Từ Chối
          </button>
        </div>
      </div>
    </div>
  );
}
