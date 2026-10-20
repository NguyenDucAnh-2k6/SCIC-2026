import React from 'react';

export default function ModifyActionModal({
  show,
  onClose,
  modifyAction,
  setModifyAction,
  modifyReason,
  setModifyReason,
  onSubmit
}) {
  if (!show) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-card">
        <div className="modal-header">
          <div className="modal-title">Điều Chỉnh Hành Động Giáo Viên</div>
          <button className="modal-close-btn" onClick={onClose}>&times;</button>
        </div>

        <div style={{ marginBottom: "16px" }}>
          <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "6px" }}>
            Chọn hành động sư phạm thay thế:
          </label>
          <select
            value={modifyAction}
            onChange={e => setModifyAction(e.target.value)}
            style={{ width: "100%", padding: "10px", borderRadius: "8px", background: "rgba(15, 23, 42, 0.9)", border: "1px solid var(--border-subtle)", color: "#fff", fontFamily: "inherit" }}
          >
            <option value="groupwork">Groupwork (Thảo luận nhóm - 10 phút)</option>
            <option value="quiz">Quiz (Kiểm tra nhanh - 3 phút)</option>
            <option value="lecture">Lecture (Tiếp tục giảng bài - 15 phút)</option>
            <option value="discussion">Discussion (Hỏi đáp toàn lớp - 8 phút)</option>
            <option value="break">Break (Nghỉ giải lao - 5 phút)</option>
          </select>
        </div>

        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "6px" }}>
            Lý do điều chỉnh (Tùy chọn):
          </label>
          <input
            type="text"
            value={modifyReason}
            onChange={e => setModifyReason(e.target.value)}
            placeholder="Ví dụ: Sinh viên cần giải lao vì không khí ngột ngạt..."
            style={{ width: "100%", padding: "10px", borderRadius: "8px", background: "rgba(15, 23, 42, 0.9)", border: "1px solid var(--border-subtle)", color: "#fff", fontFamily: "inherit" }}
          />
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px" }}>
          <button className="btn btn-glass" onClick={onClose}>Hủy</button>
          <button
            className="btn btn-primary"
            onClick={() => {
              onClose();
              onSubmit(modifyAction, modifyReason || "Điều chỉnh thủ công theo đánh giá của giáo viên.");
            }}
          >
            Xác Nhận & Áp Dụng
          </button>
        </div>
      </div>
    </div>
  );
}
