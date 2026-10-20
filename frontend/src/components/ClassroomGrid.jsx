import React from 'react';
import StudentCard from './StudentCard';

export default function ClassroomGrid({
  students,
  filteredStudents,
  activeFilter,
  setActiveFilter,
  currentAction,
  onSelectStudent
}) {
  return (
    <section className="seating-section">
      <div className="section-header">
        <div className="section-title">
          <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ color: "var(--primary-glow)" }}>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
          Sơ Đồ Chỗ Ngồi Lớp Học Thông Minh ({students.length} Sinh Viên)
        </div>

        <div className="filter-pills">
          <button className={`filter-pill ${activeFilter === "all" ? "active" : ""}`} onClick={() => setActiveFilter("all")}>
            Tất cả ({students.length})
          </button>
          <button className={`filter-pill ${activeFilter === "weak" ? "active" : ""}`} onClick={() => setActiveFilter("weak")} style={{ color: "#fca5a5" }}>
            Nhóm Yếu Q25
          </button>
          <button className={`filter-pill ${activeFilter === "low-att" ? "active" : ""}`} onClick={() => setActiveFilter("low-att")}>
            Mất tập trung
          </button>
          <button className={`filter-pill ${activeFilter === "tired" ? "active" : ""}`} onClick={() => setActiveFilter("tired")}>
            Mệt mỏi cao
          </button>
        </div>
      </div>

      {/* Podium */}
      <div className="podium-board">
        <div className="podium-title">
          <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          Bục Giảng & Màn Hình Tương Tác
        </div>
        <div>
          <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginRight: "6px" }}>Hoạt động:</span>
          <span className="podium-action-active">{currentAction.toUpperCase()}</span>
        </div>
      </div>

      {/* Seating Grid */}
      <div className="classroom-grid">
        {students.map(s => {
          const isVisible = filteredStudents.some(fs => fs.id === s.id);
          return (
            <StudentCard
              key={s.id}
              student={s}
              isVisible={isVisible}
              onSelect={onSelectStudent}
            />
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: "20px", fontSize: "0.78rem", color: "var(--text-muted)", justifyContent: "center", paddingTop: "10px", borderTop: "1px solid var(--border-subtle)", flexWrap: "wrap" }}>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className="mastery-indicator high"></span> Nắm vững (&gt;70%)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className="mastery-indicator med"></span> Trung bình (40-70%)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className="mastery-indicator low"></span> Nguy cơ hổng kiến thức (&lt;40%)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ color: "var(--color-danger)", fontWeight: 800 }}>!</span> Cần giáo viên can thiệp
        </span>
      </div>
    </section>
  );
}
