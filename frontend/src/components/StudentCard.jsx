import React from 'react';

export default function StudentCard({ student, isVisible, onSelect }) {
  const mClass = student.mastery > 0.7 ? "high" : (student.mastery >= 0.4 ? "med" : "low");

  return (
    <div
      className={`desk-card ${student.isWeak ? "is-weak" : ""}`}
      style={{ opacity: isVisible ? 1 : 0.2 }}
      onClick={() => onSelect(student)}
      title={`Sinh viên ${student.id} • Mastery: ${(student.mastery * 100).toFixed(0)}%`}
    >
      <div className="desk-header">
        <span className="student-id">{student.id}</span>
        <span className={`mastery-indicator ${mClass}`}></span>
      </div>
      <div className="desk-meters">
        <div className="micro-bar-wrapper" title={`Năng lực (Mastery): ${(student.mastery * 100).toFixed(0)}%`}>
          <span className="label">M</span>
          <div className="micro-bar-track">
            <div className="micro-bar-fill mas" style={{ width: `${student.mastery * 100}%` }}></div>
          </div>
        </div>
        <div className="micro-bar-wrapper" title={`Độ tập trung: ${(student.attention * 100).toFixed(0)}%`}>
          <span className="label">Att</span>
          <div className="micro-bar-track">
            <div className="micro-bar-fill att" style={{ width: `${student.attention * 100}%` }}></div>
          </div>
        </div>
        <div className="micro-bar-wrapper" title={`Mệt mỏi: ${(student.fatigue * 100).toFixed(0)}%`}>
          <span className="label">Fat</span>
          <div className="micro-bar-track">
            <div className="micro-bar-fill fat" style={{ width: `${student.fatigue * 100}%` }}></div>
          </div>
        </div>
      </div>
    </div>
  );
}
