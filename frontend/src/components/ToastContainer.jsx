import React from 'react';

export default function ToastContainer({ toasts }) {
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className="toast">
          {t.type === "success" && (
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="#10b981">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"/>
            </svg>
          )}
          {t.type === "warning" && (
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="#f59e0b">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
            </svg>
          )}
          {t.type === "info" && (
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="#38bdf8">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          )}
          <span>{t.msg}</span>
        </div>
      ))}
    </div>
  );
}
