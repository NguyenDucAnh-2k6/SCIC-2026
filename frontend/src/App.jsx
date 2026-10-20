import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import TelemetryRow from './components/TelemetryRow';
import ClassroomGrid from './components/ClassroomGrid';
import RecommendationCard from './components/RecommendationCard';
import WhatIfPanel from './components/WhatIfPanel';
import AuditTrail from './components/AuditTrail';
import StudentModal from './components/StudentModal';
import ModifyActionModal from './components/ModifyActionModal';
import ToastContainer from './components/ToastContainer';

export default function App() {
  const [classId] = useState("class_scic_01");
  const [students, setStudents] = useState([]);
  const [isAutoPlay, setIsAutoPlay] = useState(false);
  const [sessionDuration, setSessionDuration] = useState(180); // 45 | 90 | 180
  const [policyMode, setPolicyMode] = useState("cql_explicit"); // "cql_explicit" | "cql_irl"
  const [activeDecision, setActiveDecision] = useState(null);
  const activeDecisionRef = useRef(null);

  const [telemetry, setTelemetry] = useState({
    meanMastery: 0.56,
    q25Mastery: 0.38,
    q75Mastery: 0.71,
    meanAttention: 0.78,
    meanFatigue: 0.18,
    workload: 3.5,
    timeRemainingMinutes: 180.0,
    timeElapsedMinutes: 0.0,
    co2: 650,
    temp: 24.5,
    noise: 45,
    currentAction: "lecture"
  });

  const isSteppingRef = useRef(false);
  const currentActionRef = useRef(telemetry.currentAction);
  useEffect(() => {
    currentActionRef.current = telemetry.currentAction;
  }, [telemetry.currentAction]);

  // Auto-tick timer: khi bật AutoPlay, sau mỗi 5 giây tự động tiến triển 5 phút lớp học
  useEffect(() => {
    let timer = null;
    if (isAutoPlay) {
      timer = setInterval(async () => {
        if (isSteppingRef.current) return;
        if (telemetry.timeRemainingMinutes <= 0.0) {
          setIsAutoPlay(false);
          addToast(`Buổi học (${sessionDuration} phút) đã hoàn tất! Các mô hình đã ghi nhận toàn bộ tiến trình học tập.`, "success");
          return;
        }
        isSteppingRef.current = true;
        try {
          await stepClassroom(currentActionRef.current || "lecture", 5.0);
        } catch (e) {
          console.warn("Auto-step error:", e);
        } finally {
          isSteppingRef.current = false;
        }
      }, 5000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isAutoPlay, telemetry.timeRemainingMinutes]);

  const [recommendation, setRecommendation] = useState({
    decision_id: "dec_init",
    recommended_action: "groupwork",
    planned_duration_s: 600,
    evidence_level: "cql_offline_policy",
    coverage: 0.92,
    uncertainty: 0.08,
    explanation: "Phát hiện 28% sinh viên nhóm dưới (Q25) đang mất tập trung khi nghe giảng kéo dài. Đề xuất chuyển sang thảo luận nhóm để kích hoạt học tập đồng đẳng.",
    latency_ms: 1.8
  });

  const [activeFilter, setActiveFilter] = useState("all");
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [showModifyModal, setShowModifyModal] = useState(false);
  const [modifyAction, setModifyAction] = useState("groupwork");
  const [modifyReason, setModifyReason] = useState("");
  const [auditTrail, setAuditTrail] = useState([]);
  const [whatIfSims, setWhatIfSims] = useState([
    { candidate_action: "groupwork", mean_delta_mastery: 0.045, median_reward: 0.42 },
    { candidate_action: "quiz", mean_delta_mastery: 0.035, median_reward: 0.31 },
    { candidate_action: "lecture", mean_delta_mastery: 0.020, median_reward: 0.15 },
    { candidate_action: "discussion", mean_delta_mastery: 0.030, median_reward: 0.35 },
    { candidate_action: "break", mean_delta_mastery: 0.000, median_reward: 0.28 }
  ]);
  const [toasts, setToasts] = useState([]);
  const stepCountRef = useRef(0);

  const addToast = (msg, type = "info") => {
    const id = Date.now() + Math.random();
    setToasts(prev => {
      // Giữ tối đa 3 toast mới nhất để không nghẽn DOM và hiệu ứng mượt mà
      const trimmed = prev.slice(-2);
      return [...trimmed, { id, msg, type }];
    });
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 2500);
  };

  // Khởi tạo ban đầu
  useEffect(() => {
    initClassroom();
  }, []);

  // Khi thay đổi chế độ thuật toán (CQL vs Inverse RL), gọi lại khuyến nghị ngay
  useEffect(() => {
    fetchRecommendation(telemetry.timeRemainingMinutes, telemetry.workload, policyMode);
  }, [policyMode]);

  const initClassroom = async (notify = false, duration = sessionDuration) => {
    try {
      const res = await fetch("/v1/demo/init-classroom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_id: classId, cohort_size: 48, seed: 42, duration_minutes: duration })
      });
      if (res.ok) {
        const data = await res.json();
        setStudents(data.students || []);
        if (data.telemetry) setTelemetry(prev => ({ ...prev, ...data.telemetry }));
        fetchRecommendation(data.telemetry?.timeRemainingMinutes || duration, data.telemetry?.workload || 3.5, policyMode);
        runWhatIf(true);
        if (notify) addToast(`Đã nạp thành công 48 sinh viên! Buổi học: ${duration} phút.`, "success");
        return;
      }
    } catch (err) {
      console.warn("Backend not reached, fallback to local cohort generator:", err);
    }
    generateLocalCohort();
    setTelemetry(prev => ({ ...prev, timeRemainingMinutes: duration, timeElapsedMinutes: 0.0 }));
    runWhatIf(true);
    if (notify) addToast(`Đã nạp thành công 48 sinh viên! Buổi học: ${duration} phút.`, "success");
  };

  const handleDurationChange = (newDuration) => {
    setSessionDuration(newDuration);
    initClassroom(true, newDuration);
  };

  const generateLocalCohort = () => {
    const list = [];
    const ednetIds = [
      "u10023", "u10025", "u10034", "u10041", "u10052", "u10067", "u10078", "u10089",
      "u10101", "u10115", "u10122", "u10134", "u10145", "u10156", "u10167", "u10178",
      "u10190", "u10203", "u10214", "u10228", "u10239", "u10250", "u10261", "u10275",
      "u10286", "u10298", "u10310", "u10321", "u10335", "u10346", "u10359", "u10370",
      "u10382", "u10394", "u10405", "u10418", "u10429", "u10441", "u10452", "u10465",
      "u10476", "u10488", "u10501", "u10512", "u10524", "u10536", "u10547", "u10559"
    ];

    for (let i = 0; i < 48; i++) {
      const rawM = (Math.random() + Math.random() + Math.random()) / 3.0;
      const mastery = Math.min(0.95, Math.max(0.15, rawM + (Math.random() * 0.2 - 0.1)));
      const attention = Math.min(0.95, Math.max(0.35, 0.75 + (Math.random() * 0.3 - 0.15)));
      const fatigue = Math.min(0.80, Math.max(0.05, 0.15 + (Math.random() * 0.2 - 0.05)));

      list.push({
        id: ednetIds[i] || `u10${i < 10 ? '0' + i : i}`,
        row: Math.floor(i / 8) + 1,
        col: (i % 8) + 1,
        mastery: parseFloat(mastery.toFixed(3)),
        attention: parseFloat(attention.toFixed(3)),
        fatigue: parseFloat(fatigue.toFixed(3)),
        isWeak: mastery < 0.40,
        recentLogs: [
          { q: `q${1000 + i} (Algebra)`, ans: mastery > 0.4 ? "Correct (b)" : "Incorrect (a)", elapsed: `${Math.floor(15 + Math.random() * 20)}s` },
          { q: `q${2000 + i} (Geometry)`, ans: mastery > 0.55 ? "Correct (c)" : "Incorrect (d)", elapsed: `${Math.floor(20 + Math.random() * 25)}s` }
        ]
      });
    }
    setStudents(list);
    computeStats(list);
  };

  const computeStats = (currentStudents) => {
    if (!currentStudents.length) return;
    const masteries = currentStudents.map(s => s.mastery).sort((a, b) => a - b);
    const attentions = currentStudents.map(s => s.attention);
    const fatigues = currentStudents.map(s => s.fatigue);

    const meanM = masteries.reduce((a, b) => a + b, 0) / masteries.length;
    const q25 = masteries[Math.floor(masteries.length * 0.25)];
    const q75 = masteries[Math.floor(masteries.length * 0.75)];
    const meanAtt = attentions.reduce((a, b) => a + b, 0) / attentions.length;
    const meanFat = fatigues.reduce((a, b) => a + b, 0) / fatigues.length;

    setTelemetry(prev => ({
      ...prev,
      meanMastery: meanM,
      q25Mastery: q25,
      q75Mastery: q75,
      meanAttention: meanAtt,
      meanFatigue: meanFat
    }));
  };

  const fetchRecommendation = async (timeRem = telemetry.timeRemainingMinutes, workload = telemetry.workload, mode = policyMode) => {
    try {
      const res = await fetch("/v1/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          class_id: classId,
          topic: "EdNet Offline RL",
          time_remaining_minutes: timeRem,
          current_workload: workload,
          run_simulation: false,
          policy_mode: mode
        })
      });
      if (res.ok) {
        const data = await res.json();
        setRecommendation(data);
      }
    } catch (e) {
      console.warn("Could not fetch recommendation from API:", e);
    }
  };

  const ACTION_DURATIONS = {
    groupwork: 10,
    quiz: 3,
    lecture: 15,
    discussion: 8,
    break: 5
  };

  const stepClassroom = async (action, dtMinutes = 5.0, fromDecision = false) => {
    stepCountRef.current += 1;
    const currentStepCount = stepCountRef.current;
    currentActionRef.current = action;

    try {
      const res = await fetch("/v1/demo/step-classroom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          class_id: classId,
          action: action,
          duration_minutes: dtMinutes
        })
      });
      if (res.ok) {
        const data = await res.json();
        setStudents(data.students || []);
        if (data.telemetry) {
          setTelemetry(prev => ({
            ...prev,
            ...data.telemetry,
            currentAction: action // Luôn đồng bộ hành động vừa thực thi
          }));
        }

        // Kiểm tra tiến trình hoạt động giáo viên đang phê duyệt thực thi
        if (activeDecisionRef.current && activeDecisionRef.current.isExecuting) {
          const currentElapsedInAct = (activeDecisionRef.current.elapsed || 0) + dtMinutes;
          if (currentElapsedInAct >= activeDecisionRef.current.duration) {
            addToast(`Hoạt động [${action.toUpperCase()}] (${activeDecisionRef.current.duration} phút) đã hoàn tất! AI đang tính toán bước kế tiếp.`, "success");
            setActiveDecision(null);
            activeDecisionRef.current = null;
            fetchRecommendation(data.telemetry?.timeRemainingMinutes, data.telemetry?.workload, policyMode)
              .catch(err => console.warn("Background rec error:", err));
          } else {
            const updatedDecision = { ...activeDecisionRef.current, elapsed: currentElapsedInAct };
            setActiveDecision(updatedDecision);
            activeDecisionRef.current = updatedDecision;
          }
        } else if (!fromDecision) {
          addToast(`Đang thực thi [${action.toUpperCase()}] (+${dtMinutes} phút)...`, "info");
          fetchRecommendation(data.telemetry?.timeRemainingMinutes, data.telemetry?.workload, policyMode)
            .catch(err => console.warn("Background rec error:", err));
        }

        // Chỉ chạy What-If khi thao tác thủ công hoặc mỗi 3 bước (15 phút lớp học) để tránh nghẽn CPU
        if (!isAutoPlay || currentStepCount % 3 === 0) {
          runWhatIf(true).catch(err => console.warn("Background what-if error:", err));
        }
        return;
      }
    } catch (e) {
      console.warn("Backend step-classroom failed, using local simulation:", e);
    }

    // Local simulation fallback
    const dtFactor = dtMinutes / 5.0;
    const updated = students.map(s => {
      let m = s.mastery;
      let a = s.attention;
      let f = s.fatigue;
      const noiseM = (Math.random() - 0.5) * 0.015;
      const noiseA = (Math.random() - 0.5) * 0.02;
      const noiseF = (Math.random() - 0.5) * 0.015;

      const effectiveAtt = a * Math.max(0.1, 1.0 - 0.7 * f * f);

      if (action === "lecture") {
        m = Math.min(1.0, Math.max(0.0, m + 0.018 * dtFactor * effectiveAtt + noiseM));
        a = Math.max(0.10, Math.min(1.0, a - 0.05 * dtFactor + noiseA));
        f = Math.min(1.0, f + 0.045 * dtFactor + noiseF);
      } else if (action === "quiz") {
        if (f > 0.55) {
          m = Math.max(0.0, m - 0.012 * dtFactor + noiseM);
          a = Math.max(0.10, a - 0.04 * dtFactor + noiseA);
          f = Math.min(1.0, f + 0.075 * dtFactor + noiseF);
        } else {
          m = Math.min(1.0, m + 0.032 * dtFactor + noiseM);
          a = Math.min(1.0, a + 0.03 * dtFactor + noiseA);
          f = Math.min(1.0, f + 0.055 * dtFactor + noiseF);
        }
      } else if (action === "groupwork") {
        if (f > 0.65) {
          m = Math.min(1.0, m + 0.002 * dtFactor + noiseM);
          a = Math.max(0.10, a - 0.03 * dtFactor + noiseA);
          f = Math.min(1.0, f + 0.04 * dtFactor + noiseF);
        } else {
          m = Math.min(1.0, m + 0.035 * dtFactor + noiseM);
          a = Math.min(1.0, a + 0.04 * dtFactor + noiseA);
          f = Math.min(1.0, f + 0.025 * dtFactor + noiseF);
        }
      } else if (action === "discussion") {
        m = Math.min(1.0, m + 0.022 * dtFactor * effectiveAtt + noiseM);
        a = Math.min(1.0, a + 0.05 * dtFactor + noiseA);
        f = Math.min(1.0, f + 0.035 * dtFactor + noiseF);
      } else if (action === "break") {
        m = Math.max(0.0, m - 0.002 * dtFactor + noiseM);
        a = Math.min(1.0, a + 0.25 * dtFactor + noiseA);
        f = Math.max(0.05, f - 0.28 * dtFactor + noiseF);
      }

      return {
        ...s,
        mastery: parseFloat(m.toFixed(3)),
        attention: parseFloat(a.toFixed(3)),
        fatigue: parseFloat(f.toFixed(3)),
        isWeak: m < 0.40
      };
    });

    setStudents(updated);
    computeStats(updated);

    const costMap = { lecture: 1.0, quiz: 1.8, groupwork: 2.0, discussion: 1.5, break: -1.5 };
    const actCost = (costMap[action] || 1.0) * dtFactor;

    setTelemetry(prev => {
      const co2Delta = action === "break" ? -140 : (action === "lecture" ? 15 : 22);
      const newCo2 = Math.min(1400, Math.max(450, prev.co2 + co2Delta * dtFactor));
      const newElapsed = (prev.timeElapsedMinutes || 0) + dtMinutes;
      const newRemaining = Math.max(0.0, 180.0 - newElapsed);
      const newWorkload = Math.max(0.5, (prev.workload * 0.95) + actCost);

      return {
        ...prev,
        timeElapsedMinutes: newElapsed,
        timeRemainingMinutes: newRemaining,
        workload: parseFloat(newWorkload.toFixed(1)),
        co2: Math.round(newCo2),
        currentAction: action
      };
    });

    if (activeDecisionRef.current && activeDecisionRef.current.isExecuting) {
      const currentElapsedInAct = (activeDecisionRef.current.elapsed || 0) + dtMinutes;
      if (currentElapsedInAct >= activeDecisionRef.current.duration) {
        addToast(`Hoạt động [${action.toUpperCase()}] (${activeDecisionRef.current.duration} phút) đã hoàn tất! AI đang tính toán bước kế tiếp.`, "success");
        setActiveDecision(null);
        activeDecisionRef.current = null;
        fetchRecommendation(telemetry.timeRemainingMinutes, telemetry.workload, policyMode);
      } else {
        const updatedDecision = { ...activeDecisionRef.current, elapsed: currentElapsedInAct };
        setActiveDecision(updatedDecision);
        activeDecisionRef.current = updatedDecision;
      }
    } else if (!fromDecision) {
      addToast(`Đang thực thi [${action.toUpperCase()}] (+${dtMinutes} phút)...`, "info");
      fetchRecommendation(telemetry.timeRemainingMinutes, telemetry.workload, policyMode);
    }

    runWhatIf(true);
  };

  const handleTeacherDecision = async (status, overrideAction = null, reason = "") => {
    const executedAction = overrideAction || recommendation.recommended_action || "groupwork";
    const durationMin = ACTION_DURATIONS[executedAction] || 5;

    try {
      await fetch("/v1/teacher-decisions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          decision_id: recommendation.decision_id || `dec_${Date.now()}`,
          status: status,
          executed_action: executedAction,
          actual_duration_s: durationMin * 60,
          override_reason: reason
        })
      });
    } catch (e) {
      console.warn("Could not sync teacher decision:", e);
    }

    const timeStr = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setAuditTrail(prev => [
      {
        time: timeStr,
        status: status,
        action: executedAction,
        reason: reason || (status === "ACCEPTED" ? "Chấp thuận gợi ý từ mô hình AI." : "Giáo viên ghi đè quyết định.")
      },
      ...prev
    ]);

    if (status !== "REJECTED") {
      // Khóa và hiển thị đúng hành động giáo viên đã chọn kèm theo dõi tiến độ
      const decisionObj = {
        action: executedAction,
        status: status,
        duration: durationMin,
        elapsed: 0,
        isExecuting: true
      };
      setActiveDecision(decisionObj);
      activeDecisionRef.current = decisionObj;
      currentActionRef.current = executedAction;
      setTelemetry(prev => ({ ...prev, currentAction: executedAction }));

      await stepClassroom(executedAction, durationMin, true);
      addToast(`Giáo viên đã ${status === "ACCEPTED" ? "chấp nhận" : "chọn"} [${executedAction.toUpperCase()}]. Lớp học đang thực thi ${durationMin} phút!`, "success");
    } else {
      addToast("Giáo viên đã từ chối khuyến nghị. Giữ nguyên hoạt động lớp.", "warning");
    }
  };

  const handleRequestNextRecommendation = async () => {
    setActiveDecision(null);
    activeDecisionRef.current = null;
    addToast("Đang xin gợi ý sư phạm cho bước tiếp theo...", "info");
    await fetchRecommendation(telemetry.timeRemainingMinutes, telemetry.workload, policyMode);
  };

  const runWhatIf = async (silent = false) => {
    if (!silent) addToast("Đang chạy mô phỏng What-If Monte Carlo...", "info");
    try {
      const res = await fetch("/v1/simulations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          class_id: classId,
          candidate_actions: ["groupwork", "quiz", "lecture", "discussion", "break"],
          duration_minutes: 5.0,
          num_samples: 4
        })
      });
      if (res.ok) {
        const data = await res.json();
        setWhatIfSims(data.simulations);
        if (!silent) addToast("Mô phỏng What-If hoàn tất!", "success");
        return;
      }
    } catch (e) {
      console.warn("Simulation fallback:", e);
    }
    if (!silent) addToast("Mô phỏng What-If hoàn tất!", "success");
  };

  const filteredStudents = students.filter(s => {
    if (activeFilter === "weak") return s.mastery < 0.40;
    if (activeFilter === "low-att") return s.attention < 0.50;
    if (activeFilter === "tired") return s.fatigue > 0.40;
    return true;
  });

  return (
    <div className="app-container">
      {/* Top Header with Policy Mode Switcher */}
      <Header
        isAutoPlay={isAutoPlay}
        setIsAutoPlay={setIsAutoPlay}
        policyMode={policyMode}
        setPolicyMode={setPolicyMode}
        sessionDuration={sessionDuration}
        onDurationChange={handleDurationChange}
        onInitClassroom={() => initClassroom(true, sessionDuration)}
        onStepClassroom={stepClassroom}
        onRunWhatIf={() => runWhatIf(false)}
        addToast={addToast}
        telemetryAction={telemetry.currentAction}
      />

      {/* Main Layout */}
      <main className="main-layout">
        <div className="center-content">
          {/* Telemetry Row */}
          <TelemetryRow telemetry={telemetry} />

          {/* Seating Map Section */}
          <ClassroomGrid
            students={students}
            filteredStudents={filteredStudents}
            activeFilter={activeFilter}
            setActiveFilter={setActiveFilter}
            currentAction={telemetry.currentAction}
            onSelectStudent={setSelectedStudent}
          />
        </div>

        {/* Right Sidebar: AI Decision Support & Teacher Action */}
        <aside className="sidebar-panel">
          <RecommendationCard
            recommendation={recommendation}
            activeDecision={activeDecision}
            onTeacherDecision={handleTeacherDecision}
            onOpenModifyModal={() => setShowModifyModal(true)}
            onRequestNextRecommendation={handleRequestNextRecommendation}
          />

          <WhatIfPanel whatIfSims={whatIfSims} />

          <AuditTrail auditTrail={auditTrail} />
        </aside>
      </main>

      {/* Modals & Toasts */}
      <StudentModal
        student={selectedStudent}
        onClose={() => setSelectedStudent(null)}
      />

      <ModifyActionModal
        show={showModifyModal}
        onClose={() => setShowModifyModal(false)}
        modifyAction={modifyAction}
        setModifyAction={setModifyAction}
        modifyReason={modifyReason}
        setModifyReason={setModifyReason}
        onSubmit={(act, rsn) => handleTeacherDecision("MODIFIED", act, rsn)}
      />

      <ToastContainer toasts={toasts} />
    </div>
  );
}
