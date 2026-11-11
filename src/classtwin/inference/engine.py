import time
import numpy as np
from typing import Dict, Any, List, Optional
from ..policies.action_space import (
    PEDAGOGICAL_ACTIONS,
    DEFAULT_DURATIONS_MINUTES,
    get_valid_action_mask
)
from ..policies.rules import RuleBasedPolicy
from ..policies.cql import DiscreteCQL
from ..twin.classroom_env import ClassroomDigitalTwinEnv
from ..twin.scenarios import counterfactual_twin_rollout
from ..data.telemetry_state import build_state_vector, STATE_DIM_EXTENDED, STATE_DIM_LEGACY
from .state_store import ClassroomStateStore
from ..utils.logging import get_logger

logger = get_logger("classtwin.inference.engine")

class DecisionSupportEngine:
    """
    Implements Algorithm 4 (Online Inference & Pedagogical Decision Support Flow).
    Coordinates real-time ingestion, state cache, safe fallback, CQL policy, and digital twin simulation.
    Features robust tolerance for missing telemetry fields via empirical prior imputation.
    """
    def __init__(
        self,
        state_store: ClassroomStateStore,
        cql_policy: Optional[DiscreteCQL] = None,
        irl_policy: Optional[DiscreteCQL] = None,
        min_coverage_threshold: float = 0.60,
        max_uncertainty_threshold: float = 0.45,
        d_state: Optional[int] = None
    ):
        self.state_store = state_store
        self.cql_policy = cql_policy
        self.irl_policy = irl_policy
        self.min_coverage = min_coverage_threshold
        self.max_uncertainty = max_uncertainty_threshold
        self.rule_policy = RuleBasedPolicy()

        # Detect expected state dimension from policy network
        if d_state is not None:
            self.d_state = d_state
        elif cql_policy is not None and hasattr(cql_policy, "q_net"):
            self.d_state = cql_policy.q_net.net[0].in_features
        elif irl_policy is not None and hasattr(irl_policy, "q_net"):
            self.d_state = irl_policy.q_net.net[0].in_features
        else:
            self.d_state = STATE_DIM_EXTENDED

    def generate_recommendation(
        self,
        class_id: str,
        topic: str,
        time_remaining_minutes: float = 45.0,
        current_workload: float = 4.0,
        run_simulation_rollout: bool = True,
        env: Optional[ClassroomDigitalTwinEnv] = None,
        policy_mode: str = "cql_explicit",
        custom_telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes online recommendation pipeline.
        Tolerant to missing telemetry fields when deployed to third-party LMS/client APIs.
        """
        start_time = time.perf_counter()
        decision_id = f"dec_{class_id}_{int(time.time() * 1000)}"

        # 1. Fetch class aggregate state
        summary = self.state_store.get_class_summary(class_id)
        coverage = summary["coverage"]
        uncertainty = summary["uncertainty"]

        # 0. Check Session Completion
        if time_remaining_minutes <= 0.0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            rec = {
                "decision_id": decision_id,
                "class_id": class_id,
                "topic": topic,
                "recommended_action": "completed",
                "planned_duration_s": 0,
                "evidence_level": "session_completed",
                "coverage": coverage,
                "uncertainty": uncertainty,
                "policy_mode": policy_mode,
                "explanation": "Buổi học (3 giờ) đã hoàn thành! Các mô hình số đã ghi nhận toàn bộ tiến trình và phản ứng năng lực của học sinh.",
                "requires_teacher_decision": False,
                "is_completed": True,
                "latency_ms": round(elapsed_ms, 2)
            }
            self.state_store.record_decision(decision_id, rec)
            return rec

        # 2. Formulate Robust Multi-Dimensional State Vector
        if env is not None:
            # Sync target state dimension
            env.d_state = self.d_state
            state_vec = env.get_state_vector()
            current_att = float(state_vec[5]) if len(state_vec) > 5 else 0.75
            current_fat = float(state_vec[7]) if len(state_vec) > 7 else (float(state_vec[6]) if len(state_vec) > 6 else 0.20)
            valid_mask = get_valid_action_mask(time_remaining_minutes, env.cumulative_workload)
            coverage = 1.0
            uncertainty = 0.05
            summary = {
                "mean_mastery": float(state_vec[0]),
                "q25_mastery": float(state_vec[1]),
                "q75_mastery": float(state_vec[2]),
                "n_enrolled": len(env.students)
            }
        else:
            # Construct from store summary and custom telemetry with missingness imputation
            merged_telemetry = {
                "mean_mastery": summary.get("mean_mastery", 0.55),
                "q25_mastery": summary.get("q25_mastery", 0.40),
                "q75_mastery": summary.get("q75_mastery", 0.70),
                "time_remaining_norm": time_remaining_minutes / 180.0,
                "teacher_workload_norm": current_workload / 20.0
            }
            if custom_telemetry:
                merged_telemetry.update(custom_telemetry)

            state_vec, _, feat_dict = build_state_vector(
                telemetry=merged_telemetry,
                d_state=self.d_state,
                episode_duration_minutes=180.0
            )
            current_att = feat_dict.get("mean_attention", 0.75)
            current_fat = feat_dict.get("session_fatigue_rate", 0.20)
            valid_mask = get_valid_action_mask(time_remaining_minutes, current_workload)

        # Check Safe Fallback conditions
        if coverage < self.min_coverage or uncertainty > self.max_uncertainty:
            action, reason = self.rule_policy.select_action(
                {"mean_mastery": summary["mean_mastery"], "mean_fatigue": current_fat, "mean_attention": current_att},
                valid_mask
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            rec = {
                "decision_id": decision_id,
                "class_id": class_id,
                "topic": topic,
                "recommended_action": action,
                "planned_duration_s": int(DEFAULT_DURATIONS_MINUTES[action] * 60),
                "evidence_level": "heuristic_fallback",
                "coverage": coverage,
                "uncertainty": uncertainty,
                "policy_mode": policy_mode,
                "explanation": f"Chưa đủ dữ liệu quan sát tin cậy ({coverage:.1%} sinh viên). {reason}",
                "requires_teacher_decision": True,
                "latency_ms": round(elapsed_ms, 2)
            }
            self.state_store.record_decision(decision_id, rec)
            return rec

        # Check last action to avoid monotonous consecutive repetition
        last_action_name = env.last_action if env is not None else None
        last_action_idx = PEDAGOGICAL_ACTIONS.index(last_action_name) if last_action_name in PEDAGOGICAL_ACTIONS else None

        # 3. Select Action via Policy (CQL Explicit vs Inverse RL vs Rule baseline)
        if policy_mode == "cql_irl" and self.irl_policy:
            _, chosen_action = self.irl_policy.select_action(
                state_vec,
                valid_mask,
                deterministic=False,
                temperature=0.35,
                repetition_action_idx=last_action_idx,
                repetition_penalty=0.6
            )
            evidence_level = "inverse_rl_policy"
            policy_desc = "Inverse RL (Học hàm thưởng ẩn MaxEnt từ chuyên gia)"
        elif self.cql_policy:
            _, chosen_action = self.cql_policy.select_action(
                state_vec,
                valid_mask,
                deterministic=False,
                temperature=0.35,
                repetition_action_idx=last_action_idx,
                repetition_penalty=0.6
            )
            evidence_level = "cql_offline_policy"
            policy_desc = "Offline CQL (Hàm thưởng đa mục tiêu cân bằng hiểu bài & kiệt sức)"
        else:
            chosen_action, _ = self.rule_policy.select_action(
                {"mean_mastery": summary["mean_mastery"], "mean_fatigue": current_fat, "mean_attention": current_att},
                valid_mask
            )
            evidence_level = "rule_based"
            policy_desc = "Heuristic Rule Based"

        duration_m = DEFAULT_DURATIONS_MINUTES[chosen_action]

        # 5. Counterfactual Simulation Rollout (What-if projection)
        rollout_stats = {}
        if run_simulation_rollout:
            sim_env = ClassroomDigitalTwinEnv(
                cohort_size=summary.get("n_enrolled", 53),
                d_state=self.d_state,
                seed=42
            )
            sim_env.reset()
            rollout_stats = counterfactual_twin_rollout(
                base_env=sim_env,
                candidate_action=chosen_action,
                duration_minutes=duration_m,
                num_mc_samples=10
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if chosen_action == "break":
            specific_reason = f"Mức độ mệt mỏi nhận thức cao ({current_fat:.1%}) và nồng độ CO2 phòng học tăng. Đề xuất giải lao {duration_m:.0f} phút để phục hồi sự tập trung và giảm tải nhận thức."
        elif chosen_action == "groupwork":
            specific_reason = f"Độ hiểu bài trung bình đạt {summary['mean_mastery']:.1%}, đề xuất học nhóm đồng đẳng (Vygotsky ZPD) trong {duration_m:.0f} phút để hỗ trợ nhóm sinh viên yếu (Q25: {summary['q25_mastery']:.1%})."
        elif chosen_action == "quiz":
            specific_reason = f"Độ tập trung cao ({current_att:.1%}), đề xuất bài kiểm tra ngắn {duration_m:.0f} phút để củng cố và đánh giá kịp thời mức độ lĩnh hội."
        elif chosen_action == "discussion":
            specific_reason = f"Tỷ lệ sinh viên chưa nắm bài tăng ({summary['q25_mastery']:.1%}), đề xuất thảo luận toàn lớp {duration_m:.0f} phút để gỡ bỏ điểm nghẽn kiến thức."
        else:
            specific_reason = f"Tiến trình lớp học ổn định (hiểu bài: {summary['mean_mastery']:.1%}, mệt mỏi: {current_fat:.1%}), đề xuất tiếp tục bài giảng lý thuyết {duration_m:.0f} phút."

        explanation = f"[{policy_desc}] {specific_reason}"

        rec = {
            "decision_id": decision_id,
            "class_id": class_id,
            "topic": topic,
            "recommended_action": chosen_action,
            "planned_duration_s": int(duration_m * 60),
            "evidence_level": evidence_level,
            "coverage": coverage,
            "uncertainty": uncertainty,
            "simulation_gain": rollout_stats,
            "explanation": explanation,
            "requires_teacher_decision": True,
            "latency_ms": round(elapsed_ms, 2)
        }

        self.state_store.record_decision(decision_id, rec)
        return rec
