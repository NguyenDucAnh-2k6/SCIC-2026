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
from .state_store import ClassroomStateStore
from ..utils.logging import get_logger

logger = get_logger("classtwin.inference.engine")

class DecisionSupportEngine:
    """
    Implements Algorithm 4 (Online Inference & Pedagogical Decision Support Flow).
    Coordinates real-time ingestion, state cache, safe fallback, CQL policy, and digital twin simulation.
    """
    def __init__(
        self,
        state_store: ClassroomStateStore,
        cql_policy: Optional[DiscreteCQL] = None,
        min_coverage_threshold: float = 0.60,
        max_uncertainty_threshold: float = 0.45
    ):
        self.state_store = state_store
        self.cql_policy = cql_policy
        self.min_coverage = min_coverage_threshold
        self.max_uncertainty = max_uncertainty_threshold
        self.rule_policy = RuleBasedPolicy()

    def generate_recommendation(
        self,
        class_id: str,
        topic: str,
        time_remaining_minutes: float = 45.0,
        current_workload: float = 4.0,
        run_simulation_rollout: bool = True
    ) -> Dict[str, Any]:
        """
        Executes online recommendation pipeline.
        """
        start_time = time.perf_counter()
        decision_id = f"dec_{class_id}_{int(time.time() * 1000)}"

        # 1. Fetch class aggregate state
        summary = self.state_store.get_class_summary(class_id)
        coverage = summary["coverage"]
        uncertainty = summary["uncertainty"]

        # 2. Check Safe Fallback conditions
        if coverage < self.min_coverage or uncertainty > self.max_uncertainty:
            valid_mask = get_valid_action_mask(time_remaining_minutes, current_workload)
            action, reason = self.rule_policy.select_action(
                {"mean_mastery": summary["mean_mastery"]},
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
                "explanation": f"Chưa đủ dữ liệu quan sát tin cậy ({coverage:.1%} sinh viên). {reason}",
                "requires_teacher_decision": True,
                "latency_ms": round(elapsed_ms, 2)
            }
            self.state_store.record_decision(decision_id, rec)
            return rec

        # 3. Formulate State Vector
        state_vec = np.array([
            summary["mean_mastery"],
            summary["q25_mastery"],
            summary["q75_mastery"],
            0.02,  # variance
            max(0.0, 0.40 - summary["q25_mastery"]),  # rho_low proxy
            0.75,  # attention
            0.20,  # fatigue
            0.65,  # CO2
            0.80,  # Temp
            0.45,  # Noise
            time_remaining_minutes / 60.0,
            0.0    # last act
        ], dtype=np.float32)

        valid_mask = get_valid_action_mask(time_remaining_minutes, current_workload)

        # 4. Select Action via Policy (or Rule baseline if policy not loaded)
        if self.cql_policy:
            _, chosen_action = self.cql_policy.select_action(state_vec, valid_mask, deterministic=True)
            evidence_level = "cql_offline_policy"
        else:
            chosen_action, _ = self.rule_policy.select_action(
                {"mean_mastery": summary["mean_mastery"]},
                valid_mask
            )
            evidence_level = "rule_based"

        duration_m = DEFAULT_DURATIONS_MINUTES[chosen_action]

        # 5. Counterfactual Simulation Rollout (What-if projection)
        rollout_stats = {}
        if run_simulation_rollout:
            sim_env = ClassroomDigitalTwinEnv(cohort_size=summary["n_enrolled"], seed=42)
            sim_env.reset()
            rollout_stats = counterfactual_twin_rollout(
                base_env=sim_env,
                candidate_action=chosen_action,
                duration_minutes=duration_m,
                num_mc_samples=10
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        explanation = (
            f"Dựa trên phân phối năng lực: độ hiểu bài trung bình đạt {summary['mean_mastery']:.1%}, "
            f"nhóm 25% sinh viên yếu ở mức {summary['q25_mastery']:.1%}. "
            f"Đề xuất thực hiện '{chosen_action}' trong {duration_m:.0f} phút."
        )

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
