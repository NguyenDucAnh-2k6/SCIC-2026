from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .learner_dynamics import IndividualLearnerDynamics

class ClassroomDigitalTwinEnv:
    """
    Classroom Digital Twin Environment (M2 Simulation Level).
    Simulates a cohort of 40-100 students throughout an academic lecture session (e.g. 60 minutes).
    """
    def __init__(
        self,
        cohort_size: int = 60,
        episode_duration_minutes: float = 60.0,
        weak_threshold: float = 0.40,
        seed: Optional[int] = None
    ):
        self.cohort_size = cohort_size
        self.episode_duration_minutes = episode_duration_minutes
        self.weak_threshold = weak_threshold
        if seed is not None:
            np.random.seed(seed)

        self.students: List[IndividualLearnerDynamics] = []
        self.time_elapsed_minutes: float = 0.0
        self.last_action: Optional[str] = None
        self.cumulative_workload: float = 0.0
        self.initial_q25: float = 0.0
        self.context_env = {"co2_ppm": 650.0, "temp_c": 24.5, "noise_db": 45.0}
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets classroom state with a realistic heterogeneous student distribution."""
        self.time_elapsed_minutes = 0.0
        self.last_action = None
        self.cumulative_workload = 0.0

        # Beta distribution for diverse classroom abilities (some weak, majority average, some advanced)
        initial_masteries = np.random.beta(a=2.0, b=2.0, size=self.cohort_size)
        self.students = [
            IndividualLearnerDynamics(student_id=f"sim_u_{i:03d}", initial_mastery=float(m))
            for i, m in enumerate(initial_masteries)
        ]
        self.initial_q25 = float(np.percentile(initial_masteries, 25))
        return self.get_state_vector()

    def get_state_vector(self) -> np.ndarray:
        """
        Extracts aggregate classroom state vector x_t matching Equation (7) in proposal:
        [mean_m, q25_m, q75_m, var_m, rho_low, mean_att, mean_fat, co2/1000, temp/30, noise/100, T_remain/60, last_act_idx]
        """
        if len(self.students) == 0:
            return np.zeros(12, dtype=np.float32)
        masteries = np.array([s.mastery for s in self.students])
        attentions = np.array([s.attention for s in self.students])
        fatigues = np.array([s.fatigue for s in self.students])

        mean_m = float(np.mean(masteries))
        q25_m = float(np.percentile(masteries, 25))
        q75_m = float(np.percentile(masteries, 75))
        var_m = float(np.var(masteries))
        rho_low = float(np.mean(masteries < self.weak_threshold))

        mean_att = float(np.mean(attentions))
        mean_fat = float(np.mean(fatigues))

        co2_norm = self.context_env["co2_ppm"] / 1000.0
        temp_norm = self.context_env["temp_c"] / 30.0
        noise_norm = self.context_env["noise_db"] / 100.0

        time_remain_norm = max(0.0, (self.episode_duration_minutes - self.time_elapsed_minutes) / self.episode_duration_minutes)

        action_map = {"lecture": 0.0, "quiz": 1.0, "groupwork": 2.0, "discussion": 3.0, "break": 4.0}
        last_act_val = action_map.get(self.last_action, -1.0) / 4.0

        state = np.array([
            mean_m, q25_m, q75_m, var_m, rho_low,
            mean_att, mean_fat,
            co2_norm, temp_norm, noise_norm,
            time_remain_norm, last_act_val
        ], dtype=np.float32)

        return state

    def step(
        self,
        action: str,
        duration_minutes: float = 5.0,
        scenario_params: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Applies pedagogical action across all students and computes multi-objective reward.
        """
        prev_m = np.mean([s.mastery for s in self.students])
        prev_q25 = np.percentile([s.mastery for s in self.students], 25)

        # Step individual dynamics
        for student in self.students:
            student.step(
                action=action,
                duration_minutes=duration_minutes,
                scenario_params=scenario_params,
                context_vector=self.context_env
            )

        self.time_elapsed_minutes += duration_minutes

        new_m = np.mean([s.mastery for s in self.students])
        new_q25 = np.percentile([s.mastery for s in self.students], 25)

        delta_m = float(new_m - prev_m)
        delta_q25 = float(new_q25 - prev_q25)

        # Teacher workload cost
        workload_cost = {"lecture": 1.0, "quiz": 1.8, "groupwork": 2.0, "discussion": 1.5, "break": 0.2}.get(action, 1.0)
        self.cumulative_workload += workload_cost

        # Switch penalty
        switch_penalty = 0.5 if (self.last_action is not None and self.last_action != action) else 0.0
        self.last_action = action

        # Reward weights matching technical design: w_m=10.0, w_l=5.0, w_c=0.3, w_s=0.2
        reward = (10.0 * delta_m) + (5.0 * delta_q25) - (0.3 * workload_cost) - (0.2 * switch_penalty)

        terminated = bool(self.time_elapsed_minutes >= self.episode_duration_minutes)
        next_state = self.get_state_vector()

        info = {
            "delta_mean_mastery": delta_m,
            "delta_low_mastery": delta_q25,
            "workload_cost": workload_cost,
            "cumulative_workload": self.cumulative_workload,
            "time_elapsed": self.time_elapsed_minutes
        }

        return next_state, float(reward), terminated, info
