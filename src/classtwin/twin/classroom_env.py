import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .learner_dynamics import IndividualLearnerDynamics
from classtwin.data.multisource import MultiSourceDataManager
from classtwin.state_estimation.gru_dkt import GRUDKT

class ClassroomDigitalTwinEnv:
    """
    Classroom Digital Twin Environment (M2 Simulation Level).
    Simulates a cohort of 40-100 students throughout an academic lecture session (e.g. 60 minutes).
    Integrated with empirical data across ASSISTments, In-Gauge, SocioPatterns, and Affect logs,
    driven by neural state estimation from trained GRU-DKT.
    """
    def __init__(
        self,
        cohort_size: int = 53,
        episode_duration_minutes: float = 180.0,
        weak_threshold: float = 0.40,
        use_real_data: bool = True,
        class_id: Optional[int] = 12262,
        seed: Optional[int] = None,
        dkt_model: Optional[GRUDKT] = None,
        dkt_model_path: Optional[Path] = None,
        probe_indices: Optional[List[int]] = None,
        reward_mode: str = "explicit",
        latent_reward_net: Optional[Any] = None
    ):
        self.cohort_size = cohort_size
        self.episode_duration_minutes = episode_duration_minutes
        self.weak_threshold = weak_threshold
        self.use_real_data = use_real_data
        self.class_id = class_id
        self.reward_mode = reward_mode
        self.latent_reward_net = latent_reward_net
        self.data_mgr = MultiSourceDataManager() if use_real_data else None
        self.peer_matrix: Optional[np.ndarray] = None
        self.affect_deltas: Optional[Dict[str, Dict[str, float]]] = None
        self.step_counter: int = 0

        # Load trained GRU-DKT checkpoint if available
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        target_ckpt = dkt_model_path or (repo_root / "output" / "models" / "gru_dkt.pt")

        if dkt_model is not None:
            self.dkt_model = dkt_model
        elif target_ckpt.exists():
            try:
                self.dkt_model = GRUDKT(num_questions=12280, d_model=64, d_hidden=128, num_layers=1, dropout=0.0)
                self.dkt_model.load_state_dict(torch.load(target_ckpt, map_location="cpu"))
                self.dkt_model.eval()
            except Exception:
                self.dkt_model = None
        else:
            self.dkt_model = None

        self.probe_indices = probe_indices or list(range(100, 130))

        if seed is not None:
            np.random.seed(seed)

        self.students: List[IndividualLearnerDynamics] = []
        self.time_elapsed_minutes: float = 0.0
        self.last_action: Optional[str] = None
        self.consecutive_action_count: int = 0
        self.total_break_minutes: float = 0.0
        self.action_history: List[str] = []
        self.cumulative_workload: float = 0.0
        self.initial_q25: float = 0.0
        self.context_env = {"co2_ppm": 650.0, "temp_c": 24.5, "noise_db": 45.0}
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets classroom state with either empirical cohort distribution or parametric distribution."""
        self.time_elapsed_minutes = 0.0
        self.last_action = None
        self.consecutive_action_count = 0
        self.total_break_minutes = 0.0
        self.action_history = []
        self.cumulative_workload = 0.0
        self.step_counter = 0

        if self.use_real_data and self.data_mgr is not None:
            cohort_data = self.data_mgr.get_real_classroom_cohort(class_id=self.class_id)
            initial_masteries = cohort_data["initial_masteries"]
            student_ids = cohort_data["student_ids"]
            self.cohort_size = len(initial_masteries)

            self.students = [
                IndividualLearnerDynamics(student_id=str(sid), initial_mastery=float(m))
                for sid, m in zip(student_ids, initial_masteries)
            ]
            self.peer_matrix = self.data_mgr.get_real_peer_network(num_nodes=self.cohort_size)
            self.affect_deltas = self.data_mgr.get_affect_empirical_rates()["action_deltas"]
            self.context_env = self.data_mgr.get_environmental_context(step_idx=0)
        else:
            # Fallback Beta distribution
            initial_masteries = np.random.beta(a=2.0, b=2.0, size=self.cohort_size)
            self.students = [
                IndividualLearnerDynamics(student_id=f"sim_u_{i:03d}", initial_mastery=float(m))
                for i, m in enumerate(initial_masteries)
            ]
            self.peer_matrix = None
            self.affect_deltas = None

        self.initial_q25 = float(np.percentile([s.mastery for s in self.students], 25))
        return self.get_state_vector()

    def clone(self) -> "ClassroomDigitalTwinEnv":
        """
        Fast lightweight copy of classroom digital twin environment for counterfactual rollouts.
        Shares immutable neural models and dataset references; clones only dynamic learner states.
        Avoids slow Python copy.deepcopy overhead on large PyTorch models.
        """
        import copy
        new_env = copy.copy(self)
        new_env.students = []
        for s in self.students:
            ns = IndividualLearnerDynamics(s.student_id, s.mastery)
            ns.attention = s.attention
            ns.fatigue = s.fatigue
            ns.hidden_state = s.hidden_state.clone() if s.hidden_state is not None else None
            ns.uncertainty = s.uncertainty
            ns.n_observed = s.n_observed
            new_env.students.append(ns)
        new_env.context_env = dict(self.context_env)
        new_env.consecutive_action_count = self.consecutive_action_count
        new_env.total_break_minutes = self.total_break_minutes
        new_env.action_history = list(self.action_history)
        return new_env

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
        Leverages peer contact matrix for groupwork and empirical environmental sensor stream.
        """
        self.step_counter += 1
        if self.use_real_data and self.data_mgr is not None:
            self.context_env = self.data_mgr.get_environmental_context(step_idx=self.step_counter)

        # Track repetition and session break budget
        if self.last_action == action:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1
        self.action_history.append(action)
        if action == "break":
            self.total_break_minutes += duration_minutes

        if action == "break" and isinstance(self.context_env, dict) and "co2_ppm" in self.context_env:
            # During break, opening windows / ventilation reduces CO2 rapidly
            self.context_env["co2_ppm"] = max(450.0, float(self.context_env["co2_ppm"]) - 140.0 * (duration_minutes / 5.0))

        prev_m = np.mean([s.mastery for s in self.students])
        prev_q25 = np.percentile([s.mastery for s in self.students], 25)
        prev_fatigue = float(np.mean([s.fatigue for s in self.students]))
        current_masteries = np.array([s.mastery for s in self.students])

        # Step individual dynamics
        for idx, student in enumerate(self.students):
            peer_gain = 0.0
            if action in ["groupwork", "discussion"] and self.peer_matrix is not None:
                # Diminishing marginal peer gain when repeated continuously
                repeat_decay = 0.5 if self.consecutive_action_count >= 2 else 1.0
                weights = self.peer_matrix[idx]
                advantages = np.maximum(0.0, current_masteries - student.mastery)
                if weights.sum() > 0:
                    peer_gain = float(0.04 * repeat_decay * np.sum(weights * advantages) / (weights.sum() + 1e-5))

            student.step(
                action=action,
                duration_minutes=duration_minutes,
                scenario_params=scenario_params,
                context_vector=self.context_env,
                peer_mastery_gain=peer_gain,
                affect_deltas=self.affect_deltas,
                dkt_model=self.dkt_model,
                probe_indices=self.probe_indices
            )

        self.time_elapsed_minutes += duration_minutes

        new_m = np.mean([s.mastery for s in self.students])
        new_q25 = np.percentile([s.mastery for s in self.students], 25)
        mean_fatigue = float(np.mean([s.fatigue for s in self.students]))

        delta_m = float(new_m - prev_m)
        delta_q25 = float(new_q25 - prev_q25)

        # Teacher workload cost: balanced cost structure (Break relief is moderate, not a cheat code)
        workload_cost = {"lecture": 1.0, "quiz": 1.4, "groupwork": 1.7, "discussion": 1.3, "break": -0.4}.get(action, 1.0)
        self.cumulative_workload = max(0.0, self.cumulative_workload * 0.95 + workload_cost)

        # Switching penalty (only when erratic, but not penalized if transitioning sensibly)
        switch_penalty = 0.3 if (self.last_action is not None and self.last_action != action) else 0.0

        # Monotony & Repetition Penalty (Anti-Reward Hacking / Diversity Regularizer)
        # Penalizes repeating the same action 2 or more times in a row
        if self.consecutive_action_count >= 2:
            if action == "break":
                monotony_penalty = 2.0 * (self.consecutive_action_count - 1)  # Harsh penalty for consecutive breaks
            else:
                monotony_penalty = 0.6 * (self.consecutive_action_count - 1)
        else:
            monotony_penalty = 0.0

        # Opportunity cost of excessive break time (> 15% of total session duration)
        excess_break_penalty = 0.0
        if action == "break" and (self.total_break_minutes > 0.15 * self.episode_duration_minutes):
            excess_break_penalty = 1.5 * ((self.total_break_minutes - 0.15 * self.episode_duration_minutes) / 5.0)

        # 1. Cognitive Fatigue & Stress Penalty: w_s * Stress
        fatigue_penalty = 2.5 * max(0.0, mean_fatigue - 0.40) ** 2  # Phạt tăng lũy tiến khi mệt mỏi > 40%

        # 2. Environmental In-Gauge CO2 Penalty: CO2 > 1000 ppm gây buồn ngủ và giảm tập trung
        co2_ppm = float(self.context_env.get("co2_ppm", 650.0)) if isinstance(self.context_env, dict) else 650.0
        co2_penalty = 1.2 * max(0.0, (co2_ppm - 1000.0) / 400.0)

        # 3. Knowledge Stagnation Penalty: Phạt ép học sinh nghe giảng/làm bài khi đã quá mệt và độ hiểu bài không tăng
        stagnation_penalty = 1.5 if (mean_fatigue > 0.50 and delta_m <= 0.002 and action in ["lecture", "quiz"]) else 0.0

        # 4. Cognitive Recovery Bonus (balanced, not excessive)
        recovery_bonus = 0.6 * max(0.0, prev_fatigue - mean_fatigue) if action == "break" else 0.0

        # 5. Curriculum Foundation Bonus (Lecture is valuable when mastery is low/introductory)
        curriculum_bonus = 0.4 if (action == "lecture" and new_m < 0.55) else 0.0

        # 6. Formative Assessment Bonus (Quiz following lecture or discussion)
        assessment_bonus = 0.5 if (action == "quiz" and self.last_action in ["lecture", "discussion"]) else 0.0

        # Teacher Workload Overrun Penalty
        workload_penalty = 1.0 if self.cumulative_workload > 15.0 else 0.0

        # Explicit Multi-Objective Reward
        explicit_reward = (
            (10.0 * delta_m)
            + (5.0 * delta_q25)
            - (0.3 * workload_cost)
            - (0.2 * switch_penalty)
            - monotony_penalty
            - excess_break_penalty
            - fatigue_penalty
            - workload_penalty
            - co2_penalty
            - stagnation_penalty
            + recovery_bonus
            + curriculum_bonus
            + assessment_bonus
        )

        self.last_action = action

        terminated = bool(self.time_elapsed_minutes >= self.episode_duration_minutes)
        next_state = self.get_state_vector()

        # Latent Inverse RL Reward
        if self.reward_mode == "irl" and self.latent_reward_net is not None:
            action_map = {"lecture": 0, "quiz": 1, "groupwork": 2, "discussion": 3, "break": 4}
            act_idx = action_map.get(action, 0)
            reward = float(self.latent_reward_net.compute_latent_reward(next_state, act_idx))
        else:
            reward = float(explicit_reward)

        info = {
            "delta_mean_mastery": delta_m,
            "delta_low_mastery": delta_q25,
            "workload_cost": workload_cost,
            "cumulative_workload": self.cumulative_workload,
            "time_elapsed": self.time_elapsed_minutes,
            "fatigue_penalty": fatigue_penalty,
            "co2_penalty": co2_penalty,
            "stagnation_penalty": stagnation_penalty,
            "recovery_bonus": recovery_bonus,
            "explicit_reward": float(explicit_reward),
            "reward_mode": self.reward_mode
        }

        return next_state, reward, terminated, info
