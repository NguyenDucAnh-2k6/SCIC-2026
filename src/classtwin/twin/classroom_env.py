import copy
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .learner_dynamics import IndividualLearnerDynamics
from classtwin.data.orchestrator import DataOrchestrator, MultiSourceDataManager
from classtwin.data.telemetry_state import (
    STATE_DIM_EXTENDED,
    STATE_DIM_LEGACY,
    FEATURE_NAMES_32,
    DEFAULT_IMPUTED_VALUES,
    build_state_vector
)
from classtwin.state_estimation.gru_dkt import GRUDKT
from classtwin.state_estimation.probe_mastery import compute_probe_mastery

class ClassroomDigitalTwinEnv:
    """
    Classroom Digital Twin Environment (M2 Simulation Level).
    Simulates a cohort of 30-100 students throughout an academic lecture session (e.g. 60-180 minutes).
    Integrated with empirical data across:
    - ASSISTments (250 real classroom cohorts, masteries & scaffolding)
    - ASSISTments Affect (dynamic Markov transitions: bored, confused, frustrated, concentrating)
    - In-Gauge (16 longitudinal classroom vi climate streams: CO2, temperature, noise)
    - Empatica E4 Wearables (student physiological signals: EDA, HR, ACC, skin temperature)
    - SocioPatterns (HighSchool2013 and Primary face-to-face contacts & friendship ties)
    - Cognitive Tutor / Bridge to Algebra (step-level transactions, error rates, latencies)
    - OULAD (LMS clickstream, VLE activity, long-term dropout risk)
    - EdNet (KT1 knowledge tracing, KT3 video attention decay, KT4 session fatigue breaks)
    """
    def __init__(
        self,
        cohort_size: int = 53,
        episode_duration_minutes: float = 180.0,
        weak_threshold: float = 0.40,
        use_real_data: bool = True,
        class_id: Optional[int] = 12262,
        seed: Optional[int] = None,
        d_state: int = STATE_DIM_EXTENDED,
        dkt_model: Optional[GRUDKT] = None,
        dkt_model_path: Optional[Path] = None,
        probe_indices: Optional[List[int]] = None,
        reward_mode: str = "explicit",
        latent_reward_net: Optional[Any] = None,
        randomize_cohort: bool = False,
        data_root: Optional[Any] = None,
        datasets_cfg: Optional[Dict[str, Any]] = None
    ):
        self.cohort_size = cohort_size
        self.episode_duration_minutes = episode_duration_minutes
        self.weak_threshold = weak_threshold
        self.use_real_data = use_real_data
        self.class_id = class_id
        self.d_state = d_state
        self.reward_mode = reward_mode
        self.latent_reward_net = latent_reward_net
        self.randomize_cohort = randomize_cohort
        self.datasets_cfg = datasets_cfg or {}
        self.data_mgr = MultiSourceDataManager(data_root=data_root, datasets_cfg=self.datasets_cfg) if use_real_data else None

        self.peer_matrix: Optional[np.ndarray] = None
        self.peer_metrics: Dict[str, float] = {"clustering_coef": 0.38, "density": 0.12, "isolated_ratio": 0.08}
        self.affect_deltas: Optional[Dict[str, Dict[str, float]]] = None
        self.oulad_priors: Dict[str, float] = {}
        self.cognitive_tutor_priors: Dict[str, float] = {}
        self.ednet_priors: Dict[str, float] = {}
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
        self.context_env = {"co2_ppm": 680.0, "temp_c": 24.2, "noise_db": 48.0}
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets classroom state with multi-source empirical cohorts or parametric distributions."""
        self.time_elapsed_minutes = 0.0
        self.last_action = None
        self.consecutive_action_count = 0
        self.total_break_minutes = 0.0
        self.action_history = []
        self.cumulative_workload = 0.0
        self.step_counter = 0

        if self.use_real_data and self.data_mgr is not None:
            # Multi-classroom random sampling for domain randomization
            cohort_data = self.data_mgr.get_real_classroom_cohort(
                class_id=self.class_id,
                randomize=self.randomize_cohort
            )
            initial_masteries = cohort_data["initial_masteries"]
            student_ids = cohort_data["student_ids"]
            self.cohort_size = len(initial_masteries)

            self.students = [
                IndividualLearnerDynamics(student_id=str(sid), initial_mastery=float(m))
                for sid, m in zip(student_ids, initial_masteries)
            ]

            # SocioPatterns HighSchool & friendship topology
            self.peer_matrix, self.peer_metrics = self.data_mgr.get_real_peer_network(
                num_nodes=self.cohort_size,
                network_type="highschool"
            )

            # ASSISTments Affect transition rates
            self.affect_deltas = self.data_mgr.get_affect_empirical_rates()["action_deltas"]

            # Multi-room In-Gauge IoT context
            if self.randomize_cohort:
                self.data_mgr.set_active_classroom_room()
            self.context_env = self.data_mgr.get_environmental_context(step_idx=0)

            # Load priors from OULAD, Cognitive Tutor & EdNet
            self.oulad_priors = self.data_mgr.get_oulad_lms_priors()
            self.cognitive_tutor_priors = self.data_mgr.get_cognitive_tutor_priors()
            self.ednet_priors = self.data_mgr.get_ednet_behavioral_priors()
        else:
            # Parametric Beta distribution fallback
            initial_masteries = np.random.beta(a=2.0, b=2.0, size=self.cohort_size)
            self.students = [
                IndividualLearnerDynamics(student_id=f"sim_u_{i:03d}", initial_mastery=float(m))
                for i, m in enumerate(initial_masteries)
            ]
            self.peer_matrix = None
            self.peer_metrics = {"clustering_coef": 0.35, "density": 0.10, "isolated_ratio": 0.05}
            self.affect_deltas = None
            self.oulad_priors = {"vle_click_intensity": 0.45, "lms_dropout_risk": 0.25, "submission_regularity": 0.75}

        # Initialize student neural probe states consistently with DKT model if available
        if self.dkt_model is not None and self.probe_indices:
            for s in self.students:
                s.hidden_state = torch.randn(1, 1, 128) * 0.05 + (s.mastery - 0.5) * 0.25
                s.mastery, s.uncertainty = compute_probe_mastery(
                    self.dkt_model, s.hidden_state, self.probe_indices, 0
                )
            self.cognitive_tutor_priors = {"step_error_rate": 0.28, "hint_intensity": 0.18, "step_latency_norm": 0.30}
            self.ednet_priors = {"video_engagement_decay_rate": 0.045, "session_fatigue_acceleration": 0.038}

        self.initial_q25 = float(np.percentile([s.mastery for s in self.students], 25))
        return self.get_state_vector()

    def clone(self) -> "ClassroomDigitalTwinEnv":
        """
        Fast lightweight copy of classroom digital twin environment for counterfactual rollouts.
        Shares immutable neural models and dataset references; clones dynamic learner states.
        """
        new_env = copy.copy(self)
        new_env.students = []
        for s in self.students:
            ns = IndividualLearnerDynamics(s.student_id, s.mastery)
            ns.attention = s.attention
            ns.fatigue = s.fatigue
            ns.eda_stress = s.eda_stress
            ns.hr_strain = s.hr_strain
            ns.acc_restlessness = s.acc_restlessness
            ns.skin_temp_norm = s.skin_temp_norm
            ns.affect_probs = dict(s.affect_probs)
            ns.step_error_rate = s.step_error_rate
            ns.hint_request_intensity = s.hint_request_intensity
            ns.step_latency_norm = s.step_latency_norm
            ns.hidden_state = s.hidden_state.clone() if s.hidden_state is not None else None
            ns.uncertainty = s.uncertainty
            ns.n_observed = s.n_observed
            new_env.students.append(ns)

        new_env.context_env = dict(self.context_env)
        new_env.consecutive_action_count = self.consecutive_action_count
        new_env.total_break_minutes = self.total_break_minutes
        new_env.action_history = list(self.action_history)
        new_env.peer_metrics = dict(self.peer_metrics)
        new_env.oulad_priors = dict(self.oulad_priors)
        new_env.cognitive_tutor_priors = dict(self.cognitive_tutor_priors)
        new_env.ednet_priors = dict(self.ednet_priors)
        return new_env

    def get_state_vector(self) -> np.ndarray:
        """
        Extracts aggregate classroom state vector x_t in R^{d_state}.
        If d_state == 32: extended multi-source representation across all 7 datasets.
        If d_state == 12: legacy 12-dimensional slice for backward compatibility.
        """
        if len(self.students) == 0:
            return np.zeros(self.d_state, dtype=np.float32)

        masteries = np.array([s.mastery for s in self.students])
        attentions = np.array([s.attention for s in self.students])
        fatigues = np.array([s.fatigue for s in self.students])

        mean_m = float(np.mean(masteries))
        q25_m = float(np.percentile(masteries, 25))
        q75_m = float(np.percentile(masteries, 75))
        var_m = float(np.var(masteries))
        rho_weak = float(np.mean(masteries < self.weak_threshold))

        mean_att = float(np.mean(attentions))
        mean_fat = float(np.mean(fatigues))

        # Video media retention (EdNet-KT3 decay)
        video_decay = self.ednet_priors.get("video_engagement_decay_rate", 0.045)
        consec_lecture_time = (self.consecutive_action_count * 5.0) if self.last_action == "lecture" else 0.0
        video_retention = float(np.clip(1.0 - video_decay * consec_lecture_time, 0.20, 1.0))

        # Affective state distribution averages
        p_frust = float(np.mean([s.affect_probs.get("frustrated", 0.1) for s in self.students]))
        p_conf = float(np.mean([s.affect_probs.get("confused", 0.12) for s in self.students]))
        p_conc = float(np.mean([s.affect_probs.get("concentrating", 0.6) for s in self.students]))
        p_bored = float(np.mean([s.affect_probs.get("bored", 0.15) for s in self.students]))

        # Physiological wearable signals (Empatica E4)
        mean_eda = float(np.mean([s.eda_stress for s in self.students]))
        mean_hr = float(np.mean([s.hr_strain for s in self.students]))
        mean_acc = float(np.mean([s.acc_restlessness for s in self.students]))
        mean_skin_temp = float(np.mean([s.skin_temp_norm for s in self.students]))

        # Environmental IoT (In-Gauge)
        co2_norm = self.context_env["co2_ppm"] / 1200.0
        temp_norm = self.context_env["temp_c"] / 35.0
        noise_norm = self.context_env["noise_db"] / 100.0

        # SocioPatterns graph metrics
        peer_clust = self.peer_metrics.get("clustering_coef", 0.38)
        if self.peer_matrix is not None and len(masteries) > 1 and np.sum(self.peer_matrix) > 0:
            diff_matrix = np.abs(masteries[:, None] - masteries[None, :])
            weighted_diff = np.sum(self.peer_matrix * diff_matrix) / (np.sum(self.peer_matrix) + 1e-5)
            friend_homo = float(np.clip(1.0 - weighted_diff, 0.05, 0.95))
        else:
            friend_homo = float(self.peer_metrics.get("density", 0.35))
        iso_ratio = self.peer_metrics.get("isolated_ratio", 0.08)

        # OULAD LMS metrics (coupled with pedagogical actions, student attention and weakness)
        base_vle = self.oulad_priors.get("vle_click_intensity", 0.45)
        action_vle_mod = {"quiz": 0.25, "groupwork": 0.15, "discussion": 0.05, "lecture": -0.05, "break": -0.20}.get(self.last_action, 0.0)
        vle_click = float(np.clip(base_vle + action_vle_mod * mean_att, 0.05, 1.0))

        base_reg = self.oulad_priors.get("submission_regularity", 0.75)
        sub_reg = float(np.clip(base_reg * (0.6 + 0.4 * mean_att) - 0.2 * mean_fat, 0.1, 1.0))

        base_risk = self.oulad_priors.get("lms_dropout_risk", 0.25)
        lms_risk = float(np.clip(base_risk * 0.5 + 0.35 * rho_weak + 0.15 * mean_fat, 0.05, 0.95))

        # Cognitive Tutor step-level metrics (Bridge to Algebra)
        mean_step_err = float(np.mean([s.step_error_rate for s in self.students]))
        mean_hint_int = float(np.mean([s.hint_request_intensity for s in self.students]))
        mean_step_lat = float(np.mean([s.step_latency_norm for s in self.students]))

        # Session progress & teacher budget
        time_remain_norm = max(0.0, (self.episode_duration_minutes - self.time_elapsed_minutes) / self.episode_duration_minutes)
        teacher_workload_norm = min(1.0, self.cumulative_workload / 20.0)

        action_map = {"lecture": 0.0, "quiz": 1.0, "groupwork": 2.0, "discussion": 3.0, "break": 4.0}
        last_act_val = action_map.get(self.last_action, -1.0) / 4.0
        consec_norm = min(1.0, self.consecutive_action_count / 5.0)

        telemetry_dict = {
            "mean_mastery": mean_m,
            "q25_mastery": q25_m,
            "q75_mastery": q75_m,
            "var_mastery": var_m,
            "rho_weak": rho_weak,
            "mean_attention": mean_att,
            "video_engagement_ratio": video_retention,
            "session_fatigue_rate": mean_fat,
            "prob_frustrated": p_frust,
            "prob_confused": p_conf,
            "prob_concentrating": p_conc,
            "prob_bored": p_bored,
            "eda_stress_level": mean_eda,
            "hr_strain": mean_hr,
            "acc_restlessness": mean_acc,
            "skin_temp_norm": mean_skin_temp,
            "co2_norm": co2_norm,
            "temp_norm": temp_norm,
            "noise_norm": noise_norm,
            "peer_clustering_coef": peer_clust,
            "friendship_homophily": friend_homo,
            "isolated_student_ratio": iso_ratio,
            "lms_vle_activity_rate": vle_click,
            "lms_submission_regularity": sub_reg,
            "lms_risk_at_dropout": lms_risk,
            "step_error_rate": mean_step_err,
            "hint_request_intensity": mean_hint_int,
            "step_latency_norm": mean_step_lat,
            "time_remaining_norm": time_remain_norm,
            "teacher_workload_norm": teacher_workload_norm,
            "last_action_norm": last_act_val,
            "consecutive_repetition_norm": consec_norm
        }

        state_vec, _, _ = build_state_vector(
            telemetry_dict,
            d_state=self.d_state,
            episode_duration_minutes=self.episode_duration_minutes
        )
        return state_vec

    def step(
        self,
        action: str,
        duration_minutes: float = 5.0,
        scenario_params: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Applies pedagogical action across all students and computes multi-objective reward.
        Leverages peer contact matrix for groupwork, empirical wearables, and environmental sensor stream.
        """
        self.step_counter += 1
        if self.use_real_data and self.data_mgr is not None:
            self.context_env = self.data_mgr.get_environmental_context(step_idx=self.step_counter)

        if self.last_action == action:
            self.consecutive_action_count += 1
        else:
            self.consecutive_action_count = 1
        self.action_history.append(action)
        if action == "break":
            self.total_break_minutes += duration_minutes

        if action == "break" and isinstance(self.context_env, dict) and "co2_ppm" in self.context_env:
            self.context_env["co2_ppm"] = max(450.0, float(self.context_env["co2_ppm"]) - 140.0 * (duration_minutes / 5.0))

        prev_m = np.mean([s.mastery for s in self.students])
        prev_q25 = np.percentile([s.mastery for s in self.students], 25)
        prev_fatigue = float(np.mean([s.fatigue for s in self.students]))
        current_masteries = np.array([s.mastery for s in self.students])

        # Step individual learner dynamics with wearable stream
        for idx, student in enumerate(self.students):
            peer_gain = 0.0
            if action in ["groupwork", "discussion"] and self.peer_matrix is not None:
                repeat_decay = 0.5 if self.consecutive_action_count >= 2 else 1.0
                weights = self.peer_matrix[idx]
                advantages = np.maximum(0.0, current_masteries - student.mastery)
                if weights.sum() > 0:
                    peer_gain = float(0.04 * repeat_decay * np.sum(weights * advantages) / (weights.sum() + 1e-5))

            wearable_sample = None
            if self.use_real_data and self.data_mgr is not None:
                wearable_sample = self.data_mgr.get_wearable_student_telemetry(idx, self.step_counter)

            student.step(
                action=action,
                duration_minutes=duration_minutes,
                scenario_params=scenario_params,
                context_vector=self.context_env,
                peer_mastery_gain=peer_gain,
                affect_deltas=self.affect_deltas,
                dkt_model=self.dkt_model,
                probe_indices=self.probe_indices,
                wearable_telemetry=wearable_sample
            )

        self.time_elapsed_minutes += duration_minutes

        new_m = np.mean([s.mastery for s in self.students])
        new_q25 = np.percentile([s.mastery for s in self.students], 25)
        mean_fatigue = float(np.mean([s.fatigue for s in self.students]))
        mean_eda_stress = float(np.mean([s.eda_stress for s in self.students]))

        delta_m = float(new_m - prev_m)
        delta_q25 = float(new_q25 - prev_q25)

        # Teacher workload cost
        workload_cost = {"lecture": 1.0, "quiz": 1.4, "groupwork": 1.7, "discussion": 1.3, "break": -0.4}.get(action, 1.0)
        self.cumulative_workload = max(0.0, self.cumulative_workload * 0.95 + workload_cost)

        switch_penalty = 0.3 if (self.last_action is not None and self.last_action != action) else 0.0

        # Monotony & Repetition Penalty (Anti-Reward Hacking)
        if self.consecutive_action_count >= 2:
            if action == "break":
                monotony_penalty = 2.0 * (self.consecutive_action_count - 1)
            else:
                monotony_penalty = 0.6 * (self.consecutive_action_count - 1)
        else:
            monotony_penalty = 0.0

        excess_break_penalty = 0.0
        if action == "break" and (self.total_break_minutes > 0.15 * self.episode_duration_minutes):
            excess_break_penalty = 1.5 * ((self.total_break_minutes - 0.15 * self.episode_duration_minutes) / 5.0)

        # 1. Cognitive Fatigue & Wearable Stress Penalty
        fatigue_penalty = 2.5 * max(0.0, mean_fatigue - 0.40) ** 2
        wearable_stress_penalty = 0.8 * max(0.0, mean_eda_stress - 0.45)

        # 2. Environmental In-Gauge CO2 Penalty
        co2_ppm = float(self.context_env.get("co2_ppm", 680.0)) if isinstance(self.context_env, dict) else 680.0
        co2_penalty = 1.2 * max(0.0, (co2_ppm - 1000.0) / 400.0)

        # 3. Knowledge Stagnation Penalty
        stagnation_penalty = 1.5 if (mean_fatigue > 0.50 and delta_m <= 0.002 and action in ["lecture", "quiz"]) else 0.0

        # 4. Cognitive Recovery Bonus
        recovery_bonus = 0.6 * max(0.0, prev_fatigue - mean_fatigue) if action == "break" else 0.0

        # 5. Curriculum Foundation Bonus
        curriculum_bonus = 0.4 if (action == "lecture" and new_m < 0.55) else 0.0

        # 6. Formative Assessment Bonus
        assessment_bonus = 0.5 if (action == "quiz" and self.last_action in ["lecture", "discussion"]) else 0.0

        # Teacher Workload Overrun Penalty
        workload_penalty = 1.0 if self.cumulative_workload > 15.0 else 0.0

        explicit_reward = (
            (10.0 * delta_m)
            + (5.0 * delta_q25)
            - (0.3 * workload_cost)
            - (0.2 * switch_penalty)
            - monotony_penalty
            - excess_break_penalty
            - fatigue_penalty
            - wearable_stress_penalty
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
            "wearable_stress_penalty": wearable_stress_penalty,
            "co2_penalty": co2_penalty,
            "stagnation_penalty": stagnation_penalty,
            "recovery_bonus": recovery_bonus,
            "explicit_reward": float(explicit_reward),
            "reward_mode": self.reward_mode
        }

        return next_state, reward, terminated, info
