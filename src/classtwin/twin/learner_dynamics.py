import numpy as np
import torch
from typing import Dict, Any, Optional, List
from classtwin.state_estimation.probe_mastery import compute_probe_mastery

class IndividualLearnerDynamics:
    """
    Models the latent state transition of a single student i:
    z_{i, t+1} ~ T_{theta, phi}(z_{i, t}, a_t, c_t, Delta t, context_t)
    
    Multi-modal latent state vector z_i comprises:
    - mastery: [0, 1] computed directly from trained GRU-DKT probe probability or cognitive growth
    - attention / engagement: [0, 1] with video/lecture decay dynamics
    - cognitive fatigue: [0, 1] with session accumulation and recovery
    - physiological signals (Empatica E4): EDA stress, HR strain, ACC restlessness, skin temp
    - affective state: probabilities for Concentrating, Confused, Frustrated, Bored
    - micro step-level dynamics (Cognitive Tutor): step error rate, hint intensity, latency
    - hidden_state: [1, 1, 128] PyTorch GRU recurrent latent embedding
    - uncertainty: epistemic uncertainty from probe variance & sample size
    """
    def __init__(self, student_id: str, initial_mastery: float = 0.5):
        self.student_id = student_id
        self.mastery = float(np.clip(initial_mastery, 0.0, 1.0))
        self.attention = 0.80
        self.fatigue = 0.10
        self.hidden_state: Optional[torch.Tensor] = None
        self.uncertainty: float = 0.50
        self.n_observed: int = 0

        # Physiological signals (Empatica E4)
        self.eda_stress: float = 0.20
        self.hr_strain: float = 0.30
        self.acc_restlessness: float = 0.15
        self.skin_temp_norm: float = 0.50

        # Affective state probabilities
        self.affect_probs: Dict[str, float] = {
            "concentrating": 0.65,
            "confused": 0.12,
            "frustrated": 0.08,
            "bored": 0.15
        }

        # Cognitive Tutor step-level metrics
        self.step_error_rate: float = max(0.05, 1.0 - self.mastery)
        self.hint_request_intensity: float = 0.15
        self.step_latency_norm: float = 0.25

    def step(
        self,
        action: str,
        duration_minutes: float,
        scenario_params: Optional[Dict[str, float]] = None,
        context_vector: Optional[Dict[str, float]] = None,
        peer_mastery_gain: float = 0.0,
        affect_deltas: Optional[Dict[str, Dict[str, float]]] = None,
        dkt_model: Optional[torch.nn.Module] = None,
        probe_indices: Optional[List[int]] = None,
        wearable_telemetry: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Executes one pedagogical transition step across all multi-modal state channels.
        Leverages empirical affective transition rates, physiological wearables, and physical peer contacts.
        """
        phi = scenario_params or {}
        phi_mastery_boost = phi.get("mastery_boost", 1.0)
        phi_fatigue_rate = phi.get("fatigue_rate", 1.0)

        dt_factor = duration_minutes / 5.0  # normalize relative to 5-min baseline

        # Environmental factors from In-Gauge sensors (CO2 > 1000 ppm accelerates cognitive fatigue)
        co2_ppm = context_vector.get("co2_ppm", 650.0) if context_vector else 650.0
        temp_c = context_vector.get("temp_c", 24.0) if context_vector else 24.0
        env_fatigue_mult = 1.0
        if co2_ppm > 1000.0:
            env_fatigue_mult += (co2_ppm - 1000.0) / 1000.0 * 0.55
        if temp_c > 27.0:
            env_fatigue_mult += 0.25

        # Empirical affect deltas if provided (from ASSISTments Affect dataset)
        affect_action_deltas = (affect_deltas or {}).get(action, {})
        emp_att_delta = affect_action_deltas.get("concentrating", None)

        # Cognitive Overload Threshold: khi mệt mỏi quá cao (> 0.55), não bộ suy giảm nhận thức
        fatigue_penalty_factor = max(0.05, 1.0 - (self.fatigue ** 2)) if self.fatigue > 0.55 else 1.0
        effective_attention = self.attention * (1.0 - 0.7 * (self.fatigue ** 2))

        # Khả năng tiếp thu cá nhân ngẫu nhiên
        individual_jitter = np.random.normal(0, 0.01)

        # Biological attention and fatigue evolution (EdNet KT3/KT4 grounded)
        if action == "lecture":
            d_attention = (emp_att_delta if emp_att_delta is not None else -0.05) * dt_factor
            d_fatigue = 0.045 * dt_factor * phi_fatigue_rate * env_fatigue_mult
        elif action == "quiz":
            if self.fatigue > 0.55:
                d_attention = -0.04 * dt_factor
                d_fatigue = 0.075 * dt_factor * phi_fatigue_rate * env_fatigue_mult
            else:
                d_attention = 0.03 * dt_factor * fatigue_penalty_factor
                d_fatigue = 0.055 * dt_factor * phi_fatigue_rate * env_fatigue_mult
        elif action == "groupwork":
            if self.fatigue > 0.65:
                d_attention = -0.03 * dt_factor
                d_fatigue = 0.04 * dt_factor * phi_fatigue_rate
            else:
                d_attention = 0.04 * dt_factor * fatigue_penalty_factor
                d_fatigue = 0.035 * dt_factor * phi_fatigue_rate
        elif action == "discussion":
            if self.fatigue > 0.60:
                d_attention = -0.02 * dt_factor
                d_fatigue = 0.035 * dt_factor * phi_fatigue_rate
            else:
                d_attention = 0.05 * dt_factor * fatigue_penalty_factor
                d_fatigue = 0.028 * dt_factor * phi_fatigue_rate
        elif action == "break":
            d_attention = 0.25 * dt_factor * (1.0 - self.attention * 0.5)
            d_fatigue = -0.28 * dt_factor * (0.5 + self.fatigue * 0.5)
        else:
            d_attention, d_fatigue = 0.0, 0.0

        noise_bio = np.random.normal(0, 0.01, size=2)
        self.attention = float(np.clip(self.attention + d_attention + noise_bio[0], 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue + d_fatigue + noise_bio[1], 0.0, 1.0))

        # Physiological Wearables Telemetry Update (Empatica E4)
        if wearable_telemetry is not None:
            self.eda_stress = float(np.clip(0.7 * self.eda_stress + 0.3 * wearable_telemetry["eda_stress"], 0.0, 1.0))
            self.hr_strain = float(np.clip(0.7 * self.hr_strain + 0.3 * wearable_telemetry["hr_strain"], 0.0, 1.0))
            self.acc_restlessness = float(np.clip(0.6 * self.acc_restlessness + 0.4 * wearable_telemetry["acc_restlessness"], 0.0, 1.0))
            self.skin_temp_norm = float(wearable_telemetry.get("skin_temp_norm", 0.50))
        else:
            # Mechanistic physiological evolution
            target_eda = 0.50 * self.fatigue + (0.35 if action == "quiz" else 0.10)
            self.eda_stress = float(np.clip(self.eda_stress + 0.15 * (target_eda - self.eda_stress), 0.0, 1.0))
            target_hr = 0.30 + 0.40 * self.fatigue + (0.20 if action in ["quiz", "discussion"] else 0.0)
            self.hr_strain = float(np.clip(self.hr_strain + 0.10 * (target_hr - self.hr_strain), 0.0, 1.0))
            # Restlessness increases with boredom or extreme fatigue
            boredom = self.affect_probs.get("bored", 0.15)
            target_acc = 0.15 + 0.45 * boredom + 0.30 * max(0.0, self.fatigue - 0.5)
            self.acc_restlessness = float(np.clip(self.acc_restlessness + 0.15 * (target_acc - self.acc_restlessness), 0.0, 1.0))

        # Affective Markov Transitions (ASSISTments Affect)
        for aff_state, delta in affect_action_deltas.items():
            curr_val = self.affect_probs.get(aff_state, 0.25)
            self.affect_probs[aff_state] = float(np.clip(curr_val + delta * dt_factor, 0.02, 0.95))
        # Re-normalize affect probabilities to sum to 1
        total_aff = sum(self.affect_probs.values())
        if total_aff > 0:
            for k in self.affect_probs:
                self.affect_probs[k] /= total_aff

        # Cognitive Tutor Step-Level Transactions (Bridge to Algebra)
        # Error rate is inversely proportional to current mastery and positively correlated with fatigue
        self.step_error_rate = float(np.clip((1.0 - self.mastery) * 0.7 + self.fatigue * 0.3, 0.02, 0.98))
        # Hint intensity surges if confusion/frustration is high and mastery is low
        conf_frust = self.affect_probs.get("confused", 0.1) + self.affect_probs.get("frustrated", 0.1)
        self.hint_request_intensity = float(np.clip(0.1 + 0.5 * conf_frust * (1.0 - self.mastery), 0.0, 1.0))
        # Latency slows down when fatigued or solving difficult problems
        self.step_latency_norm = float(np.clip(0.20 + 0.40 * self.fatigue + 0.25 * (1.0 - self.mastery), 0.05, 1.0))

        # NEURAL KNOWLEDGE TRACING DYNAMICS (GRU-DKT)
        if dkt_model is not None and probe_indices:
            if action != "break":
                if action == "quiz":
                    n_q = min(2, len(probe_indices))
                    q_batch = [int(x) for x in np.random.choice(probe_indices, size=n_q, replace=False)]
                elif action in ["lecture", "groupwork", "discussion"]:
                    q_batch = [int(np.random.choice(probe_indices))]
                else:
                    q_batch = []

                for q in q_batch:
                    if self.hidden_state is not None:
                        with torch.no_grad():
                            h_vec = self.hidden_state[-1].squeeze(0)
                            w_q = dkt_model.out_proj.weight[q]
                            b_q = dkt_model.out_proj.bias[q]
                            logit_q = torch.dot(h_vec, w_q) + b_q
                            p_base = float(torch.sigmoid(logit_q).item())
                    else:
                        p_base = self.mastery

                    p_actual = float(np.clip(
                        p_base * effective_attention + peer_mastery_gain + individual_jitter,
                        0.05,
                        0.95
                    ))
                    y_resp = 1.0 if np.random.rand() < p_actual else 0.0

                    q_tensor = torch.tensor([[q]], dtype=torch.long)
                    resp_tensor = torch.tensor([[y_resp]], dtype=torch.float32)
                    el_tensor = torch.tensor([[duration_minutes * 60.0 / max(1, len(q_batch))]], dtype=torch.float32)

                    with torch.no_grad():
                        _, self.hidden_state = dkt_model(q_tensor, resp_tensor, el_tensor, self.hidden_state)
                        self.n_observed += 1

            if self.hidden_state is not None:
                self.mastery, self.uncertainty = compute_probe_mastery(
                    dkt_model, self.hidden_state, probe_indices, self.n_observed
                )
        else:
            # Parametric fallback if GRU checkpoint is not mounted
            if action == "lecture":
                d_mastery = (0.018 * effective_attention + individual_jitter) * dt_factor * phi_mastery_boost
            elif action == "quiz":
                if self.fatigue > 0.55:
                    d_mastery = -0.012 * dt_factor * (1.0 + (self.fatigue - 0.55) * 2.0)
                else:
                    d_mastery = 0.032 * dt_factor * phi_mastery_boost * fatigue_penalty_factor if effective_attention > 0.3 else 0.008
            elif action == "groupwork":
                if self.fatigue > 0.65:
                    d_mastery = 0.002 * dt_factor
                else:
                    d_mastery = (0.022 + 1.2 * peer_mastery_gain + individual_jitter) * dt_factor * phi_mastery_boost * fatigue_penalty_factor
            elif action == "discussion":
                if self.fatigue > 0.60:
                    d_mastery = 0.005 * dt_factor
                else:
                    d_mastery = (0.028 + 0.6 * peer_mastery_gain) * dt_factor * phi_mastery_boost * fatigue_penalty_factor
            else:
                d_mastery = 0.0

            noise_m = np.random.normal(0, 0.01)
            self.mastery = float(np.clip(self.mastery + d_mastery + noise_m, 0.0, 1.0))

        return {
            "mastery": self.mastery,
            "attention": self.attention,
            "fatigue": self.fatigue,
            "eda_stress": self.eda_stress,
            "hr_strain": self.hr_strain,
            "acc_restlessness": self.acc_restlessness,
            "step_error_rate": self.step_error_rate,
            "uncertainty": self.uncertainty
        }
