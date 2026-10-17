import numpy as np
from typing import Dict, Any, Optional

class IndividualLearnerDynamics:
    """
    Models the latent state transition of a single student i:
    z_{i, t+1} ~ T_{theta, phi}(z_{i, t}, a_t, c_t, Delta t, context_t)
    
    Latent state vector z_i comprises:
    - mastery: [0, 1]
    - attention / engagement: [0, 1]
    - cognitive fatigue: [0, 1]
    """
    def __init__(self, student_id: str, initial_mastery: float = 0.5):
        self.student_id = student_id
        self.mastery = float(np.clip(initial_mastery, 0.0, 1.0))
        self.attention = 0.8
        self.fatigue = 0.1

    def step(
        self,
        action: str,
        duration_minutes: float,
        scenario_params: Optional[Dict[str, float]] = None,
        context_vector: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Executes one pedagogical transition step on student's latent state.
        
        Action impacts:
        - 'lecture': moderate mastery gain (+0.02), attention decays, fatigue increases.
        - 'quiz': validates/reinforces mastery (+0.03 if attention > 0.4), high attention spike (+0.1), fatigue increases (+0.05).
        - 'groupwork': boosts engagement (+0.15), peer learning mastery (+0.04), moderate fatigue.
        - 'discussion': high engagement (+0.20), deep understanding (+0.03), low fatigue.
        - 'break': resets attention (+0.30), drains fatigue (-0.20), mastery unchanged.
        """
        phi = scenario_params or {}
        phi_mastery_boost = phi.get("mastery_boost", 1.0)
        phi_fatigue_rate = phi.get("fatigue_rate", 1.0)

        dt_factor = duration_minutes / 5.0  # normalize relative to 5-min baseline

        # Environmental factors (e.g. CO2 > 1000 ppm accelerates fatigue)
        co2_ppm = context_vector.get("co2_ppm", 600.0) if context_vector else 600.0
        env_fatigue_mult = 1.3 if co2_ppm > 1000.0 else 1.0

        if action == "lecture":
            d_mastery = 0.02 * dt_factor * phi_mastery_boost * self.attention
            d_attention = -0.05 * dt_factor
            d_fatigue = 0.04 * dt_factor * phi_fatigue_rate * env_fatigue_mult
        elif action == "quiz":
            d_mastery = 0.03 * dt_factor * phi_mastery_boost if self.attention > 0.3 else 0.005
            d_attention = 0.08
            d_fatigue = 0.06 * dt_factor * phi_fatigue_rate * env_fatigue_mult
        elif action == "groupwork":
            d_mastery = 0.04 * dt_factor * phi_mastery_boost
            d_attention = 0.12
            d_fatigue = 0.03 * dt_factor * phi_fatigue_rate
        elif action == "discussion":
            d_mastery = 0.035 * dt_factor * phi_mastery_boost
            d_attention = 0.15
            d_fatigue = 0.02 * dt_factor
        elif action == "break":
            d_mastery = 0.0
            d_attention = 0.25
            d_fatigue = -0.25 * dt_factor
        else:
            d_mastery, d_attention, d_fatigue = 0.0, 0.0, 0.0

        # Stochastic noise
        noise = np.random.normal(0, 0.01, size=3)
        self.mastery = float(np.clip(self.mastery + d_mastery + noise[0], 0.0, 1.0))
        self.attention = float(np.clip(self.attention + d_attention + noise[1], 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue + d_fatigue + noise[2], 0.0, 1.0))

        return {
            "mastery": self.mastery,
            "attention": self.attention,
            "fatigue": self.fatigue
        }
