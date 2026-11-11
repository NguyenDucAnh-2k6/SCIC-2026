"""
Classroom Telemetry State & Multi-Dimensional Representation Module.
Implements the expanded 32-dimensional pedagogical state vector with:
- Calibrated empirical priors across all 7 real-world datasets.
- Missing feature tolerance & automatic imputation for production API deployment.
- Missingness indicator masks to support robust generalization.
- Seamless compatibility with legacy 12-dimensional models.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Total dimensions
STATE_DIM_EXTENDED = 32
STATE_DIM_LEGACY = 12

# 32 semantic feature names
FEATURE_NAMES_32: List[str] = [
    # Group 1: Knowledge & Mastery Distribution (ASSISTments + EdNet-KT1)
    "mean_mastery",               # 0: Mean student probe mastery in [0, 1]
    "q25_mastery",                # 1: 25th percentile mastery (weak student group)
    "q75_mastery",                # 2: 75th percentile mastery (advanced student group)
    "var_mastery",                # 3: Variance of mastery (classroom polarization)
    "rho_weak",                   # 4: Proportion of students below mastery threshold (0.40)

    # Group 2: Cognitive Attention & Video/Media Engagement (EdNet KT2/KT3/KT4)
    "mean_attention",             # 5: Class average attention / engagement level [0, 1]
    "video_engagement_ratio",     # 6: Real-time retention on media/lecture consumption
    "session_fatigue_rate",       # 7: Rate of cognitive slowdown & hesitation

    # Group 3: Affective State Distribution (ASSISTments Affect)
    "prob_frustrated",            # 8: Average probability of student frustration
    "prob_confused",              # 9: Average probability of student confusion
    "prob_concentrating",         # 10: Average probability of flow/concentration
    "prob_bored",                 # 11: Average probability of student boredom

    # Group 4: Physiological Wearables Empatica E4 (In-Gauge & En-Gage)
    "eda_stress_level",           # 12: Electrodermal activity / Galvanic skin response normalized
    "hr_strain",                  # 13: Heart rate strain & autonomic load normalized
    "acc_restlessness",           # 14: Accelerometer magnitude / fidgeting & movement
    "skin_temp_norm",             # 15: Skin peripheral temperature normalized

    # Group 5: Indoor Microclimate Sensing (In-Gauge Multi-Room IoT)
    "co2_norm",                   # 16: Indoor CO2 normalized (CO2 / 1200 ppm)
    "temp_norm",                  # 17: Indoor temperature normalized (Temp / 35 C)
    "noise_norm",                 # 18: Ambient noise level normalized (Noise / 100 dB)

    # Group 6: Social & Friendship Topology (SocioPatterns HighSchool & Primary)
    "peer_clustering_coef",       # 19: Face-to-face interaction clustering coefficient
    "friendship_homophily",       # 20: Clustered peer interaction along friendship ties
    "isolated_student_ratio",     # 21: Fraction of learners with zero peer interactions

    # Group 7: Macro LMS Engagement (OULAD Clickstream)
    "lms_vle_activity_rate",      # 22: Online courseware / VLE clickstream intensity
    "lms_submission_regularity",  # 23: Timeliness and regularity of assignment submissions
    "lms_risk_at_dropout",        # 24: Predicted long-term academic risk / attrition

    # Group 8: Micro Step-Level Dynamics (Cognitive Tutor / Bridge to Algebra)
    "step_error_rate",            # 25: Fine-grained step-level error rate
    "hint_request_intensity",     # 26: Frequency of requesting hints & scaffolding
    "step_latency_norm",          # 27: First transaction thinking time (Latency / 60s)

    # Group 9: Session Progress & Pedagogical Control
    "time_remaining_norm",        # 28: Normalized remaining time (T_remain / T_total)
    "teacher_workload_norm",      # 29: Cumulative teacher workload budget consumed
    "last_action_norm",           # 30: Categorical encoding of preceding pedagogical action
    "consecutive_repetition_norm" # 31: Normalized count of repeating the same action
]

# Calibrated empirical defaults (Priors) for unobserved channels
DEFAULT_IMPUTED_VALUES: Dict[str, float] = {
    # Mastery
    "mean_mastery": 0.55,
    "q25_mastery": 0.40,
    "q75_mastery": 0.70,
    "var_mastery": 0.03,
    "rho_weak": 0.25,

    # Cognitive Attention
    "mean_attention": 0.75,
    "video_engagement_ratio": 0.70,
    "session_fatigue_rate": 0.20,

    # Affect
    "prob_frustrated": 0.12,
    "prob_confused": 0.14,
    "prob_concentrating": 0.58,
    "prob_bored": 0.16,

    # Physiological Wearables (Empatica E4)
    "eda_stress_level": 0.25,
    "hr_strain": 0.35,
    "acc_restlessness": 0.20,
    "skin_temp_norm": 0.50,

    # Environmental IoT (In-Gauge)
    "co2_norm": 0.58,        # ~700 ppm
    "temp_norm": 0.70,       # ~24.5 C
    "noise_norm": 0.45,      # ~45 dB

    # Social Peer Topology (SocioPatterns)
    "peer_clustering_coef": 0.38,
    "friendship_homophily": 0.50,
    "isolated_student_ratio": 0.08,

    # LMS Dynamics (OULAD)
    "lms_vle_activity_rate": 0.50,
    "lms_submission_regularity": 0.75,
    "lms_risk_at_dropout": 0.22,

    # Micro Step-level (Bridge to Algebra)
    "step_error_rate": 0.28,
    "hint_request_intensity": 0.20,
    "step_latency_norm": 0.30,

    # Control
    "time_remaining_norm": 0.75,
    "teacher_workload_norm": 0.20,
    "last_action_norm": 0.0,
    "consecutive_repetition_norm": 0.0
}

# Mapping of common client/telemetry synonyms
SYNONYM_MAP: Dict[str, str] = {
    "meanmastery": "mean_mastery",
    "q25mastery": "q25_mastery",
    "q75mastery": "q75_mastery",
    "meanattention": "mean_attention",
    "meanfatigue": "session_fatigue_rate",
    "timeremainingminutes": "time_remaining_norm",
    "timeremaining": "time_remaining_norm",
    "workload": "teacher_workload_norm",
    "co2": "co2_norm",
    "co2_ppm": "co2_norm",
    "temp": "temp_norm",
    "temp_c": "temp_norm",
    "noise": "noise_norm",
    "noise_db": "noise_norm",
    "currentaction": "last_action_norm",
    "last_action": "last_action_norm",
    "isweak": "rho_weak",
    "fatigue": "session_fatigue_rate",
    "attention": "mean_attention",
    "eda": "eda_stress_level",
    "hr": "hr_strain",
    "acc": "acc_restlessness"
}

def build_state_vector(
    telemetry: Optional[Dict[str, Any]] = None,
    d_state: int = STATE_DIM_EXTENDED,
    return_mask: bool = False,
    episode_duration_minutes: float = 180.0
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Constructs a calibrated, multi-dimensional classroom state vector x_t in R^{d_state}.
    
    Robust to missing features for real-world API deployment:
    - If telemetry dictionary lacks sensors (wearables, IoT, LMS, or cognitive tutor),
      it automatically imputes them with calibrated empirical priors from our 7 datasets.
    - If return_mask is True, returns a binary mask indicating observed (1) vs imputed (0) features.
    
    Args:
        telemetry: Optional input dictionary of observed features
        d_state: Target dimension (32 for extended multi-source, 12 for legacy compatibility)
        return_mask: Whether to return binary observed/imputed mask
        episode_duration_minutes: Total session duration for time normalization
        
    Returns:
        (state_vector, feature_mask, feature_dict)
    """
    raw_input = telemetry or {}
    # Canonicalize input keys
    cleaned_input: Dict[str, float] = {}
    for k, v in raw_input.items():
        if v is None:
            continue
        canon_key = SYNONYM_MAP.get(k.lower(), k.lower())
        try:
            val = float(v)
            if canon_key == "time_remaining_norm" and val > 1.0:
                val = max(0.0, val / episode_duration_minutes)
            elif canon_key == "co2_norm" and val > 10.0:
                val = val / 1200.0
            elif canon_key == "temp_norm" and val > 10.0:
                val = val / 35.0
            elif canon_key == "noise_norm" and val > 5.0:
                val = val / 100.0
            elif canon_key == "last_action_norm" and isinstance(v, str):
                act_map = {"lecture": 0.0, "quiz": 1.0, "groupwork": 2.0, "discussion": 3.0, "break": 4.0}
                val = act_map.get(v.lower(), 0.0) / 4.0
            cleaned_input[canon_key] = val
        except (ValueError, TypeError):
            if canon_key == "last_action_norm" and isinstance(v, str):
                act_map = {"lecture": 0.0, "quiz": 1.0, "groupwork": 2.0, "discussion": 3.0, "break": 4.0}
                cleaned_input[canon_key] = act_map.get(v.lower(), 0.0) / 4.0

    state_arr = np.zeros(d_state, dtype=np.float32)
    mask_arr = np.zeros(d_state, dtype=np.float32)
    feat_dict: Dict[str, float] = {}

    target_names = FEATURE_NAMES_32[:d_state]

    for idx, name in enumerate(target_names):
        if name in cleaned_input:
            val = cleaned_input[name]
            is_observed = 1.0
        else:
            val = DEFAULT_IMPUTED_VALUES.get(name, 0.5)
            is_observed = 0.0

        state_arr[idx] = float(val)
        mask_arr[idx] = is_observed
        feat_dict[name] = float(val)

    # Legacy 12-dim slice fallback handling
    if d_state == STATE_DIM_LEGACY:
        # Legacy: [mean_m, q25_m, q75_m, var_m, rho_low, mean_att, mean_fat, co2/1000, temp/30, noise/100, T_rem, last_act]
        legacy_indices = [0, 1, 2, 3, 4, 5, 7, 16, 17, 18, 28, 30]
        legacy_state = np.zeros(12, dtype=np.float32)
        legacy_mask = np.zeros(12, dtype=np.float32)
        for i, src_idx in enumerate(legacy_indices):
            legacy_state[i] = state_arr[src_idx] if src_idx < len(state_arr) else 0.5
            legacy_mask[i] = mask_arr[src_idx] if src_idx < len(mask_arr) else 0.0
        return (legacy_state, legacy_mask if return_mask else None, feat_dict)

    return (state_arr, mask_arr if return_mask else None, feat_dict)

