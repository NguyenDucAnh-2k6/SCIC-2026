"""
ClassTwin Unified Data Orchestrator:
Coordinates feature extraction across all 7 heterogeneous learning & classroom telemetry datasets:
1. EdNet (KT1, KT2, KT3, KT4: Knowledge Tracing sequences & multi-action behavioral priors)
2. ASSISTments (250 real classroom cohorts, empirical baseline masteries & hint rates)
3. ASSISTments Affect (Dynamic Markov affective states: bored, confused, frustrated, concentrating)
4. In-Gauge (16 longitudinal classroom climate streams: CO2, temperature, humidity, sound, light)
5. Empatica E4 Wearables (High-frequency physiological telemetry: EDA, HR, skin temperature, ACC)
6. SocioPatterns (HighSchool2013 & Primary face-to-face peer contact graphs & clustering)
7. Cognitive Tutor / Bridge to Algebra (Fine-grained step transactions, errors, and latencies)
8. OULAD (University LMS clickstream intensity, assessment regularity, and dropout risk)

Produces the standardized 32-dimensional unified state vector x_t in R^32 with missing-field tolerance.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

from classtwin.data.assistments.extractor import ClassroomCohortExtractor
from classtwin.data.assistments_affect.extractor import AffectTransitionExtractor
from classtwin.data.in_gauge_engage.environmental import EnvironmentalSensorStreamer
from classtwin.data.in_gauge_engage.wearables import WearableTelemetryExtractor
from classtwin.data.sociopatterns.network import PeerNetworkExtractor
from classtwin.data.bridge_to_algebra.extractor import CognitiveTutorExtractor
from classtwin.data.oulad.extractor import OuladLmsExtractor
from classtwin.data.ednet.extractor import get_ednet_behavioral_priors
from classtwin.data.telemetry_state import (
    STATE_DIM_EXTENDED,
    STATE_DIM_LEGACY,
    FEATURE_NAMES_32,
    DEFAULT_IMPUTED_VALUES,
    build_state_vector
)
from classtwin.utils.logging import get_logger

logger = get_logger("classtwin.data.orchestrator")

class DataOrchestrator:
    """
    Central orchestrator unifying all 7 empirical datasets for pretraining,
    Digital Twin rollouts, and production inference with missing telemetry tolerance.
    """
    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        datasets_cfg: Optional[Dict[str, Any]] = None
    ):
        if data_root is None:
            # Search upwards for repo root containing real datasets (e.g. EdNet-KT1.zip or in_gauge_engage/longitudinal)
            curr = Path(__file__).resolve().parent
            resolved = None
            for p in [curr] + list(curr.parents):
                candidate = p / "data"
                if candidate.exists() and (
                    (candidate / "EdNet-KT1.zip").exists()
                    or (candidate / "in_gauge_engage" / "longitudinal").exists()
                    or (candidate / "shards").exists()
                ):
                    resolved = candidate
                    break
            if resolved is None:
                resolved = Path("data").resolve()
            self.data_root = resolved
        else:
            self.data_root = Path(data_root).resolve()

        self.datasets_cfg = datasets_cfg or {}

        # Resolve dataset-specific paths from YAML configuration if provided
        as_cfg = self.datasets_cfg.get("assistments", {})
        aff_cfg = self.datasets_cfg.get("assistments_affect", {})
        ige_cfg = self.datasets_cfg.get("in_gauge_engage", {})
        sp_cfg = self.datasets_cfg.get("sociopatterns", {})
        b2a_cfg = self.datasets_cfg.get("bridge_to_algebra", {})
        ou_cfg = self.datasets_cfg.get("oulad", {})

        # Initialize dedicated dataset extractors with explicit paths
        self.cohort_extractor = ClassroomCohortExtractor(
            self.data_root,
            cohort_path=as_cfg.get("cohort_path")
        )
        self.affect_extractor = AffectTransitionExtractor(
            self.data_root,
            raw_path=aff_cfg.get("raw_path"),
            cache_path=aff_cfg.get("cache_path")
        )
        self.environmental_streamer = EnvironmentalSensorStreamer(
            self.data_root,
            environmental_dir=ige_cfg.get("environmental_dir")
        )
        self.wearable_extractor = WearableTelemetryExtractor(
            self.data_root,
            wearables_dir=ige_cfg.get("wearables_dir")
        )
        self.peer_extractor = PeerNetworkExtractor(
            self.data_root,
            dir_path=sp_cfg.get("dir"),
            highschool_prox=sp_cfg.get("highschool_prox"),
            highschool_friends=sp_cfg.get("highschool_friends"),
            primary_school=sp_cfg.get("primary_school")
        )
        self.cognitive_tutor_extractor = CognitiveTutorExtractor(
            self.data_root,
            raw_path=b2a_cfg.get("raw_path"),
            cache_path=b2a_cfg.get("cache_path")
        )
        self.oulad_extractor = OuladLmsExtractor(
            self.data_root,
            dir_path=ou_cfg.get("dir"),
            cache_path=ou_cfg.get("cache_path")
        )

    # 1. ASSISTments Cohort Masteries
    def get_real_classroom_cohort(
        self,
        class_id: Optional[int] = None,
        target_size: Optional[int] = None,
        randomize: bool = False
    ) -> Dict[str, Any]:
        cohort_dict = self.cohort_extractor.extract_cohort(class_id=class_id, randomize=randomize)
        if target_size and cohort_dict.get("student_count", 0) > 0 and cohort_dict["student_count"] != target_size:
            n = cohort_dict["student_count"]
            if n > target_size:
                idx = np.random.choice(n, size=target_size, replace=False)
                cohort_dict = {
                    "class_id": cohort_dict["class_id"],
                    "student_count": target_size,
                    "student_ids": [cohort_dict["student_ids"][i] for i in idx],
                    "initial_masteries": cohort_dict["initial_masteries"][idx],
                    "hint_rates": cohort_dict["hint_rates"][idx],
                    "latencies_ms": cohort_dict["latencies_ms"][idx]
                }
        return cohort_dict

    # 2. In-Gauge Environmental Climate Stream
    def set_active_classroom_room(self, room_id: Optional[str] = None) -> str:
        return self.environmental_streamer.set_active_classroom_room(room_id=room_id)

    def get_environmental_context(self, step_idx: int = 0) -> Dict[str, float]:
        return self.environmental_streamer.get_environmental_context(step_idx=step_idx)

    def sample_environmental_reading(self, room: str = "104", elapsed_minutes: float = 0.0) -> Dict[str, float]:
        return self.environmental_streamer.sample_environmental_reading(room=room, elapsed_minutes=elapsed_minutes)

    # 3. SocioPatterns Peer Network
    def get_real_peer_network(
        self,
        num_nodes: int = 53,
        cohort_size: Optional[int] = None,
        network_type: str = "highschool"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        n = cohort_size if cohort_size is not None else num_nodes
        return self.peer_extractor.get_real_peer_network(num_nodes=n, network_type=network_type)

    def get_peer_interaction_matrix(
        self,
        cohort_size: int = 53,
        num_nodes: Optional[int] = None,
        network_type: str = "highschool"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        n = num_nodes if num_nodes is not None else cohort_size
        return self.peer_extractor.get_real_peer_network(num_nodes=n, network_type=network_type)

    # 4. ASSISTments Affect Transition Dynamics
    def get_affect_empirical_rates(self) -> Dict[str, Any]:
        return self.affect_extractor.get_affect_empirical_rates()

    # 5. Cognitive Tutor Step Dynamics
    def get_cognitive_tutor_priors(self) -> Dict[str, float]:
        return self.cognitive_tutor_extractor.get_cognitive_tutor_priors()

    # 6. OULAD LMS Activity Dynamics
    def get_oulad_lms_priors(self) -> Dict[str, float]:
        return self.oulad_extractor.get_oulad_lms_priors()

    # 7. Empatica E4 Physiological Signals
    def get_wearable_student_telemetry(
        self,
        student_idx: int = 0,
        step_idx: int = 0,
        elapsed_minutes: Optional[float] = None
    ) -> Dict[str, float]:
        step = step_idx if step_idx else (int(elapsed_minutes / 5.0) if elapsed_minutes is not None else 0)
        return self.wearable_extractor.get_wearable_student_telemetry(student_idx=student_idx, step_idx=step)

    # 8. EdNet Multi-Action Priors
    def get_ednet_behavioral_priors(self) -> Dict[str, float]:
        return get_ednet_behavioral_priors(self.data_root)

    def get_ednet_priors(self) -> Dict[str, float]:
        return get_ednet_behavioral_priors(self.data_root)

    # 9. Unified 32D State Vector Construction
    def extract_32d_state_vector(
        self,
        cohort: List[Any],
        elapsed_minutes: float = 0.0,
        room: str = "104",
        peer_matrix: Optional[np.ndarray] = None,
        telemetry_override: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts and assembles the complete 32D unified state vector x_t in R^32.
        Returns:
            (state_vector, feature_mask) where state_vector is in R^32.
        """
        masteries = [float(getattr(s, "current_mastery", s.get("current_mastery", 0.5) if isinstance(s, dict) else 0.5)) for s in cohort]
        fatigues = [float(getattr(s, "fatigue", s.get("fatigue", 0.1) if isinstance(s, dict) else 0.1)) for s in cohort]
        attentions = [float(getattr(s, "attention", s.get("attention", 0.75) if isinstance(s, dict) else 0.75)) for s in cohort]

        m_arr = np.array(masteries, dtype=np.float32)
        f_arr = np.array(fatigues, dtype=np.float32)
        a_arr = np.array(attentions, dtype=np.float32)

        raw_features: Dict[str, float] = {}

        # Group 1: Knowledge & Mastery (D1-D5)
        raw_features["mean_mastery"] = float(np.mean(m_arr))
        raw_features["q25_mastery"] = float(np.percentile(m_arr, 25))
        raw_features["q75_mastery"] = float(np.percentile(m_arr, 75))
        raw_features["var_mastery"] = float(np.var(m_arr))
        raw_features["rho_weak"] = float(np.mean(m_arr < 0.40))

        # Group 2: Cognitive Attention & Fatigue (D6-D8)
        raw_features["mean_attention"] = float(np.mean(a_arr))
        raw_features["video_engagement_ratio"] = float(np.clip(np.mean(a_arr) * 0.95, 0.0, 1.0))
        raw_features["session_fatigue_rate"] = float(np.mean(f_arr))

        # Group 3: Affect (D9-D12)
        affect_data = self.get_affect_empirical_rates()
        base_affect = affect_data.get("base_probs", {})
        raw_features["prob_frustrated"] = float(base_affect.get("frustrated", 0.389))
        raw_features["prob_confused"] = float(base_affect.get("confused", 0.045))
        raw_features["prob_concentrating"] = float(base_affect.get("concentrating", 0.682))
        raw_features["prob_bored"] = float(base_affect.get("bored", 0.257))

        # Group 4: Physiological Wearables (D13-D16)
        wearable_telem = self.get_wearable_student_telemetry(student_idx=0, elapsed_minutes=elapsed_minutes)
        raw_features["eda_stress_level"] = float(np.clip(wearable_telem.get("eda_mean", 1.2) / 5.0, 0.0, 1.0))
        raw_features["hr_strain"] = float(np.clip((wearable_telem.get("hr_mean", 75.0) - 60.0) / 60.0, 0.0, 1.0))
        raw_features["acc_restlessness"] = float(np.clip(wearable_telem.get("acc_mean", 0.2), 0.0, 1.0))
        raw_features["skin_temp_norm"] = float(np.clip((wearable_telem.get("skin_temp_mean", 33.0) - 30.0) / 10.0, 0.0, 1.0))

        # Group 5: Environmental IoT (D17-D19)
        env_readings = self.get_environmental_context(step_idx=int(elapsed_minutes / 10.0))
        raw_features["co2_norm"] = float(np.clip(env_readings.get("co2_ppm", 680.0) / 1200.0, 0.0, 1.5))
        raw_features["temp_norm"] = float(np.clip(env_readings.get("temp_c", 22.5) / 35.0, 0.0, 1.0))
        raw_features["noise_norm"] = float(np.clip(env_readings.get("noise_db", 48.0) / 100.0, 0.0, 1.0))

        # Group 6: Social Peer Topology (D20-D22)
        if peer_matrix is None:
            _, peer_metrics = self.get_real_peer_network(num_nodes=len(cohort))
        else:
            density = float(np.mean(peer_matrix > 0))
            peer_metrics = {"clustering_coef": 0.38, "density": density, "isolated_ratio": 0.08}
        raw_features["peer_clustering_coef"] = float(peer_metrics.get("clustering_coef", 0.38))
        raw_features["friendship_homophily"] = float(peer_metrics.get("density", 0.12))
        raw_features["isolated_student_ratio"] = float(peer_metrics.get("isolated_ratio", 0.08))

        # Group 7: Macro LMS Engagement (D23-D25)
        oulad_priors = self.get_oulad_lms_priors()
        raw_features["lms_vle_activity_rate"] = float(oulad_priors.get("vle_click_intensity", 0.372))
        raw_features["lms_submission_regularity"] = float(oulad_priors.get("submission_regularity", 0.842))
        raw_features["lms_risk_at_dropout"] = float(oulad_priors.get("lms_dropout_risk", 0.528))

        # Group 8: Micro Step-level Dynamics (D26-D28)
        cog_priors = self.get_cognitive_tutor_priors()
        raw_features["step_error_rate"] = float(cog_priors.get("step_error_rate", 0.1118))
        raw_features["hint_request_intensity"] = float(cog_priors.get("hint_intensity", 0.0297))
        raw_features["step_latency_norm"] = float(cog_priors.get("step_latency_norm", 0.1553))

        # Group 9: Session Control (D29-D32)
        raw_features["time_remaining_norm"] = float(np.clip(1.0 - elapsed_minutes / 180.0, 0.0, 1.0))
        raw_features["teacher_workload_norm"] = float(np.clip(elapsed_minutes * 0.12 / 15.0, 0.0, 1.0))
        raw_features["last_action_norm"] = 0.0
        raw_features["consecutive_repetition_norm"] = 0.0

        if telemetry_override:
            for k, v in telemetry_override.items():
                if v is not None:
                    raw_features[k] = float(v)

        state_vec, mask, _ = build_state_vector(raw_features, d_state=STATE_DIM_EXTENDED, return_mask=True)
        return state_vec, mask if mask is not None else np.ones(STATE_DIM_EXTENDED, dtype=np.float32)

# Drop-in alias
MultiSourceDataManager = DataOrchestrator
