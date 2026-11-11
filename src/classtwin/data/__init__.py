"""
ClassTwin Data Architecture:
Unified Orchestrator and specialized modules for all 7 empirical datasets:
- ednet: KT1, KT2, KT3, KT4 sequence dataloaders & multi-action behavioral priors
- assistments: 250 real classroom cohorts & skill builder baseline masteries
- assistments_affect: Dynamic Markov affective transitions (bored, confused, frustrated, concentrating)
- in_gauge_engage: 16 longitudinal classroom environmental streams & Empatica E4 wearable telemetry
- sociopatterns: HighSchool2013 & Primary face-to-face peer contact graphs & clustering
- bridge_to_algebra: Cognitive Tutor micro step-level problem solving, error rates, latencies
- oulad: University LMS clickstream, assessment regularity, and dropout risk
"""

from classtwin.data.orchestrator import DataOrchestrator, MultiSourceDataManager
from classtwin.data.telemetry_state import (
    STATE_DIM_EXTENDED,
    STATE_DIM_LEGACY,
    FEATURE_NAMES_32,
    DEFAULT_IMPUTED_VALUES,
    build_state_vector
)

__all__ = [
    "DataOrchestrator",
    "MultiSourceDataManager",
    "STATE_DIM_EXTENDED",
    "STATE_DIM_LEGACY",
    "FEATURE_NAMES_32",
    "DEFAULT_IMPUTED_VALUES",
    "build_state_vector"
]
