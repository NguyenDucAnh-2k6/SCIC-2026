from typing import List, Dict, Any

PEDAGOGICAL_ACTIONS = [
    "lecture",     # idx 0: Continue standard presentation
    "quiz",        # idx 1: Brief interactive quiz (3-5 min)
    "groupwork",   # idx 2: Collaborative group activity
    "discussion",  # idx 3: Whole-class guided discussion
    "break"        # idx 4: Cognitive pause / short break (2-3 min)
]

DEFAULT_DURATIONS_MINUTES = {
    "lecture": 5.0,
    "quiz": 3.0,
    "groupwork": 5.0,
    "discussion": 5.0,
    "break": 3.0
}

ACTION_WORKLOAD_COST = {
    "lecture": 1.0,
    "quiz": 1.8,
    "groupwork": 2.0,
    "discussion": 1.5,
    "break": -1.5  # Break relieves teacher cognitive fatigue
}

def get_valid_action_mask(
    time_remaining_minutes: float = 180.0,
    current_cumulative_workload: float = 0.0
) -> List[int]:
    """
    Returns boolean action mask for discrete Q-learning.
    While session is active (time_remaining_minutes > 0), all pedagogical actions
    are valid for neural Q-network evaluation without arbitrary masking.
    """
    if time_remaining_minutes <= 0.0:
        return [0, 0, 0, 0, 0]
    return [1, 1, 1, 1, 1]
