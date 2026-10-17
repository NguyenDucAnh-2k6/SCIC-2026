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
    "break": 0.2
}

def get_valid_action_mask(
    time_remaining_minutes: float,
    current_cumulative_workload: float = 0.0,
    workload_budget: float = 15.0
) -> List[int]:
    """
    Returns boolean mask (1 for valid action, 0 for masked)
    preventing actions that exceed remaining session time or teacher workload budget.
    """
    mask = []
    for action in PEDAGOGICAL_ACTIONS:
        duration = DEFAULT_DURATIONS_MINUTES[action]
        cost = ACTION_WORKLOAD_COST[action]

        is_valid = True
        # Cannot exceed remaining class time
        if duration > time_remaining_minutes:
            is_valid = False
        # Cannot violate total teacher cognitive workload budget
        if (current_cumulative_workload + cost) > workload_budget:
            # allow only 'lecture' or 'break' as low-cost fallbacks
            if action not in ["lecture", "break"]:
                is_valid = False

        mask.append(1 if is_valid else 0)

    # Always ensure at least one action is valid
    if sum(mask) == 0:
        mask[0] = 1  # fallback to 'lecture'
    return mask
