import math
from typing import Dict, List, Optional, Any
import numpy as np

def compute_as_of_features(
    history_attempts: List[Dict[str, Any]],
    current_question_id: str,
    question_skills_map: Optional[Dict[str, List[int]]] = None,
    item_difficulty_priors: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Extracts strictly as-of features up to timestamp t without future lookahead.
    """
    total_attempts = len(history_attempts)
    if total_attempts == 0:
        return {
            "history_count": 0.0,
            "smoothed_accuracy": 0.5,
            "recent_accuracy_5": 0.5,
            "mean_log_elapsed": 0.0,
            "item_prior_difficulty": item_difficulty_priors.get(current_question_id, 0.5) if item_difficulty_priors else 0.5,
            "missing_history_flag": 1.0
        }

    correct_count = sum(1 for a in history_attempts if a.get("is_correct") is True)
    # Laplace smoothing
    smoothed_acc = (correct_count + 1.0) / (total_attempts + 2.0)

    recent_5 = history_attempts[-5:]
    recent_correct = sum(1 for a in recent_5 if a.get("is_correct") is True)
    recent_acc = (recent_correct + 1.0) / (len(recent_5) + 2.0)

    elapsed_list = [
        math.log1p(max(0, float(a.get("elapsed_ms") or 0.0)))
        for a in history_attempts if a.get("elapsed_ms") is not None
    ]
    mean_elapsed = float(np.mean(elapsed_list)) if elapsed_list else 0.0

    prior_diff = item_difficulty_priors.get(current_question_id, 0.5) if item_difficulty_priors else 0.5

    return {
        "history_count": float(total_attempts),
        "smoothed_accuracy": float(smoothed_acc),
        "recent_accuracy_5": float(recent_acc),
        "mean_log_elapsed": mean_elapsed,
        "item_prior_difficulty": float(prior_diff),
        "missing_history_flag": 0.0
    }
