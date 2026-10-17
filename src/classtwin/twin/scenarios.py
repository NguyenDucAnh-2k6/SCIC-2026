import copy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .classroom_env import ClassroomDigitalTwinEnv

def counterfactual_twin_rollout(
    base_env: ClassroomDigitalTwinEnv,
    candidate_action: str,
    duration_minutes: float = 5.0,
    scenarios: Optional[List[Dict[str, float]]] = None,
    num_mc_samples: int = 20
) -> Dict[str, Any]:
    """
    Implements Algorithm 2 (Counterfactual Digital Twin Rollout).
    
    Runs multi-scenario Monte-Carlo forward projections for candidate action `candidate_action`
    without modifying the true classroom state.
    
    Returns:
        {
            "median_reward": float,
            "p10_reward": float,
            "p90_reward": float,
            "mean_delta_mastery": float,
            "mean_delta_low": float,
            "candidate_action": str,
            "duration_minutes": float
        }
    """
    if scenarios is None:
        scenarios = [
            {"mastery_boost": 1.0, "fatigue_rate": 1.0},  # neutral baseline
            {"mastery_boost": 0.5, "fatigue_rate": 1.3},  # fatigue prone
            {"mastery_boost": 1.4, "fatigue_rate": 0.8}   # high receptivity
        ]

    collected_rewards = []
    collected_delta_m = []
    collected_delta_low = []

    for scenario_phi in scenarios:
        for _ in range(num_mc_samples):
            # Deepcopy base environment state for isolated counterfactual rollout
            env_sim = copy.deepcopy(base_env)
            _, reward, _, info = env_sim.step(
                action=candidate_action,
                duration_minutes=duration_minutes,
                scenario_params=scenario_phi
            )
            collected_rewards.append(reward)
            collected_delta_m.append(info["delta_mean_mastery"])
            collected_delta_low.append(info["delta_low_mastery"])

    arr_rewards = np.array(collected_rewards)
    median_rew = float(np.median(arr_rewards))
    p10_rew = float(np.percentile(arr_rewards, 10))
    p90_rew = float(np.percentile(arr_rewards, 90))

    return {
        "candidate_action": candidate_action,
        "duration_minutes": duration_minutes,
        "median_reward": median_rew,
        "p10_reward": p10_rew,
        "p90_reward": p90_rew,
        "mean_delta_mastery": float(np.mean(collected_delta_m)),
        "mean_delta_low": float(np.mean(collected_delta_low)),
        "total_rollout_samples": len(collected_rewards)
    }
