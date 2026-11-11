import copy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .classroom_env import ClassroomDigitalTwinEnv
from classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, get_valid_action_mask, DEFAULT_DURATIONS_MINUTES

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
    """
    if scenarios is None:
        scenarios = [
            {"mastery_boost": 1.0, "fatigue_rate": 1.0},   # neutral baseline
            {"mastery_boost": 0.5, "fatigue_rate": 1.3},   # fatigue prone
            {"mastery_boost": 1.4, "fatigue_rate": 0.8},   # high receptivity
            {"mastery_boost": 1.1, "fatigue_rate": 1.2},   # stress sensitive
            {"mastery_boost": 1.3, "fatigue_rate": 0.9}    # collaborative peer learning
        ]

    collected_rewards = []
    collected_delta_m = []
    collected_delta_low = []

    for scenario_phi in scenarios:
        for _ in range(num_mc_samples):
            env_sim = base_env.clone() if hasattr(base_env, "clone") else copy.deepcopy(base_env)
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

class MultiSourceScenarioGenerator:
    """
    Massive Multi-Scenario Trajectory Generator for Pretraining Offline RL & Foundation Policies.
    Generates diverse classroom rollouts by combining:
    - 250 real ASSISTments classroom cohorts (varying sizes 30-75 and baseline masteries)
    - 16 In-Gauge microclimate rooms (CO2 accumulation, temperature variations)
    - Empatica E4 student wearable physiological profiles
    - SocioPatterns HighSchool & Primary interaction matrices
    - Diverse teacher pedagogical behavioral styles (Expert, Exploratory, Lecture-Heavy, Break-Sensitive)
    """
    def __init__(
        self,
        base_env: ClassroomDigitalTwinEnv,
        scenarios: Optional[List[Dict[str, float]]] = None
    ):
        self.base_env = base_env
        self.scenarios = scenarios or [
            {"name": "neutral", "mastery_boost": 1.0, "fatigue_rate": 1.0},
            {"name": "fatigue_prone", "mastery_boost": 0.6, "fatigue_rate": 1.4},
            {"name": "high_receptivity", "mastery_boost": 1.4, "fatigue_rate": 0.7},
            {"name": "stress_sensitive", "mastery_boost": 0.8, "fatigue_rate": 1.25},
            {"name": "active_collaborative", "mastery_boost": 1.3, "fatigue_rate": 0.85}
        ]

        self.teacher_personas = [
            "balanced_pedagogue",   # Mixes lecture, quiz, groupwork, and timely breaks
            "exploratory_random",   # Uniform exploration over valid actions (broad state-action coverage)
            "lecture_heavy",        # Focuses on direct instruction followed by quizzes
            "collaborative_first",  # Emphasizes groupwork and discussions
            "restorative"           # Sensitive to student fatigue and stress, introduces breaks
        ]

    def _sample_action_for_persona(
        self,
        persona: str,
        env: ClassroomDigitalTwinEnv,
        valid_mask: List[int],
        state: np.ndarray
    ) -> str:
        valid_indices = [i for i, m in enumerate(valid_mask) if m == 1]
        if not valid_indices:
            return "break"

        mean_fatigue = float(state[7]) if len(state) >= 8 else 0.3
        rho_weak = float(state[4]) if len(state) >= 5 else 0.2

        if persona == "exploratory_random":
            chosen_idx = np.random.choice(valid_indices)
            return PEDAGOGICAL_ACTIONS[chosen_idx]

        if persona == "restorative" or mean_fatigue > 0.50:
            if PEDAGOGICAL_ACTIONS.index("break") in valid_indices and env.last_action != "break":
                return "break"

        if persona == "collaborative_first":
            collab_acts = [a for a in ["groupwork", "discussion"] if PEDAGOGICAL_ACTIONS.index(a) in valid_indices]
            if collab_acts and np.random.rand() < 0.65:
                return np.random.choice(collab_acts)

        if persona == "lecture_heavy":
            if PEDAGOGICAL_ACTIONS.index("lecture") in valid_indices and env.consecutive_action_count < 2:
                return "lecture"
            elif PEDAGOGICAL_ACTIONS.index("quiz") in valid_indices:
                return "quiz"

        # Balanced pedagogue
        if rho_weak > 0.35 and PEDAGOGICAL_ACTIONS.index("discussion") in valid_indices:
            return "discussion"
        if env.last_action in ["lecture", "discussion"] and PEDAGOGICAL_ACTIONS.index("quiz") in valid_indices:
            return "quiz"

        # Avoid harsh consecutive repetition
        candidates = [PEDAGOGICAL_ACTIONS[i] for i in valid_indices if PEDAGOGICAL_ACTIONS[i] != env.last_action]
        if candidates:
            return np.random.choice(candidates)
        return PEDAGOGICAL_ACTIONS[np.random.choice(valid_indices)]

    def generate_pretraining_rollouts(
        self,
        num_episodes: int = 40,
        randomize_cohorts: bool = True
    ) -> Dict[str, Any]:
        """
        Runs batch simulations generating diverse trajectories for offline RL (CQL) and Inverse RL (IRL).
        
        Returns:
            {
                "states": np.ndarray [N, d_state],
                "actions": np.ndarray [N],
                "rewards": np.ndarray [N],
                "next_states": np.ndarray [N, d_state],
                "durations_s": np.ndarray [N],
                "dones": np.ndarray [N],
                "valid_masks": np.ndarray [N, 5],
                "total_transitions": int,
                "episodes_simulated": int
            }
        """
        all_s, all_a, all_r, all_next_s, all_dur, all_dones, all_masks = [], [], [], [], [], [], []

        for ep in range(num_episodes):
            env_sim = self.base_env.clone()
            env_sim.randomize_cohort = randomize_cohorts
            scenario = self.scenarios[ep % len(self.scenarios)]
            persona = self.teacher_personas[ep % len(self.teacher_personas)]

            state = env_sim.reset()
            done = False

            while not done:
                rem_time = max(0.0, env_sim.episode_duration_minutes - env_sim.time_elapsed_minutes)
                mask = get_valid_action_mask(rem_time, env_sim.cumulative_workload)

                action_name = self._sample_action_for_persona(persona, env_sim, mask, state)
                action_idx = PEDAGOGICAL_ACTIONS.index(action_name)
                dur_m = DEFAULT_DURATIONS_MINUTES[action_name]

                next_state, reward, done, _ = env_sim.step(
                    action=action_name,
                    duration_minutes=dur_m,
                    scenario_params=scenario
                )

                next_rem_time = max(0.0, env_sim.episode_duration_minutes - env_sim.time_elapsed_minutes)
                next_mask = get_valid_action_mask(next_rem_time, env_sim.cumulative_workload)

                all_s.append(state)
                all_a.append(action_idx)
                all_r.append(reward)
                all_next_s.append(next_state)
                all_dur.append(dur_m * 60.0)
                all_dones.append(1.0 if done else 0.0)
                all_masks.append(next_mask)

                state = next_state

        return {
            "states": np.array(all_s, dtype=np.float32),
            "actions": np.array(all_a, dtype=np.int64),
            "rewards": np.array(all_r, dtype=np.float32),
            "next_states": np.array(all_next_s, dtype=np.float32),
            "durations_s": np.array(all_dur, dtype=np.float32),
            "dones": np.array(all_dones, dtype=np.float32),
            "valid_masks": np.array(all_masks, dtype=np.float32),
            "total_transitions": len(all_s),
            "episodes_simulated": num_episodes
        }
