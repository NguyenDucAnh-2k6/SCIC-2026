import numpy as np
from src.classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from src.classtwin.twin.scenarios import counterfactual_twin_rollout, MultiSourceScenarioGenerator
from src.classtwin.data.telemetry_state import build_state_vector, STATE_DIM_EXTENDED, STATE_DIM_LEGACY

def test_classroom_env_step():
    # Test extended 32-dimensional multi-source state vector
    env = ClassroomDigitalTwinEnv(cohort_size=30, episode_duration_minutes=30.0, d_state=32, seed=42)
    s0 = env.reset()
    assert s0.shape == (32,)
    assert 0.0 <= s0[0] <= 1.0  # mean mastery
    assert 0.0 <= s0[12] <= 1.0 # EDA stress
    assert 0.0 <= s0[16] <= 1.5 # CO2 norm

    next_s, reward, done, info = env.step("quiz", duration_minutes=3.0)
    assert next_s.shape == (32,)
    assert isinstance(reward, float)
    assert not done
    assert "delta_mean_mastery" in info
    assert "wearable_stress_penalty" in info

def test_legacy_12dim_compatibility():
    # Backward compatibility with 12-dimensional legacy mode
    env_legacy = ClassroomDigitalTwinEnv(cohort_size=25, episode_duration_minutes=30.0, d_state=12, seed=42)
    s_leg = env_legacy.reset()
    assert s_leg.shape == (12,)
    s_next, _, _, _ = env_legacy.step("lecture", duration_minutes=5.0)
    assert s_next.shape == (12,)

def test_missing_telemetry_imputation():
    # Real-world deployment API test: client sends only partial telemetry
    partial_telemetry = {
        "meanMastery": 0.62,
        "timeRemainingMinutes": 45.0,
        "co2": 720.0
    }
    state_vec, mask, feat_dict = build_state_vector(partial_telemetry, d_state=32, return_mask=True)
    assert state_vec.shape == (32,)
    assert mask.shape == (32,)
    assert mask[0] == 1.0   # mean_mastery was observed
    assert mask[12] == 0.0  # eda_stress was imputed with empirical prior
    assert feat_dict["mean_mastery"] == 0.62
    assert feat_dict["eda_stress_level"] > 0.0

def test_counterfactual_rollout():
    env = ClassroomDigitalTwinEnv(cohort_size=20, episode_duration_minutes=30.0, seed=42)
    env.reset()
    stats = counterfactual_twin_rollout(
        base_env=env,
        candidate_action="discussion",
        duration_minutes=5.0,
        num_mc_samples=5
    )
    assert "median_reward" in stats
    assert "p10_reward" in stats
    assert "p90_reward" in stats
    assert stats["p10_reward"] <= stats["p90_reward"]

def test_multi_source_scenario_generator():
    env = ClassroomDigitalTwinEnv(cohort_size=25, episode_duration_minutes=20.0, d_state=32, seed=42)
    gen = MultiSourceScenarioGenerator(base_env=env)
    rollouts = gen.generate_pretraining_rollouts(num_episodes=2, randomize_cohorts=True)
    assert rollouts["states"].shape[1] == 32
    assert rollouts["next_states"].shape[1] == 32
    assert len(rollouts["actions"]) == len(rollouts["rewards"])
    assert rollouts["total_transitions"] > 0
