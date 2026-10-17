import numpy as np
from src.classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from src.classtwin.twin.scenarios import counterfactual_twin_rollout

def test_classroom_env_step():
    env = ClassroomDigitalTwinEnv(cohort_size=30, episode_duration_minutes=30.0, seed=42)
    s0 = env.reset()
    assert s0.shape == (12,)
    assert 0.0 <= s0[0] <= 1.0  # mean mastery

    next_s, reward, done, info = env.step("quiz", duration_minutes=3.0)
    assert next_s.shape == (12,)
    assert isinstance(reward, float)
    assert not done
    assert "delta_mean_mastery" in info

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
