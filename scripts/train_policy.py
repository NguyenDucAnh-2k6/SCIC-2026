import argparse
import sys
from pathlib import Path
import torch
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.classtwin.policies.cql import DiscreteCQL
from src.classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, get_valid_action_mask, DEFAULT_DURATIONS_MINUTES
from src.classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from src.classtwin.utils.config import load_config
from src.classtwin.utils.logging import get_logger, set_seed

logger = get_logger("scripts.train_policy")

def main():
    parser = argparse.ArgumentParser(description="Train Discrete CQL pedagogical policy.")
    parser.add_argument("--config", type=str, default="configs/policy_sim.yaml", help="Path to config file.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of simulated episodes to collect.")
    parser.add_argument("--train-steps", type=int, default=50, help="Number of gradient steps to run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    logger.info("Initializing Classroom Digital Twin simulator to collect transition buffer...")
    env = ClassroomDigitalTwinEnv(cohort_size=cfg.twin.default_cohort_size, seed=42)
    cql = DiscreteCQL(
        d_state=cfg.policy.d_state,
        num_actions=len(PEDAGOGICAL_ACTIONS),
        d_hidden=cfg.policy.d_hidden,
        lr=cfg.policy.learning_rate,
        cql_alpha=cfg.policy.cql_alpha
    )

    # 1. Collect synthetic transitions using mixed exploratory behavior policy
    buffer_s, buffer_a, buffer_r, buffer_next_s, buffer_dur, buffer_done, buffer_mask = [], [], [], [], [], [], []

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            rem_time = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
            mask = get_valid_action_mask(rem_time, env.cumulative_workload, cfg.policy.workload_budget)

            # Epsilon-greedy random exploration over valid actions
            valid_indices = [i for i, m in enumerate(mask) if m == 1]
            act_idx = np.random.choice(valid_indices)
            act_name = PEDAGOGICAL_ACTIONS[act_idx]
            dur_m = DEFAULT_DURATIONS_MINUTES[act_name]

            next_state, reward, done, _ = env.step(act_name, duration_minutes=dur_m)

            next_rem_time = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
            next_mask = get_valid_action_mask(next_rem_time, env.cumulative_workload, cfg.policy.workload_budget)

            buffer_s.append(state)
            buffer_a.append(act_idx)
            buffer_r.append(reward)
            buffer_next_s.append(next_state)
            buffer_dur.append(dur_m * 60.0)
            buffer_done.append(1.0 if done else 0.0)
            buffer_mask.append(next_mask)

            state = next_state

    logger.info(f"Collected {len(buffer_s)} transitions from {args.episodes} episodes.")

    # Convert to tensors
    states_t = torch.tensor(np.array(buffer_s), dtype=torch.float32)
    actions_t = torch.tensor(np.array(buffer_a), dtype=torch.long)
    rewards_t = torch.tensor(np.array(buffer_r), dtype=torch.float32)
    next_states_t = torch.tensor(np.array(buffer_next_s), dtype=torch.float32)
    durations_t = torch.tensor(np.array(buffer_dur), dtype=torch.float32)
    dones_t = torch.tensor(np.array(buffer_done), dtype=torch.float32)
    masks_t = torch.tensor(np.array(buffer_mask), dtype=torch.float32)

    logger.info(f"Training CQL policy for {args.train_steps} gradient steps...")
    n_samples = len(buffer_s)
    batch_size = min(64, n_samples)

    for step in range(args.train_steps):
        indices = np.random.choice(n_samples, size=batch_size, replace=False)
        metrics = cql.train_batch(
            states=states_t[indices],
            actions=actions_t[indices],
            rewards=rewards_t[indices],
            next_states=next_states_t[indices],
            durations_seconds=durations_t[indices],
            dones=dones_t[indices],
            next_action_masks=masks_t[indices]
        )
        if (step + 1) % 10 == 0:
            logger.info(f"Step {step + 1}/{args.train_steps} - Total Loss: {metrics['total_loss']:.4f}, Bellman: {metrics['bellman_loss']:.4f}, CQL: {metrics['cql_loss']:.4f}")

    save_path = Path("output/models/cql_policy.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cql.q_net.state_dict(), save_path)
    logger.info(f"Policy checkpoint saved successfully to {save_path}")

if __name__ == "__main__":
    main()
