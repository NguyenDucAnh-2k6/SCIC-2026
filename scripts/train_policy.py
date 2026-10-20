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
from src.classtwin.policies.irl import MaximumEntropyIRL, generate_expert_demonstrations
from src.classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from src.classtwin.utils.config import load_config
from src.classtwin.utils.logging import get_logger, set_seed

logger = get_logger("scripts.train_policy")

def main():
    parser = argparse.ArgumentParser(description="Train Discrete CQL pedagogical policy.")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml", help="Path to config file.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of simulated episodes to collect.")
    parser.add_argument("--train-steps", type=int, default=200, help="Number of gradient steps to run.")
    parser.add_argument("--reward-mode", type=str, choices=["explicit", "irl"], default="explicit", help="Reward formulation: 'explicit' (multi-objective penalties) or 'irl' (MaxEnt Latent Reward).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    dkt_ckpt = repo_root / "output" / "models" / "gru_dkt.pt"
    env = ClassroomDigitalTwinEnv(
        cohort_size=getattr(cfg.twin, "default_cohort_size", 53),
        use_real_data=getattr(cfg.twin, "use_real_data", True),
        class_id=getattr(cfg.twin, "class_id", 12262),
        seed=42,
        dkt_model_path=dkt_ckpt if dkt_ckpt.exists() else None,
        reward_mode=args.reward_mode
    )
    if env.dkt_model is not None:
        logger.info(f"Connected trained GRU-DKT model ({dkt_ckpt}) into ClassroomDigitalTwinEnv for Neural State Estimation.")

    irl_optimizer = None
    if args.reward_mode == "irl":
        logger.info("=== REWARD MODE: INVERSE REINFORCEMENT LEARNING (MAXENT-IRL) ===")
        logger.info("Generating expert pedagogical demonstration trajectories...")
        exp_s, exp_a = generate_expert_demonstrations(env, num_episodes=20)
        logger.info(f"Generated {len(exp_s)} expert state-action pairs.")

        # Baseline random trajectories for MaxEnt partition function
        sim_s, sim_a = [], []
        for _ in range(15):
            st = env.reset()
            d = False
            while not d:
                act = np.random.choice(PEDAGOGICAL_ACTIONS)
                dur = DEFAULT_DURATIONS_MINUTES[act]
                sim_s.append(st)
                sim_a.append(PEDAGOGICAL_ACTIONS.index(act))
                st, _, d, _ = env.step(act, duration_minutes=dur)

        irl_optimizer = MaximumEntropyIRL(d_state=cfg.policy.d_state, num_actions=len(PEDAGOGICAL_ACTIONS), lr=1e-3)
        exp_s_t = torch.tensor(exp_s, dtype=torch.float32)
        exp_a_t = torch.tensor(exp_a, dtype=torch.long)
        sim_s_t = torch.tensor(np.array(sim_s), dtype=torch.float32)
        sim_a_t = torch.tensor(np.array(sim_a), dtype=torch.long)

        logger.info("Optimizing Latent Reward Net R_psi(s, a) via Maximum Entropy...")
        bs_irl = min(64, len(exp_s), len(sim_s))
        for step in range(150):
            idx_e = np.random.choice(len(exp_s), size=bs_irl, replace=False)
            idx_s = np.random.choice(len(sim_s), size=bs_irl, replace=False)
            loss_irl = irl_optimizer.train_step(
                expert_states=exp_s_t[idx_e],
                expert_actions=exp_a_t[idx_e],
                simulated_states=sim_s_t[idx_s],
                simulated_actions=sim_a_t[idx_s]
            )
            if (step + 1) % 30 == 0:
                logger.info(f"IRL Step {step + 1}/150 - MaxEnt Loss: {loss_irl:.4f}")

        irl_save_path = Path("output/models/latent_reward_net.pt")
        irl_optimizer.save(irl_save_path)
        logger.info(f"Learned Latent Reward Net saved to {irl_save_path}")

        env.latent_reward_net = irl_optimizer
    else:
        logger.info("=== REWARD MODE: REFORMULATED EXPLICIT MULTI-OBJECTIVE ===")
        logger.info("Incorporating CO2 in-gauge penalty, knowledge stagnation penalty, and recovery bonus.")

    cql = DiscreteCQL(
        d_state=cfg.policy.d_state,
        num_actions=len(PEDAGOGICAL_ACTIONS),
        d_hidden=cfg.policy.d_hidden,
        lr=cfg.policy.learning_rate,
        cql_alpha=cfg.policy.cql_alpha
    )

    # Collect transitions using exploratory behavior policy
    buffer_s, buffer_a, buffer_r, buffer_next_s, buffer_dur, buffer_done, buffer_mask = [], [], [], [], [], [], []

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            rem_time = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
            mask = get_valid_action_mask(rem_time, env.cumulative_workload)

            # Epsilon-greedy exploration
            valid_indices = [i for i, m in enumerate(mask) if m == 1]
            act_idx = np.random.choice(valid_indices)
            act_name = PEDAGOGICAL_ACTIONS[act_idx]
            dur_m = DEFAULT_DURATIONS_MINUTES[act_name]

            next_state, reward, done, _ = env.step(act_name, duration_minutes=dur_m)

            next_rem_time = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
            next_mask = get_valid_action_mask(next_rem_time, env.cumulative_workload)

            buffer_s.append(state)
            buffer_a.append(act_idx)
            buffer_r.append(reward)
            buffer_next_s.append(next_state)
            buffer_dur.append(dur_m * 60.0)
            buffer_done.append(1.0 if done else 0.0)
            buffer_mask.append(next_mask)

            state = next_state

    logger.info(f"Collected {len(buffer_s)} transitions from {args.episodes} episodes using {args.reward_mode.upper()} reward.")

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

    save_filename = "irl_cql_policy.pt" if args.reward_mode == "irl" else "cql_policy.pt"
    save_path = Path(f"output/models/{save_filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cql.q_net.state_dict(), save_path)
    logger.info(f"Policy checkpoint saved successfully to {save_path}")

    # Post-training diagnostic check for Action Diversity & Mode Collapse
    logger.info("Conducting post-training policy health check (Action Entropy & Mode Collapse diagnostics)...")
    diag_actions = {a: 0 for a in PEDAGOGICAL_ACTIONS}
    for _ in range(5):
        eval_st = env.reset()
        eval_done = False
        last_eval_act_idx = None
        while not eval_done:
            r_t = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
            m_t = get_valid_action_mask(r_t, env.cumulative_workload)
            eval_idx, eval_act = cql.select_action(
                eval_st, m_t,
                deterministic=False,
                temperature=0.35,
                repetition_action_idx=last_eval_act_idx,
                repetition_penalty=1.2
            )
            diag_actions[eval_act] += 1
            last_eval_act_idx = eval_idx
            d_min = DEFAULT_DURATIONS_MINUTES[eval_act]
            eval_st, _, eval_done, _ = env.step(eval_act, duration_minutes=d_min)

    total_diag_steps = sum(diag_actions.values())
    probs = [c / total_diag_steps for c in diag_actions.values() if c > 0]
    action_entropy = -float(np.sum(probs * np.log2(probs))) if probs else 0.0
    max_entropy = float(np.log2(len(PEDAGOGICAL_ACTIONS)))
    norm_entropy = action_entropy / max_entropy if max_entropy > 0 else 0.0
    mode_collapse_flag = norm_entropy < 0.40

    dist_str = ", ".join(f"{k}: {v/total_diag_steps*100:.1f}%" for k, v in diag_actions.items())
    logger.info(f"Diagnostic Action Distribution (5 eps): [{dist_str}]")
    logger.info(
        f"Shannon Entropy: {action_entropy:.3f}/{max_entropy:.3f} ({norm_entropy*100:.1f}%) | "
        f"Mode Collapse Status: {'[WARNING] LIKELY COLLAPSE' if mode_collapse_flag else '[HEALTHY] BALANCED POLICY'}"
    )

if __name__ == "__main__":
    main()
