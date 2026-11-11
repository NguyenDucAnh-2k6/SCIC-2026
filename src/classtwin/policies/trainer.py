"""
Offline RL (Discrete CQL & MaxEnt-IRL) Policy Training Module.
Simulates multi-source Digital Twin rollouts, caches transition replay buffers,
and optimizes Conservative Q-Learning policies with Anti-Mode-Collapse regularization.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from tqdm import tqdm

from classtwin.policies.cql import DiscreteCQL
from classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, get_valid_action_mask, DEFAULT_DURATIONS_MINUTES
from classtwin.policies.rules import RuleBasedPolicy
from classtwin.policies.irl import MaximumEntropyIRL, generate_expert_demonstrations
from classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from classtwin.utils.logging import get_logger

logger = get_logger("classtwin.policies.trainer")

def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def train_cql_policy(
    cfg: Any,
    output_dir: Path,
    episodes: int = 25,
    train_steps: int = 50,
    reward_mode: str = "explicit",
    data_root: Optional[Path] = None,
    use_cache: bool = True
) -> Tuple[Path, Dict[str, Any]]:
    """
    Trains Discrete CQL policy over multi-source Digital Twin environment.
    Supports caching simulated transition buffers for instant subsequent training.
    """
    logger.info(f"=== STAGE: DIGITAL TWIN SIMULATION & OFFLINE RL POLICY (CQL - {reward_mode.upper()}) ===")
    models_dir = output_dir / "models"
    artifacts_dir = output_dir / "artifacts"
    cache_dir = (data_root or Path("data")) / "cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    d_root = data_root or Path(getattr(cfg.data, "data_root", "data"))
    use_real = getattr(cfg.twin, "use_real_data", True)
    cid = getattr(cfg.twin, "class_id", 12262)
    dkt_ckpt = models_dir / "gru_dkt.pt"
    d_state_dim = getattr(cfg.policy, "d_state", 32)

    d_datasets = getattr(cfg.data, "datasets", {})

    env = ClassroomDigitalTwinEnv(
        cohort_size=getattr(cfg.twin, "default_cohort_size", 53),
        use_real_data=use_real,
        class_id=cid,
        d_state=d_state_dim,
        seed=42,
        dkt_model_path=dkt_ckpt if dkt_ckpt.exists() else None,
        reward_mode=reward_mode,
        randomize_cohort=True,
        data_root=d_root,
        datasets_cfg=d_datasets
    )
    if env.dkt_model is not None:
        logger.info(f"Connected trained GRU-DKT model ({dkt_ckpt}) into ClassroomDigitalTwinEnv for Neural State Estimation.")

    # 1. Optional IRL Latent Reward Net Optimization
    if reward_mode == "irl":
        logger.info("Reward Mode IRL: Generating stochastic expert pedagogical demonstration trajectories...")
        exp_s, exp_a = generate_expert_demonstrations(env, num_episodes=20)
        logger.info(f"Generated {len(exp_s)} expert state-action pairs.")

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

        irl_optimizer = MaximumEntropyIRL(d_state=d_state_dim, num_actions=len(PEDAGOGICAL_ACTIONS), lr=1e-3)
        exp_s_t = torch.tensor(exp_s, dtype=torch.float32)
        exp_a_t = torch.tensor(exp_a, dtype=torch.long)
        sim_s_t = torch.tensor(np.array(sim_s), dtype=torch.float32)
        sim_a_t = torch.tensor(np.array(sim_a), dtype=torch.long)

        logger.info("Optimizing Latent Reward Net R_psi(s, a) via Maximum Entropy...")
        bs_irl = min(64, len(exp_s), len(sim_s))
        for step in tqdm(range(100), desc="Optimizing Latent Reward Net (MaxEnt IRL)", unit="step"):
            idx_e = np.random.choice(len(exp_s), size=bs_irl, replace=False)
            idx_s = np.random.choice(len(sim_s), size=bs_irl, replace=False)
            irl_optimizer.train_step(
                expert_states=exp_s_t[idx_e],
                expert_actions=exp_a_t[idx_e],
                simulated_states=sim_s_t[idx_s],
                simulated_actions=sim_a_t[idx_s]
            )

        irl_save_path = models_dir / "latent_reward_net.pt"
        irl_optimizer.save(irl_save_path)
        logger.info(f"Learned Latent Reward Net saved to {irl_save_path}")
        env.latent_reward_net = irl_optimizer

    # 2. Check transition buffer cache for fast reload
    buffer_cache_file = cache_dir / f"rollout_transitions_{reward_mode}_dim{d_state_dim}.pt"
    loaded_from_cache = False
    
    if use_cache and buffer_cache_file.exists():
        try:
            cached_data = torch.load(buffer_cache_file)
            states_t = cached_data["states"]
            actions_t = cached_data["actions"]
            rewards_t = cached_data["rewards"]
            next_states_t = cached_data["next_states"]
            durations_t = cached_data["durations"]
            dones_t = cached_data["dones"]
            masks_t = cached_data["masks"]
            logger.info(f"[CACHE HIT] Loaded {len(states_t)} precomputed transition steps from {buffer_cache_file}!")
            loaded_from_cache = True
        except Exception as e:
            logger.warning(f"Could not load cached transition buffer: {e}. Re-simulating...")

    if not loaded_from_cache:
        logger.info(f"Simulating {episodes} classroom sessions to populate transition dataset...")
        buffer_s, buffer_a, buffer_r, buffer_next_s, buffer_dur, buffer_done, buffer_mask = [], [], [], [], [], [], []
        rule_policy = RuleBasedPolicy()

        for ep in tqdm(range(episodes), desc="Simulating Classroom Rollouts", unit="ep"):
            state = env.reset()
            done = False
            last_action_name = None
            while not done:
                rem_time = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
                mask = get_valid_action_mask(rem_time, env.cumulative_workload)

                if np.random.rand() < 0.50:
                    valid_idxs = [i for i, m in enumerate(mask) if m == 1]
                    act_idx = np.random.choice(valid_idxs)
                    act_name = PEDAGOGICAL_ACTIONS[act_idx]
                else:
                    act_name, _ = rule_policy.select_action(
                        {"mean_mastery": float(state[0]), "mean_fatigue": float(state[6]), "rho_low": float(state[2]), "mean_attention": float(state[5])},
                        mask
                    )
                    if act_name == last_action_name and np.random.rand() < 0.60:
                        other_valid = [i for i, m in enumerate(mask) if m == 1 and PEDAGOGICAL_ACTIONS[i] != act_name]
                        if other_valid:
                            act_idx = np.random.choice(other_valid)
                            act_name = PEDAGOGICAL_ACTIONS[act_idx]
                        else:
                            act_idx = PEDAGOGICAL_ACTIONS.index(act_name)
                    else:
                        act_idx = PEDAGOGICAL_ACTIONS.index(act_name)

                last_action_name = act_name
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

        states_t = torch.tensor(np.array(buffer_s), dtype=torch.float32)
        actions_t = torch.tensor(np.array(buffer_a), dtype=torch.long)
        rewards_t = torch.tensor(np.array(buffer_r), dtype=torch.float32)
        next_states_t = torch.tensor(np.array(buffer_next_s), dtype=torch.float32)
        durations_t = torch.tensor(np.array(buffer_dur), dtype=torch.float32)
        dones_t = torch.tensor(np.array(buffer_done), dtype=torch.float32)
        masks_t = torch.tensor(np.array(buffer_mask), dtype=torch.float32)

        # Save cache for next run
        try:
            torch.save({
                "states": states_t,
                "actions": actions_t,
                "rewards": rewards_t,
                "next_states": next_states_t,
                "durations": durations_t,
                "dones": dones_t,
                "masks": masks_t
            }, buffer_cache_file)
            logger.info(f"Cached {len(states_t)} transition steps to {buffer_cache_file}")
        except Exception:
            pass

    cql = DiscreteCQL(
        d_state=d_state_dim,
        num_actions=len(PEDAGOGICAL_ACTIONS),
        d_hidden=getattr(cfg.policy, "d_hidden", 128),
        lr=getattr(cfg.policy, "learning_rate", 3e-4),
        cql_alpha=getattr(cfg.policy, "cql_alpha", 1.0)
    )

    n_samples = len(states_t)
    batch_size = min(getattr(cfg.policy, "batch_size", 128), n_samples)
    final_metrics = {}

    logger.info(f"Training CQL policy for {train_steps} gradient steps (batch_size={batch_size})...")
    with tqdm(range(train_steps), desc="Training Discrete CQL Policy", unit="step") as pbar_cql:
        for step in pbar_cql:
            indices = np.random.choice(n_samples, size=batch_size, replace=False)
            final_metrics = cql.train_batch(
                states=states_t[indices],
                actions=actions_t[indices],
                rewards=rewards_t[indices],
                next_states=next_states_t[indices],
                durations_seconds=durations_t[indices],
                dones=dones_t[indices],
                next_action_masks=masks_t[indices]
            )
            pbar_cql.set_postfix({
                "loss": f"{final_metrics.get('total_loss', 0.0):.4f}",
                "bellman": f"{final_metrics.get('bellman_loss', 0.0):.4f}"
            })

    save_filename = "irl_cql_policy.pt" if reward_mode == "irl" else "cql_policy.pt"
    policy_path = models_dir / save_filename
    torch.save(cql.q_net.state_dict(), policy_path)

    # Diagnostic check for entropy & mode collapse
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
            eval_st, _, eval_done, _ = env.step(eval_act, duration_minutes=DEFAULT_DURATIONS_MINUTES[eval_act])

    total_diag_steps = sum(diag_actions.values())
    probs = [c / total_diag_steps for c in diag_actions.values() if c > 0]
    action_entropy = -float(np.sum(probs * np.log2(probs))) if probs else 0.0
    max_entropy = float(np.log2(len(PEDAGOGICAL_ACTIONS)))
    norm_entropy = action_entropy / max_entropy if max_entropy > 0 else 0.0
    mode_collapse_flag = norm_entropy < 0.40

    logger.info(f"Shannon Entropy: {action_entropy:.3f}/{max_entropy:.3f} ({norm_entropy*100:.1f}%) | Mode Collapse: {'[WARNING] COLLAPSED' if mode_collapse_flag else '[HEALTHY]'}")

    metadata = {
        "policy_type": f"Discrete-CQL-{reward_mode.upper()}",
        "reward_mode": reward_mode,
        "d_state": d_state_dim,
        "actions": PEDAGOGICAL_ACTIONS,
        "cql_alpha": getattr(cfg.policy, "cql_alpha", 1.0),
        "final_loss": final_metrics.get("total_loss", 0.0),
        "action_entropy": round(action_entropy, 3),
        "normalized_entropy": round(norm_entropy, 3),
        "mode_collapse": mode_collapse_flag,
        "checkpoint_sha256": compute_file_sha256(policy_path)
    }
    with open(artifacts_dir / "policy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Stage Complete. Policy saved to {policy_path}")
    return policy_path, metadata

