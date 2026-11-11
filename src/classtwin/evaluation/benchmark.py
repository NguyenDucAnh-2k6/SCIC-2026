"""
ClassTwin Rigorous Pre-Deployment Benchmark Suite:
Evaluates Neural Estimator (GRU-DKT), Offline RL Policy (Discrete CQL),
Latent IRL Policy, and baseline heuristics over multi-source Digital Twin environment.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from classtwin.state_estimation.gru_dkt import GRUDKT
from classtwin.state_estimation.calibration import compute_ece, compute_brier
from classtwin.state_estimation.probe_mastery import compute_probe_mastery
from classtwin.policies.cql import DiscreteCQL
from classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, DEFAULT_DURATIONS_MINUTES, get_valid_action_mask
from classtwin.policies.rules import RuleBasedPolicy
from classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from classtwin.utils.logging import get_logger

logger = get_logger("classtwin.evaluation.benchmark")

def evaluate_gru_dkt(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        logger.warning(f"Model checkpoint not found at {model_path}. Skipping.")
        return {"status": "missing_checkpoint"}

    state_dict = torch.load(model_path, map_location="cpu")
    num_q = state_dict["q_embedding.weight"].shape[0] - 1
    model = GRUDKT(num_questions=num_q, d_model=64, d_hidden=128, num_layers=1, dropout=0.0)
    model.load_state_dict(state_dict)
    model.eval()

    probes = [101, 102, 103, 104, 105, 106, 107, 108]
    hidden = torch.zeros(1, 1, 128)
    initial_mastery, initial_unc = compute_probe_mastery(model, hidden, probes, n_observed=0)

    q_seq = torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108, 109, 110]], dtype=torch.long)
    resp_correct = torch.ones((1, 10), dtype=torch.float32)
    el_seq = torch.full((1, 10), 15.0, dtype=torch.float32)

    logits_c, hidden_c = model(q_seq, resp_correct, el_seq)
    mastery_c, unc_c = compute_probe_mastery(model, hidden_c, probes, n_observed=10)

    resp_incorrect = torch.zeros((1, 10), dtype=torch.float32)
    logits_inc, hidden_inc = model(q_seq, resp_incorrect, el_seq)
    mastery_inc, unc_inc = compute_probe_mastery(model, hidden_inc, probes, n_observed=10)

    target_logits_c = logits_c.gather(2, q_seq.unsqueeze(-1)).squeeze(-1)
    probs_c = torch.sigmoid(target_logits_c).squeeze(0).detach().numpy()
    labels_c = resp_correct.squeeze(0).numpy()
    ece_c = compute_ece(probs_c, labels_c)
    brier_c = compute_brier(probs_c, labels_c)

    results = {
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "initial_mastery": round(float(initial_mastery), 4),
        "post_correct_mastery": round(float(mastery_c), 4),
        "post_incorrect_mastery": round(float(mastery_inc), 4),
        "mastery_responsiveness_delta": round(float(mastery_c - mastery_inc), 4),
        "uncertainty_initial": round(float(initial_unc), 4),
        "uncertainty_post_10_obs": round(float(unc_c), 4),
        "sample_ece": round(float(ece_c), 4),
        "sample_brier": round(float(brier_c), 4)
    }
    return results

def evaluate_policy_rollouts(
    policy_name: str,
    policy_fn,
    num_episodes: int = 15,
    episode_duration_minutes: float = 90.0,
    dkt_model: Optional[GRUDKT] = None,
    d_state: int = 32,
    data_root: Optional[Path] = None
) -> Dict[str, Any]:
    env = ClassroomDigitalTwinEnv(
        cohort_size=48,
        episode_duration_minutes=episode_duration_minutes,
        use_real_data=True,
        d_state=d_state,
        seed=42,
        dkt_model=dkt_model,
        data_root=data_root
    )

    action_counts = {a: 0 for a in PEDAGOGICAL_ACTIONS}
    cumulative_rewards, delta_masteries, delta_q25s, final_fatigues, max_fatigues = [], [], [], [], []
    fatigue_violations, co2_violations, total_steps = 0, 0, 0

    for ep in tqdm(range(num_episodes), desc=f"Benchmark [{policy_name}]", unit="ep", leave=False):
        state = env.reset()
        done = False
        ep_reward = 0.0
        init_m = np.mean([s.mastery for s in env.students])
        init_q25 = np.percentile([s.mastery for s in env.students], 25)
        ep_max_fat = 0.0

        while not done:
            rem_time = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
            mask = get_valid_action_mask(rem_time, env.cumulative_workload)
            action = policy_fn(state, mask, env)
            action_counts[action] += 1
            dur = DEFAULT_DURATIONS_MINUTES[action]

            mean_fat = float(np.mean([s.fatigue for s in env.students]))
            co2_ppm = float(env.context_env.get("co2_ppm", 600.0))
            if mean_fat > 0.50 and action != "break":
                fatigue_violations += 1
            if co2_ppm > 1000.0 and action != "break":
                co2_violations += 1

            ep_max_fat = max(ep_max_fat, mean_fat)
            state, reward, done, _ = env.step(action, duration_minutes=dur)
            ep_reward += reward
            total_steps += 1

        final_m = np.mean([s.mastery for s in env.students])
        final_q25 = np.percentile([s.mastery for s in env.students], 25)
        cumulative_rewards.append(ep_reward)
        delta_masteries.append(final_m - init_m)
        delta_q25s.append(final_q25 - init_q25)
        final_fatigues.append(float(np.mean([s.fatigue for s in env.students])))
        max_fatigues.append(ep_max_fat)

    total_acts = sum(action_counts.values())
    probs = [c / total_acts for c in action_counts.values() if c > 0]
    entropy = -float(np.sum(probs * np.log2(probs))) if probs else 0.0
    norm_entropy = entropy / np.log2(len(PEDAGOGICAL_ACTIONS))
    dom_action = max(action_counts, key=action_counts.get)
    dom_pct = action_counts[dom_action] / total_acts

    return {
        "policy_name": policy_name,
        "mean_cumulative_reward": round(float(np.mean(cumulative_rewards)), 3),
        "mean_delta_mastery": round(float(np.mean(delta_masteries)), 4),
        "mean_delta_q25_equity": round(float(np.mean(delta_q25s)), 4),
        "mean_final_fatigue": round(float(np.mean(final_fatigues)), 3),
        "mean_max_fatigue": round(float(np.mean(max_fatigues)), 3),
        "fatigue_violation_rate": round(fatigue_violations / max(1, total_steps), 4),
        "co2_violation_rate": round(co2_violations / max(1, total_steps), 4),
        "action_entropy": round(entropy, 3),
        "normalized_entropy": round(norm_entropy, 3),
        "dominant_action": f"{dom_action} ({dom_pct*100:.1f}%)",
        "action_distribution": {k: round(v / total_acts * 100, 1) for k, v in action_counts.items()},
        "mode_collapse": bool(norm_entropy < 0.40 or dom_pct > 0.80)
    }

def run_comprehensive_evaluation(
    output_dir: Path,
    num_episodes: int = 15,
    d_state: int = 32,
    data_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes benchmark over GRU-DKT, CQL, IRL, and baseline policies.
    """
    logger.info("=== CLASSTWIN COMPREHENSIVE BENCHMARK EVALUATION ===")
    models_dir = output_dir / "models"
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dkt_path = models_dir / "gru_dkt.pt"
    dkt_results = evaluate_gru_dkt(dkt_path)

    dkt_model = None
    if dkt_path.exists():
        state_dict = torch.load(dkt_path, map_location="cpu")
        num_q = state_dict["q_embedding.weight"].shape[0] - 1
        dkt_model = GRUDKT(num_questions=num_q, d_model=64, d_hidden=128, num_layers=1, dropout=0.0)
        dkt_model.load_state_dict(state_dict)
        dkt_model.eval()

    def uniform_random_policy(state, mask, env):
        valid = [i for i, m in enumerate(mask) if m == 1]
        return PEDAGOGICAL_ACTIONS[np.random.choice(valid)]

    rule_engine = RuleBasedPolicy()
    def rule_based_policy(state, mask, env):
        act, _ = rule_engine.select_action(
            {"mean_mastery": float(state[0]), "mean_fatigue": float(state[6]), "rho_low": float(state[2]), "mean_attention": float(state[5])},
            mask
        )
        return act

    def load_cql_callable(ckpt_path: Path):
        cql = DiscreteCQL(d_state=d_state, num_actions=len(PEDAGOGICAL_ACTIONS))
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location="cpu")
            in_dim = state_dict.get("net.0.weight", torch.empty(0, 0)).shape[1] if "net.0.weight" in state_dict else None
            if in_dim == d_state:
                cql.q_net.load_state_dict(state_dict)
            else:
                logger.warning(
                    f"Checkpoint '{ckpt_path.name}' has dimension {in_dim} != current d_state {d_state}. "
                    "Skipping stale checkpoint."
                )
        last_act = [None]
        def cql_policy(state, mask, env):
            idx, act = cql.select_action(state, mask, deterministic=False, temperature=0.35, repetition_action_idx=last_act[0], repetition_penalty=1.2)
            last_act[0] = idx
            return act
        return cql_policy

    cql_callable = load_cql_callable(models_dir / "cql_policy.pt")
    irl_callable = load_cql_callable(models_dir / "irl_cql_policy.pt" if (models_dir / "irl_cql_policy.pt").exists() else models_dir / "cql_policy.pt")

    policies = {
        "Uniform Random": uniform_random_policy,
        "Expert Heuristic": rule_based_policy,
        "CQL (Explicit)": cql_callable,
        "IRL-CQL (Latent)": irl_callable
    }

    policy_reports = {}
    for name, p_fn in policies.items():
        logger.info(f"Benchmarking policy: {name} ({num_episodes} episodes)...")
        rep = evaluate_policy_rollouts(name, p_fn, num_episodes=num_episodes, d_state=d_state, dkt_model=dkt_model, data_root=data_root)
        policy_reports[name] = rep

    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ"),
        "state_estimator": dkt_results,
        "policies": policy_reports
    }
    report_file = artifacts_dir / "evaluation_report.json"
    def np_json_default(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=np_json_default)

    logger.info(f"Evaluation complete. Report saved to {report_file}")
    return full_report

