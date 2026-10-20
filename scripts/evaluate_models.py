"""
Pre-Deployment Comprehensive Evaluation Suite for ClassTwin Models:
- Neural Knowledge Tracing State Estimator (GRU-DKT)
- Offline RL Policy (Discrete CQL)
- Inverse RL Policy (MaxEnt IRL)
- Baselines: Rule-Based Heuristics & Uniform Random

Measures:
1. Knowledge Tracing: Calibration (ECE, Brier), Probe Sensitivity, Hidden Dynamics
2. Pedagogical Policies:
   - Action Mode Collapse / Reward Hacking Diagnostic (Shannon Action Entropy, Dominant Action %)
   - Learning Outcomes (Delta Mastery, Delta Q25 Equity)
   - Student Wellbeing (Mean Final Fatigue, Max Fatigue)
   - Teacher Burden (Cumulative Workload)
   - Safety Constraints Violations (% High Fatigue unaddressed, % High CO2 unaddressed)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.classtwin.state_estimation.gru_dkt import GRUDKT
from src.classtwin.state_estimation.calibration import compute_ece, compute_brier
from src.classtwin.state_estimation.probe_mastery import compute_probe_mastery
from src.classtwin.policies.cql import DiscreteCQL
from src.classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, DEFAULT_DURATIONS_MINUTES, get_valid_action_mask
from src.classtwin.policies.rules import RuleBasedPolicy
from src.classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from src.classtwin.utils.logging import get_logger

logger = get_logger("classtwin.evaluate")

def evaluate_gru_dkt(model_path: Path) -> Dict[str, Any]:
    print("\n" + "="*70)
    print(" 1. EVALUATING GRU-DKT NEURAL KNOWLEDGE ESTIMATOR")
    print("="*70)
    
    if not model_path.exists():
        print(f"[WARN] Model checkpoint not found at {model_path}. Skipping.")
        return {"status": "missing_checkpoint"}

    model = GRUDKT(num_questions=12280, d_model=64, d_hidden=128, num_layers=1, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Synthetic probe validation test
    probes = [101, 102, 103, 104, 105, 106, 107, 108]
    hidden = torch.zeros(1, 1, 128)
    initial_mastery, initial_unc = compute_probe_mastery(model, hidden, probes, n_observed=0)

    # Simulate a sequence of 10 consecutive correct answers
    q_seq = torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108, 109, 110]], dtype=torch.long)
    resp_correct = torch.ones((1, 10), dtype=torch.float32)
    el_seq = torch.full((1, 10), 15.0, dtype=torch.float32)

    logits_c, hidden_c = model(q_seq, resp_correct, el_seq)
    mastery_c, unc_c = compute_probe_mastery(model, hidden_c, probes, n_observed=10)

    # Simulate a sequence of 10 consecutive incorrect answers
    resp_incorrect = torch.zeros((1, 10), dtype=torch.float32)
    logits_inc, hidden_inc = model(q_seq, resp_incorrect, el_seq)
    mastery_inc, unc_inc = compute_probe_mastery(model, hidden_inc, probes, n_observed=10)

    # Compute mock calibration on sequence predictions
    probs_c = torch.sigmoid(logits_c.gather(2, q_seq.unsqueeze(-1)).squeeze(-1)).detach().numpy().flatten()
    labels_c = resp_correct.numpy().flatten()
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

    print(f" • Total Parameters: {results['num_parameters']:,}")
    print(f" • Baseline Mastery (0 obs): {results['initial_mastery']:.3f} (Uncertainty: {results['uncertainty_initial']:.3f})")
    print(f" • After 10 Correct Answers:  {results['post_correct_mastery']:.3f} (Uncertainty: {results['uncertainty_post_10_obs']:.3f})")
    print(f" • After 10 Incorrect Answers: {results['post_incorrect_mastery']:.3f}")
    print(f" • Dynamic Range Responsiveness (Delta): {results['mastery_responsiveness_delta']:.3f}")
    print(f" • Calibration ECE: {results['sample_ece']:.4f} | Brier Score: {results['sample_brier']:.4f}")
    return results

def compute_action_entropy(action_counts: Dict[str, int]) -> float:
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in action_counts.values() if c > 0]
    return -float(np.sum(probs * np.log2(probs)))

def evaluate_policy_rollouts(
    policy_name: str,
    policy_fn,
    num_episodes: int = 15,
    episode_duration_minutes: float = 90.0,
    dkt_model: Optional[GRUDKT] = None
) -> Dict[str, Any]:
    env = ClassroomDigitalTwinEnv(
        cohort_size=48,
        episode_duration_minutes=episode_duration_minutes,
        use_real_data=True,
        seed=42,
        dkt_model=dkt_model
    )

    action_counts = {a: 0 for a in PEDAGOGICAL_ACTIONS}
    cumulative_rewards = []
    delta_masteries = []
    delta_q25s = []
    final_fatigues = []
    max_fatigues = []
    final_workloads = []
    fatigue_violations = 0
    co2_violations = 0
    total_steps = 0

    for ep in range(num_episodes):
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

            # Safety check: student fatigue and CO2
            mean_fat = float(np.mean([s.fatigue for s in env.students]))
            co2_ppm = float(env.context_env.get("co2_ppm", 600.0))
            if mean_fat > 0.50 and action not in ["break"]:
                fatigue_violations += 1
            if co2_ppm > 1000.0 and action not in ["break"]:
                co2_violations += 1

            ep_max_fat = max(ep_max_fat, mean_fat)

            next_state, reward, done, _ = env.step(action, duration_minutes=dur)
            ep_reward += reward
            state = next_state
            total_steps += 1

        cumulative_rewards.append(ep_reward)
        final_m = np.mean([s.mastery for s in env.students])
        final_q25 = np.percentile([s.mastery for s in env.students], 25)
        delta_masteries.append(float(final_m - init_m))
        delta_q25s.append(float(final_q25 - init_q25))
        final_fatigues.append(float(np.mean([s.fatigue for s in env.students])))
        max_fatigues.append(ep_max_fat)
        final_workloads.append(float(env.cumulative_workload))

    # Calculate metrics
    entropy = compute_action_entropy(action_counts)
    max_entropy = np.log2(len(PEDAGOGICAL_ACTIONS)) # log2(5) = 2.3219 bits
    norm_entropy = entropy / max_entropy

    action_pct = {a: round(action_counts[a] / max(1, total_steps) * 100, 1) for a in PEDAGOGICAL_ACTIONS}
    dominant_action = max(action_counts, key=action_counts.get)
    dominant_pct = action_pct[dominant_action]

    # Mode collapse flag
    is_mode_collapsed = bool(dominant_pct > 65.0 or norm_entropy < 0.40)

    return {
        "policy": policy_name,
        "episodes": num_episodes,
        "total_steps": total_steps,
        "action_distribution_pct": action_pct,
        "action_entropy_bits": round(entropy, 3),
        "normalized_entropy": round(norm_entropy, 3),
        "dominant_action": f"{dominant_action} ({dominant_pct}%)",
        "is_mode_collapsed": is_mode_collapsed,
        "mean_cumulative_reward": round(float(np.mean(cumulative_rewards)), 3),
        "mean_delta_mastery": round(float(np.mean(delta_masteries)), 3),
        "mean_delta_q25": round(float(np.mean(delta_q25s)), 3),
        "mean_final_fatigue": round(float(np.mean(final_fatigues)), 3),
        "mean_max_fatigue": round(float(np.mean(max_fatigues)), 3),
        "mean_teacher_workload": round(float(np.mean(final_workloads)), 2),
        "fatigue_violation_rate": round(fatigue_violations / max(1, total_steps) * 100, 1),
        "co2_violation_rate": round(co2_violations / max(1, total_steps) * 100, 1)
    }

def main():
    print("="*70)
    print("       CLASSTWIN RIGOROUS PRE-DEPLOYMENT BENCHMARK SUITE       ")
    print("="*70)

    models_dir = repo_root / "output" / "models"
    dkt_path = models_dir / "gru_dkt.pt"
    cql_path = models_dir / "cql_policy.pt"
    irl_path = models_dir / "irl_cql_policy.pt"

    # 1. State Estimator Evaluation
    dkt_inst = None
    dkt_metrics = {}
    if dkt_path.exists():
        try:
            dkt_inst = GRUDKT(num_questions=12280, d_model=64, d_hidden=128, num_layers=1, dropout=0.0)
            dkt_inst.load_state_dict(torch.load(dkt_path, map_location="cpu"))
            dkt_inst.eval()
            dkt_metrics = evaluate_gru_dkt(dkt_path)
        except Exception as e:
            print(f"[ERR] Failed loading GRU-DKT: {e}")

    # 2. Policy Rollouts Evaluation
    print("\n" + "="*70)
    print(" 2. EVALUATING PEDAGOGICAL POLICIES & REWARD HACKING DIAGNOSTIC")
    print("="*70)
    print(" Simulating 15 episodes (90 minutes each) per policy on Digital Twin...")

    rule_policy = RuleBasedPolicy()
    
    # Random Policy
    def policy_random(state, mask, env):
        valid_indices = [i for i, m in enumerate(mask) if m == 1]
        return PEDAGOGICAL_ACTIONS[np.random.choice(valid_indices)]

    # Rule-Based Policy
    def policy_rule(state, mask, env):
        act, _ = rule_policy.select_action(
            {"mean_mastery": float(state[0]), "mean_fatigue": float(state[6]), "mean_attention": float(state[5])},
            mask
        )
        return act

    # CQL Policy
    cql_inst = None
    if cql_path.exists():
        cql_inst = DiscreteCQL(d_state=12, num_actions=len(PEDAGOGICAL_ACTIONS), d_hidden=128)
        cql_inst.q_net.load_state_dict(torch.load(cql_path, map_location="cpu"))
        cql_inst.q_net.eval()

    def policy_cql(state, mask, env):
        if cql_inst is None:
            return policy_rule(state, mask, env)
        last_idx = PEDAGOGICAL_ACTIONS.index(env.last_action) if env.last_action in PEDAGOGICAL_ACTIONS else None
        _, act = cql_inst.select_action(
            state, mask, deterministic=False, temperature=0.35, repetition_action_idx=last_idx, repetition_penalty=0.6
        )
        return act

    # IRL Policy
    irl_inst = None
    if irl_path.exists():
        irl_inst = DiscreteCQL(d_state=12, num_actions=len(PEDAGOGICAL_ACTIONS), d_hidden=128)
        irl_inst.q_net.load_state_dict(torch.load(irl_path, map_location="cpu"))
        irl_inst.q_net.eval()

    def policy_irl(state, mask, env):
        if irl_inst is None:
            return policy_rule(state, mask, env)
        last_idx = PEDAGOGICAL_ACTIONS.index(env.last_action) if env.last_action in PEDAGOGICAL_ACTIONS else None
        _, act = irl_inst.select_action(
            state, mask, deterministic=False, temperature=0.35, repetition_action_idx=last_idx, repetition_penalty=0.6
        )
        return act

    results = []
    print("\n -> Benchmarking Uniform Random Policy...")
    results.append(evaluate_policy_rollouts("Uniform Random", policy_random, num_episodes=10, dkt_model=dkt_inst))

    print(" -> Benchmarking Expert Rule-Based Policy...")
    results.append(evaluate_policy_rollouts("Expert Heuristic", policy_rule, num_episodes=10, dkt_model=dkt_inst))

    if cql_inst is not None:
        print(" -> Benchmarking Discrete CQL (Explicit Multi-Objective)...")
        results.append(evaluate_policy_rollouts("CQL (Explicit)", policy_cql, num_episodes=10, dkt_model=dkt_inst))

    if irl_inst is not None:
        print(" -> Benchmarking IRL-CQL (Latent MaxEnt Reward)...")
        results.append(evaluate_policy_rollouts("IRL-CQL (Latent)", policy_irl, num_episodes=10, dkt_model=dkt_inst))

    # Print summary table
    print("\n" + "="*85)
    print(f"{'POLICY':<18} | {'DOMINANT ACT':<18} | {'ENTROPY':<8} | {'COLLAPSE?':<10} | {'Δ MASTERY':<10} | {'FATIGUE':<8}")
    print("="*85)
    for r in results:
        collapse_str = "⚠️ YES (Hacked)" if r["is_mode_collapsed"] else "✅ HEALTHY"
        print(f"{r['policy']:<18} | {r['dominant_action']:<18} | {r['normalized_entropy']:<8.2f} | {collapse_str:<10} | {r['mean_delta_mastery']:<10.3f} | {r['mean_final_fatigue']:<8.3f}")
    print("="*85)

    print("\n[ACTION DISTRIBUTIONS]")
    for r in results:
        print(f" • {r['policy']:<18}: {r['action_distribution_pct']}")

    report_path = repo_root / "output" / "artifacts" / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "gru_dkt": dkt_metrics,
            "policies": results
        }, f, indent=2)
    print(f"\n[INFO] Comprehensive evaluation artifact written to: {report_path}")

if __name__ == "__main__":
    main()
