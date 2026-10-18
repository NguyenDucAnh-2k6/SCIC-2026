import argparse
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.classtwin.data.zip_reader import StreamingZipReader
from src.classtwin.data.adapters import adapt_kt1_row, extract_user_id_from_path
from src.classtwin.data.preprocessing import assign_student_split, ShardWriter
from src.classtwin.data.sequences import create_sequence_dataloader, load_sequences_from_shards
from src.classtwin.state_estimation.gru_dkt import GRUDKT
from src.classtwin.state_estimation.calibration import compute_ece, compute_brier
from src.classtwin.twin.classroom_env import ClassroomDigitalTwinEnv
from src.classtwin.twin.scenarios import counterfactual_twin_rollout
from src.classtwin.policies.action_space import PEDAGOGICAL_ACTIONS, get_valid_action_mask, DEFAULT_DURATIONS_MINUTES
from src.classtwin.policies.rules import RuleBasedPolicy
from src.classtwin.policies.cql import DiscreteCQL
from src.classtwin.utils.config import load_config
from src.classtwin.utils.logging import get_logger, set_seed

logger = get_logger("classtwin.pipeline")

def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_existing_shards(shards_dir: Path) -> Tuple[bool, int, int, int]:
    """Checks if data/shards already contains train, val, and test Parquet files."""
    train_count = len(list((shards_dir / "train").glob("*.parquet"))) if (shards_dir / "train").exists() else 0
    val_count = len(list((shards_dir / "val").glob("*.parquet"))) if (shards_dir / "val").exists() else 0
    test_count = len(list((shards_dir / "test").glob("*.parquet"))) if (shards_dir / "test").exists() else 0
    has_data = (train_count > 0 and val_count > 0)
    return has_data, train_count, val_count, test_count

def run_stage_preprocessing(cfg, max_members=None, force=False) -> Path:
    logger.info("=== STAGE 1: DATA PREPROCESSING (ZIP -> PARQUET) ===")
    shards_dir = Path(cfg.data.shards_dir)
    manifest_path = shards_dir / "manifest.json"

    if manifest_path.exists() and not force:
        logger.info(f"Existing preprocessing manifest found at {manifest_path}. Skipping preprocessing.")
        return manifest_path

    shards_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(output_dir=str(shards_dir), shard_size=cfg.data.chunk_size_rows)

    start_t = time.perf_counter()
    with StreamingZipReader(cfg.data.raw_zip_path) as reader:
        members = reader.list_members(prefix="KT1/", max_members=max_members)
        total_m = len(members)
        logger.info(f"Processing {total_m} member archives from {cfg.data.raw_zip_path}...")

        for i, member_name in enumerate(members):
            student_id = extract_user_id_from_path(member_name)
            split = assign_student_split(student_id, seed=cfg.data.seed)
            rows = reader.read_csv_member(member_name)

            for row_idx, row in enumerate(rows):
                attempt, event = adapt_kt1_row(row, student_id=student_id, source_row=row_idx)
                if attempt:
                    writer.add_attempt(attempt.model_dump(), split=split)

            if (i + 1) % 1000 == 0 or (i + 1) == total_m:
                logger.info(f"Preprocessed {i + 1}/{total_m} members | Total records: {writer.total_records:,}")

    manifest = writer.close()
    logger.info(f"Stage 1 Complete in {time.perf_counter() - start_t:.1f}s. Manifest: {manifest}")
    return manifest_path

def run_stage_estimator(
    cfg,
    output_dir: Path,
    epochs: int = 5,
    shards_dir: Optional[Path] = None,
    num_shards: Optional[int] = None,
    max_students: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Tuple[Path, Dict[str, Any]]:
    logger.info("=== STAGE 2: STATE ESTIMATOR TRAINING & CALIBRATION (GRU-DKT) ===")
    models_dir = output_dir / "models"
    artifacts_dir = output_dir / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bs = batch_size or cfg.estimator.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 1. Load real data from Parquet shards if available
    train_shards = list((shards_dir / "train").glob("*.parquet")) if shards_dir and (shards_dir / "train").exists() else []

    if train_shards:
        logger.info(f"Loading real interaction tracks from Parquet shards in {shards_dir}/train...")
        train_dataset, q_to_idx, stats_train = load_sequences_from_shards(
            shards_dir=str(shards_dir),
            split="train",
            num_shards=num_shards,
            max_students=max_students
        )

        val_shards = list((shards_dir / "val").glob("*.parquet"))
        val_dataset = None
        if val_shards:
            val_dataset, _, _ = load_sequences_from_shards(
                shards_dir=str(shards_dir),
                split="val",
                num_shards=min(20, len(val_shards)),
                q_to_idx=q_to_idx,
                max_students=max_students // 5 if max_students else None
            )
    else:
        logger.warning("No Parquet shards found. Using synthetic sequences for testing...")
        train_seqs = []
        for s_idx in range(50):
            seq = [{"student_id": f"u{s_idx}", "question_id": f"q{t % 25}", "is_correct": (t % 4 != 0), "elapsed_ms": 12000 + (t % 10) * 800} for t in range(40)]
            train_seqs.append(seq)
        q_to_idx = {f"q{i}": i + 1 for i in range(30)}
        train_dataset = train_seqs
        val_dataset = train_seqs[:10]

    # Save question vocabulary artifact
    vocab_path = artifacts_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(q_to_idx, f, indent=2)
    logger.info(f"Saved question vocabulary ({len(q_to_idx):,} questions) to {vocab_path}")

    train_loader = create_sequence_dataloader(train_dataset, q_to_idx=q_to_idx, batch_size=bs, shuffle=True)
    val_loader = create_sequence_dataloader(val_dataset, q_to_idx=q_to_idx, batch_size=bs, shuffle=False) if val_dataset else None

    num_questions = len(q_to_idx)
    model = GRUDKT(
        num_questions=num_questions,
        d_model=cfg.estimator.d_model,
        d_hidden=cfg.estimator.d_hidden,
        dropout=cfg.estimator.dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.estimator.learning_rate, weight_decay=cfg.estimator.weight_decay)

    total_batches = len(train_loader)
    log_interval = max(20, total_batches // 10)
    logger.info(f"Training GRU-DKT ({num_questions} questions) for {epochs} epochs on {device} ({total_batches:,} batches/epoch, bs={bs})...")

    for ep in range(epochs):
        model.train()
        total_loss, total_tokens = 0.0, 0
        ep_start = time.perf_counter()

        for batch_idx, (q_t, resp_t, el_t, target_t, mask_t) in enumerate(train_loader):
            q_t = q_t.to(device)
            resp_t = resp_t.to(device)
            el_t = el_t.to(device)
            target_t = target_t.to(device)
            mask_t = mask_t.to(device)

            optimizer.zero_grad()
            logits, _ = model(q_t, resp_t, el_t)
            target_logits = logits.gather(2, q_t.unsqueeze(-1)).squeeze(-1)
            raw_loss = criterion(target_logits, target_t)
            masked_loss = (raw_loss * mask_t).sum() / (mask_t.sum() + 1e-8)
            masked_loss.backward()
            optimizer.step()

            total_loss += masked_loss.item()
            total_tokens += int(mask_t.sum().item())

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                elapsed_s = time.perf_counter() - ep_start
                throughput = (batch_idx + 1) * bs / max(1e-5, elapsed_s)
                logger.info(
                    f"Epoch {ep + 1}/{epochs} | Batch {batch_idx + 1}/{total_batches} "
                    f"({(batch_idx + 1) / total_batches * 100:.1f}%) | "
                    f"Loss: {total_loss / (batch_idx + 1):.4f} | Speed: {throughput:.0f} students/s"
                )

        ep_duration = time.perf_counter() - ep_start
        logger.info(f"Estimator Epoch {ep + 1}/{epochs} Complete in {ep_duration:.1f}s | Avg Loss: {total_loss / max(1, total_batches):.4f} | Tokens: {total_tokens:,}")

    # Calibration & validation evaluation
    ece_val, brier_val = 0.0, 0.0
    if val_loader:
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for q_t, resp_t, el_t, target_t, mask_t in val_loader:
                q_t = q_t.to(device)
                resp_t = resp_t.to(device)
                el_t = el_t.to(device)
                target_t = target_t.to(device)
                mask_t = mask_t.to(device)

                logits, _ = model(q_t, resp_t, el_t)
                target_logits = logits.gather(2, q_t.unsqueeze(-1)).squeeze(-1)
                probs = torch.sigmoid(target_logits)
                mask_bool = mask_t.bool()
                val_probs.extend(probs[mask_bool].cpu().numpy().tolist())
                val_labels.extend(target_t[mask_bool].cpu().numpy().tolist())

        if val_probs:
            probs_arr = np.array(val_probs)
            labels_arr = np.array(val_labels)
            ece_val = compute_ece(probs_arr, labels_arr)
            brier_val = compute_brier(probs_arr, labels_arr)
            logger.info(f"Estimator Calibration: ECE = {ece_val:.4f}, Brier Score = {brier_val:.4f}")

    model_path = models_dir / "gru_dkt.pt"
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_type": "GRU-DKT",
        "d_model": cfg.estimator.d_model,
        "d_hidden": cfg.estimator.d_hidden,
        "num_questions": num_questions,
        "epochs": epochs,
        "ece": float(ece_val),
        "brier_score": float(brier_val),
        "checkpoint_sha256": compute_file_sha256(model_path)
    }
    with open(artifacts_dir / "estimator_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Stage 2 Complete. Estimator saved to {model_path}")
    return model_path, metadata

def run_stage_policy(cfg, output_dir: Path, episodes: int = 25, train_steps: int = 50) -> Tuple[Path, Dict[str, Any]]:
    logger.info("=== STAGE 3: DIGITAL TWIN SIMULATION & OFFLINE RL POLICY (CQL) ===")
    models_dir = output_dir / "models"
    artifacts_dir = output_dir / "artifacts"

    env = ClassroomDigitalTwinEnv(cohort_size=cfg.twin.default_cohort_size, seed=42)
    cql = DiscreteCQL(
        d_state=env.get_state_vector().shape[0],
        num_actions=len(PEDAGOGICAL_ACTIONS),
        d_hidden=cfg.policy.d_hidden,
        lr=cfg.policy.learning_rate,
        cql_alpha=cfg.policy.cql_alpha
    )

    logger.info(f"Simulating {episodes} classroom sessions to populate transition dataset...")
    buffer_s, buffer_a, buffer_r, buffer_next_s, buffer_dur, buffer_done, buffer_mask = [], [], [], [], [], [], []

    rule_policy = RuleBasedPolicy()
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            rem_time = max(0.0, env.episode_duration_minutes - env.time_elapsed_minutes)
            mask = get_valid_action_mask(rem_time, env.cumulative_workload, cfg.policy.workload_budget)

            # Heuristic behavior policy with random exploration
            if np.random.rand() < 0.3:
                valid_idxs = [i for i, m in enumerate(mask) if m == 1]
                act_idx = np.random.choice(valid_idxs)
                act_name = PEDAGOGICAL_ACTIONS[act_idx]
            else:
                act_name, _ = rule_policy.select_action(
                    {"mean_mastery": float(state[0]), "mean_fatigue": float(state[6])},
                    mask
                )
                act_idx = PEDAGOGICAL_ACTIONS.index(act_name)

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

    logger.info(f"Transition buffer populated with {len(buffer_s)} steps. Training CQL policy...")
    states_t = torch.tensor(np.array(buffer_s), dtype=torch.float32)
    actions_t = torch.tensor(np.array(buffer_a), dtype=torch.long)
    rewards_t = torch.tensor(np.array(buffer_r), dtype=torch.float32)
    next_states_t = torch.tensor(np.array(buffer_next_s), dtype=torch.float32)
    durations_t = torch.tensor(np.array(buffer_dur), dtype=torch.float32)
    dones_t = torch.tensor(np.array(buffer_done), dtype=torch.float32)
    masks_t = torch.tensor(np.array(buffer_mask), dtype=torch.float32)

    n_samples = len(buffer_s)
    batch_size = min(cfg.policy.batch_size, n_samples)
    final_metrics = {}

    for step in range(train_steps):
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
        if (step + 1) % 10 == 0:
            logger.info(f"CQL Step {step + 1}/{train_steps} | Loss: {final_metrics['total_loss']:.4f} | Bellman: {final_metrics['bellman_loss']:.4f}")

    policy_path = models_dir / "cql_policy.pt"
    torch.save(cql.q_net.state_dict(), policy_path)

    metadata = {
        "policy_type": "Discrete-CQL",
        "d_state": env.get_state_vector().shape[0],
        "actions": PEDAGOGICAL_ACTIONS,
        "cql_alpha": cfg.policy.cql_alpha,
        "gamma_base": cfg.policy.gamma_base,
        "workload_budget": cfg.policy.workload_budget,
        "final_loss": final_metrics.get("total_loss", 0.0),
        "checkpoint_sha256": compute_file_sha256(policy_path)
    }
    with open(artifacts_dir / "policy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Stage 3 Complete. Policy saved to {policy_path}")
    return policy_path, metadata

def run_stage_release_manifest(output_dir: Path, est_meta: Dict[str, Any], pol_meta: Dict[str, Any]) -> Path:
    logger.info("=== STAGE 4: PACKAGING UNIFIED RELEASE MANIFEST ===")
    release_doc = {
        "release_id": f"classtwin_release_{int(time.time())}",
        "project": "ClassTwin - SCIC 2026",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modules": {
            "state_estimator": est_meta,
            "policy": pol_meta
        },
        "operational_thresholds": {
            "min_coverage_threshold": 0.60,
            "max_uncertainty_threshold": 0.45,
            "safe_fallback_enabled": True
        }
    }
    manifest_path = output_dir / "release_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(release_doc, f, indent=2)
    logger.info(f"Unified Release Bundle generated at {manifest_path}")
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="ClassTwin Unified Pipeline: Preprocess + State Estimator + CQL Policy")
    parser.add_argument("--config", type=str, default="configs/preprocess.yaml", help="Path to config file.")
    parser.add_argument("--output-dir", type=str, default="output", help="Root directory for models and artifacts.")
    parser.add_argument("--max-members", type=int, default=None, help="Max members to preprocess (default None = all).")
    parser.add_argument("--num-shards", type=int, default=20, help="Number of Parquet shards to train on (default 20 for fast training, set -1 for all).")
    parser.add_argument("--max-students", type=int, default=None, help="Subsample max students for fast training on CPU (default None = all).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override (default None = from config).")
    parser.add_argument("--force-preprocess", action="store_true", help="Force preprocessing even if shards exist.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Explicitly skip data preprocessing.")
    parser.add_argument("--skip-estimator", action="store_true", help="Skip state estimator training.")
    parser.add_argument("--skip-policy", action="store_true", help="Skip CQL policy training.")
    parser.add_argument("--estimator-epochs", type=int, default=None, help="Number of epochs for GRU-DKT (overrides config).")
    parser.add_argument("--policy-episodes", type=int, default=20, help="Episodes to simulate for CQL.")
    parser.add_argument("--policy-steps", type=int, default=50, help="Gradient steps for CQL.")
    args = parser.parse_args()

    set_seed(42)
    cfg = load_config(args.config)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shards_dir = Path(cfg.data.shards_dir)

    # Smart check: if shards already exist in train/ and val/
    has_shards, n_train, n_val, n_test = check_existing_shards(shards_dir)

    start_all = time.perf_counter()
    logger.info("=================================================================")
    logger.info("   CLASSTWIN UNIFIED PIPELINE (PREPROCESSING + ESTIMATOR + CQL)   ")
    logger.info("=================================================================")

    if has_shards and not args.force_preprocess:
        logger.info(
            f"[AUTO-DETECT] Phát hiện sẵn sàng Parquet shards trong '{shards_dir}' "
            f"({n_train} train shards, {n_val} val shards, {n_test} test shards)!"
        )
        logger.info(">> TỰ ĐỘNG BỎ QUA TIỀN XỬ LÝ (STAGE 1) VÀ CHẠY HUẤN LUYỆN TRỰC TIẾP!")
        args.skip_preprocess = True
    elif not has_shards and not args.skip_preprocess:
        logger.info(f"Chưa có Parquet shards trong {shards_dir}. Sẽ chạy tiền xử lý từ ZIP.")

    # Stage 1: Preprocessing
    if not args.skip_preprocess:
        run_stage_preprocessing(cfg, max_members=args.max_members, force=args.force_preprocess)

    # Determine num_shards to load (-1 means all shards)
    shards_to_load = None if args.num_shards == -1 else args.num_shards

    # Stage 2: Estimator
    est_meta = {}
    if not args.skip_estimator:
        epochs = args.estimator_epochs if args.estimator_epochs is not None else cfg.estimator.max_epochs
        _, est_meta = run_stage_estimator(
            cfg=cfg,
            output_dir=output_path,
            epochs=epochs,
            shards_dir=shards_dir,
            num_shards=shards_to_load,
            max_students=args.max_students,
            batch_size=args.batch_size
        )

    # Stage 3: Policy
    pol_meta = {}
    if not args.skip_policy:
        _, pol_meta = run_stage_policy(cfg, output_path, episodes=args.policy_episodes, train_steps=args.policy_steps)

    # Stage 4: Package Release
    manifest_p = run_stage_release_manifest(output_path, est_meta, pol_meta)

    total_elapsed = time.perf_counter() - start_all
    logger.info("=================================================================")
    logger.info(f"   PIPELINE FINISHED SUCCESSFULLY IN {total_elapsed:.1f}s !        ")
    logger.info(f"   Artifacts and models saved to: {output_path.resolve()}          ")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()
