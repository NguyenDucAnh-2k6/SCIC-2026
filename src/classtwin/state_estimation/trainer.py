"""
GRU-DKT State Estimator Training & Calibration Module.
Trains Deep Knowledge Tracing recurrent neural networks over student interaction shards
and computes ECE (Expected Calibration Error) and Brier calibration scores.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from classtwin.state_estimation.gru_dkt import GRUDKT
from classtwin.state_estimation.calibration import compute_ece, compute_brier
from classtwin.data.ednet.extractor import create_sequence_dataloader, load_sequences_from_shards
from classtwin.utils.logging import get_logger

logger = get_logger("classtwin.state_estimation.trainer")

def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def train_state_estimator(
    cfg: Any,
    output_dir: Path,
    epochs: Optional[int] = None,
    shards_dir: Optional[Path] = None,
    num_shards: Optional[int] = None,
    max_students: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Tuple[Path, Dict[str, Any]]:
    """
    Trains and calibrates GRU-DKT state estimator on Parquet shards.
    """
    logger.info("=== STAGE: STATE ESTIMATOR TRAINING & CALIBRATION (GRU-DKT) ===")
    models_dir = output_dir / "models"
    artifacts_dir = output_dir / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = epochs if epochs is not None else getattr(cfg.estimator, "max_epochs", 5)
    bs = batch_size or getattr(cfg.estimator, "batch_size", 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} (CUDA available: {torch.cuda.is_available()})")

    s_dir = shards_dir or Path(getattr(cfg.data, "shards_dir", "data/shards"))
    train_shards = list((s_dir / "train").glob("*.parquet")) if (s_dir / "train").exists() else []

    if train_shards:
        logger.info(f"Loading real interaction tracks from Parquet shards in {s_dir}/train...")
        train_dataset, q_to_idx, stats_train = load_sequences_from_shards(
            shards_dir=str(s_dir),
            split="train",
            num_shards=num_shards,
            max_students=max_students
        )
        val_shards = list((s_dir / "val").glob("*.parquet"))
        val_dataset = None
        if val_shards:
            val_n_shards = min(num_shards, len(val_shards)) if num_shards is not None else None
            val_dataset, _, _ = load_sequences_from_shards(
                shards_dir=str(s_dir),
                split="val",
                num_shards=val_n_shards,
                q_to_idx=q_to_idx,
                max_students=max_students // 5 if max_students else None
            )
    else:
        logger.warning(f"No Parquet shards found in {s_dir}. Using synthetic sequences for testing...")
        train_seqs = []
        for s_idx in range(50):
            seq = [{"student_id": f"u{s_idx}", "question_id": f"q{t % 25}", "is_correct": (t % 4 != 0), "elapsed_ms": 12000 + (t % 10) * 800} for t in range(40)]
            train_seqs.append(seq)
        q_to_idx = {f"q{i}": i + 1 for i in range(30)}
        train_dataset = train_seqs
        val_dataset = train_seqs[:10]

    vocab_path = artifacts_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(q_to_idx, f, indent=2)
    logger.info(f"Saved question vocabulary ({len(q_to_idx):,} questions) to {vocab_path}")

    train_loader = create_sequence_dataloader(train_dataset, q_to_idx=q_to_idx, batch_size=bs, shuffle=True)
    val_loader = create_sequence_dataloader(val_dataset, q_to_idx=q_to_idx, batch_size=bs, shuffle=False) if val_dataset else None

    num_questions = len(q_to_idx)
    model = GRUDKT(
        num_questions=num_questions,
        d_model=getattr(cfg.estimator, "d_model", 64),
        d_hidden=getattr(cfg.estimator, "d_hidden", 128),
        dropout=getattr(cfg.estimator, "dropout", 0.1)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=getattr(cfg.estimator, "learning_rate", 1e-3),
        weight_decay=getattr(cfg.estimator, "weight_decay", 1e-4)
    )

    total_batches = len(train_loader)
    log_interval = max(20, total_batches // 10)
    logger.info(f"Training GRU-DKT ({num_questions} questions) for {n_epochs} epochs on {device} ({total_batches:,} batches/epoch, bs={bs})...")

    for ep in range(n_epochs):
        model.train()
        total_loss, total_tokens = 0.0, 0
        ep_start = time.perf_counter()

        with tqdm(train_loader, desc=f"Estimator Epoch {ep + 1}/{n_epochs}", unit="batch", leave=True) as pbar:
            for batch_idx, (q_t, resp_t, el_t, target_t, mask_t) in enumerate(pbar):
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

                avg_l = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_l:.4f}", "toks": f"{total_tokens:,}"})

        ep_duration = time.perf_counter() - ep_start
        logger.info(f"Estimator Epoch {ep + 1}/{n_epochs} Complete in {ep_duration:.1f}s | Avg Loss: {total_loss / max(1, total_batches):.4f} | Tokens: {total_tokens:,}")

    ece_val, brier_val = 0.0, 0.0
    if val_loader:
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            with tqdm(val_loader, desc="Estimator Validation", unit="batch", leave=False) as pbar_val:
                for q_t, resp_t, el_t, target_t, mask_t in pbar_val:
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
        "d_model": getattr(cfg.estimator, "d_model", 64),
        "d_hidden": getattr(cfg.estimator, "d_hidden", 128),
        "num_questions": num_questions,
        "epochs": n_epochs,
        "ece": float(ece_val),
        "brier_score": float(brier_val),
        "checkpoint_sha256": compute_file_sha256(model_path)
    }
    with open(artifacts_dir / "estimator_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Stage Complete. Estimator saved to {model_path}")
    return model_path, metadata

