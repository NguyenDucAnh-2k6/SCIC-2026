import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.classtwin.state_estimation.gru_dkt import GRUDKT
from src.classtwin.state_estimation.calibration import compute_ece, compute_brier
from src.classtwin.data.sequences import create_sequence_dataloader, load_sequences_from_shards
from src.classtwin.utils.config import load_config
from src.classtwin.utils.logging import get_logger, set_seed

logger = get_logger("scripts.train_estimator")

def main():
    parser = argparse.ArgumentParser(description="Train GRU-DKT state estimator on Parquet shards.")
    parser.add_argument("--config", type=str, default="configs/estimator.yaml", help="Path to config file.")
    parser.add_argument("--shards-dir", type=str, default="data/shards", help="Directory containing train/, val/ Parquet shards.")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of shards to load (default None = all shards).")
    parser.add_argument("--max-students", type=int, default=None, help="Subsample max students for faster training on CPU.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save models and artifacts.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    epochs = args.epochs if args.epochs is not None else cfg.estimator.max_epochs
    bs = args.batch_size or cfg.estimator.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} (CUDA available: {torch.cuda.is_available()})")

    shards_path = Path(args.shards_dir)
    train_shards = list(shards_path.glob("train/*.parquet"))

    models_dir = Path(args.output_dir) / "models"
    artifacts_dir = Path(args.output_dir) / "artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if train_shards:
        logger.info(f"Detected {len(train_shards)} Parquet shards in {shards_path}/train. Loading real interaction tracks...")
        train_dataset, q_to_idx, stats_train = load_sequences_from_shards(
            shards_dir=str(shards_path),
            split="train",
            num_shards=args.num_shards,
            max_students=args.max_students
        )

        val_shards = list(shards_path.glob("val/*.parquet"))
        val_dataset = None
        if val_shards:
            val_dataset, _, stats_val = load_sequences_from_shards(
                shards_dir=str(shards_path),
                split="val",
                num_shards=min(20, len(val_shards)),
                q_to_idx=q_to_idx,
                max_students=args.max_students // 5 if args.max_students else None
            )
    else:
        logger.warning(f"No Parquet shards found in {shards_path}/train. Generating synthetic sequences for testing...")
        train_seqs = []
        for s_idx in range(50):
            seq = [{"student_id": f"u{s_idx}", "question_id": f"q{t % 20}", "is_correct": (t % 3 != 0), "elapsed_ms": 15000 + t * 500} for t in range(30)]
            train_seqs.append(seq)
        q_to_idx = {f"q{i}": i + 1 for i in range(25)}
        train_dataset = train_seqs
        val_dataset = train_seqs[:10]

    # Save question vocabulary
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

    for epoch in range(epochs):
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
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{total_batches} "
                    f"({(batch_idx + 1) / total_batches * 100:.1f}%) | "
                    f"Loss: {total_loss / (batch_idx + 1):.4f} | Speed: {throughput:.0f} students/s"
                )

        ep_duration = time.perf_counter() - ep_start
        avg_loss = total_loss / max(1, total_batches)
        logger.info(f"Epoch {epoch + 1}/{epochs} Complete in {ep_duration:.1f}s | Train Loss: {avg_loss:.4f} | Tokens: {total_tokens:,}")

    # Validation evaluation & calibration
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
            logger.info(f"Validation Calibration: ECE = {ece_val:.4f}, Brier Score = {brier_val:.4f}")

    save_path = models_dir / "gru_dkt.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model checkpoint successfully saved to {save_path}")

    metadata = {
        "model_type": "GRU-DKT",
        "num_questions": num_questions,
        "d_model": cfg.estimator.d_model,
        "d_hidden": cfg.estimator.d_hidden,
        "epochs": epochs,
        "ece": float(ece_val),
        "brier_score": float(brier_val),
        "shards_used": len(train_shards) if train_shards else 0
    }
    with open(artifacts_dir / "estimator_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Artifacts saved to {artifacts_dir / 'estimator_metadata.json'}")

if __name__ == "__main__":
    main()
