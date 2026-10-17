import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.classtwin.state_estimation.gru_dkt import GRUDKT
from src.classtwin.data.sequences import create_sequence_dataloader
from src.classtwin.utils.config import load_config
from src.classtwin.utils.logging import get_logger, set_seed

logger = get_logger("scripts.train_estimator")

def main():
    parser = argparse.ArgumentParser(description="Train GRU-DKT state estimator.")
    parser.add_argument("--config", type=str, default="configs/estimator.yaml", help="Path to config file.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    logger.info("Initializing synthetic sample sequence batch for verification...")
    # Generate mock student interaction sequences for training demonstration
    mock_seqs = []
    for s_idx in range(50):
        seq = []
        for t in range(30):
            seq.append({
                "question_id": f"q{t % 15}",
                "is_correct": (t % 3 != 0),
                "elapsed_ms": 15000 + t * 500
            })
        mock_seqs.append(seq)

    q_map = {f"q{i}": i + 1 for i in range(20)}
    dataloader = create_sequence_dataloader(mock_seqs, q_to_idx=q_map, batch_size=16)

    model = GRUDKT(num_questions=20, d_model=cfg.estimator.d_model, d_hidden=cfg.estimator.d_hidden)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.estimator.learning_rate)

    logger.info(f"Training GRU-DKT for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for q_t, resp_t, el_t, target_t, mask_t in dataloader:
            optimizer.zero_grad()
            logits, _ = model(q_t, resp_t, el_t)

            # Gather target logits for the specific next question
            target_logits = logits.gather(2, q_t.unsqueeze(-1)).squeeze(-1)
            raw_loss = criterion(target_logits, target_t)
            masked_loss = (raw_loss * mask_t).sum() / (mask_t.sum() + 1e-8)

            masked_loss.backward()
            optimizer.step()

            total_loss += masked_loss.item()
            total_tokens += int(mask_t.sum().item())

        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Loss: {total_loss / max(1, len(dataloader)):.4f}")

    save_path = Path("output/models/gru_dkt.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model checkpoint successfully saved to {save_path}")

if __name__ == "__main__":
    main()
