import argparse
import sys
import json
import time
from pathlib import Path

# Add project root and src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.classtwin.data.zip_reader import StreamingZipReader
from src.classtwin.data.adapters import adapt_kt1_row, extract_user_id_from_path
from src.classtwin.data.preprocessing import assign_student_split, ShardWriter
from src.classtwin.utils.config import load_config
from src.classtwin.utils.logging import get_logger

logger = get_logger("scripts.preprocess")

def main():
    parser = argparse.ArgumentParser(description="Preprocess EdNet ZIP into canonical Parquet shards.")
    parser.add_argument("--config", type=str, default="configs/preprocess.yaml", help="Path to config file.")
    parser.add_argument(
        "--max-members",
        type=int,
        default=None,
        help="Max members to process. Set to None (default) to process ALL members in the archive."
    )
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from previous checkpoint ledger.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    zip_path = cfg.data.raw_zip_path
    output_dir = Path(cfg.data.shards_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = output_dir / "processed_ledger.json"
    processed_set = set()
    if args.resume and ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                processed_set = set(json.load(f))
            logger.info(f"Resuming: found {len(processed_set)} previously committed members in ledger.")
        except Exception as e:
            logger.warning(f"Could not load ledger: {e}")

    logger.info(f"Starting preprocessing: {zip_path} -> {output_dir}")
    writer = ShardWriter(output_dir=str(output_dir), shard_size=cfg.data.chunk_size_rows)

    start_time = time.perf_counter()
    with StreamingZipReader(zip_path) as reader:
        members = reader.list_members(prefix="KT1/", max_members=args.max_members)
        total_members = len(members)
        logger.info(f"Total archive members to consider: {total_members}")

        batch_committed = 0
        for i, member_name in enumerate(members):
            if member_name in processed_set:
                continue

            student_id = extract_user_id_from_path(member_name)
            split = assign_student_split(student_id, seed=cfg.data.seed)
            rows = reader.read_csv_member(member_name)

            for row_idx, row in enumerate(rows):
                attempt, event = adapt_kt1_row(row, student_id=student_id, source_row=row_idx)
                if attempt:
                    writer.add_attempt(attempt.model_dump(), split=split)

            processed_set.add(member_name)
            batch_committed += 1

            if (batch_committed) % 1000 == 0:
                elapsed_s = time.perf_counter() - start_time
                rate = batch_committed / max(1.0, elapsed_s)
                logger.info(
                    f"Progress: {i + 1}/{total_members} members ({((i + 1) / total_members):.1%}) | "
                    f"Records: {writer.total_records:,} | Rate: {rate:.1f} members/s"
                )
                # Checkpoint ledger periodically
                with open(ledger_path, "w", encoding="utf-8") as f:
                    json.dump(list(processed_set), f)

    manifest = writer.close()
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(list(processed_set), f)

    total_time = time.perf_counter() - start_time
    logger.info(f"Preprocessing completed in {total_time:.1f}s. Total records: {writer.total_records:,}. Manifest: {manifest}")

if __name__ == "__main__":
    main()
