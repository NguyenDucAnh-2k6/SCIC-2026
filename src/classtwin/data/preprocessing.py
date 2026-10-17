import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from ..utils.logging import get_logger

logger = get_logger("classtwin.data.preprocessing")

def assign_student_split(student_id: str, seed: int = 42, train_r: float = 0.8, val_r: float = 0.1) -> str:
    """Deterministically assigns student to train, val, or test using MD5 hash (Protocol A)."""
    hash_val = hashlib.md5(f"{seed}_{student_id}".encode("utf-8")).hexdigest()
    normalized_val = int(hash_val[:8], 16) / 0xFFFFFFFF
    if normalized_val < train_r:
        return "train"
    elif normalized_val < (train_r + val_r):
        return "val"
    else:
        return "test"

class ShardWriter:
    """Buffers canonical attempts and writes them into partitioned Parquet shards."""
    def __init__(self, output_dir: str, shard_size: int = 50_000):
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.buffers: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        self.shard_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.total_records = 0
        
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "quarantine").mkdir(parents=True, exist_ok=True)

    def add_attempt(self, attempt_dict: Dict[str, Any], split: str) -> None:
        if split not in self.buffers:
            split = "train"
        self.buffers[split].append(attempt_dict)
        self.total_records += 1
        if len(self.buffers[split]) >= self.shard_size:
            self._flush_split(split)

    def _flush_split(self, split: str) -> None:
        records = self.buffers[split]
        if not records:
            return
        df = pd.DataFrame(records)
        shard_idx = self.shard_counts[split]
        shard_path = self.output_dir / split / f"shard_{shard_idx:05d}.parquet"
        df.to_parquet(shard_path, index=False, compression="snappy")
        logger.info(f"Committed {len(df)} records to {shard_path}")
        self.shard_counts[split] += 1
        self.buffers[split] = []

    def close(self) -> Dict[str, Any]:
        """Flushes remaining records and writes manifest."""
        for split in ["train", "val", "test"]:
            self._flush_split(split)
        manifest = {
            "total_records": self.total_records,
            "shard_counts": self.shard_counts,
            "output_dir": str(self.output_dir)
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Preprocess complete: {self.total_records} total records written.")
        return manifest
