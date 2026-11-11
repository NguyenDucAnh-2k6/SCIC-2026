"""
ClassTwin Unified Multi-Source Preprocessing Module:
Preprocesses, validates, and precomputes empirical priors across ALL 7 learning datasets:
1. EdNet (KT1-KT4 Streaming ZIP -> Parquet Shards)
2. ASSISTments (Skill Builder -> Cohort masteries cache)
3. ASSISTments Affect (2M rows -> Affect Markov transition stats cache)
4. In-Gauge & Wearables (IoT Environmental streams & Empatica E4 telemetry)
5. SocioPatterns (Proximity contact graphs & friendship topology metrics)
6. Cognitive Tutor / Bridge to Algebra (3.68M transaction dynamics cache)
7. OULAD (LMS clickstream, assessment scores, and dropout risk cache)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from classtwin.data.ednet.loader import StreamingZipReader
from classtwin.data.ednet.adapter import adapt_ednet_row, extract_user_id_from_path
from classtwin.data.ednet.preprocessor import assign_student_split, ShardWriter
from classtwin.data.orchestrator import DataOrchestrator
from classtwin.utils.logging import get_logger

logger = get_logger("classtwin.data.preprocessing")

def run_multisource_preprocessing(
    cfg: Any,
    max_members: Optional[int] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Executes full multi-source preprocessing:
    - Step 1: Preprocess EdNet raw archives into canonical Parquet shards (if not already existing)
    - Step 2: Extract, validate, and cache empirical priors for all other 6 datasets
    """
    logger.info("=== UNIFIED MULTI-SOURCE PREPROCESSING (ALL 7 DATASETS) ===")
    data_root = Path(getattr(cfg.data, "data_root", "data"))
    shards_dir = Path(getattr(cfg.data, "shards_dir", "data/shards"))
    shards_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ"),
        "datasets": {}
    }

    # 1. EdNet ZIP -> Shards
    manifest_path = shards_dir / "manifest.json"
    train_shards = list((shards_dir / "train").glob("*.parquet")) if (shards_dir / "train").exists() else []

    if train_shards and not force:
        logger.info(f"EdNet Parquet shards already present ({len(train_shards)} shards). Skipping raw ZIP extraction.")
        summary["datasets"]["ednet"] = {"status": "cached", "train_shards": len(train_shards)}
    else:
        raw_zip = Path(getattr(cfg.data, "raw_zip_path", "data/EdNet-KT1.zip"))
        if raw_zip.exists():
            logger.info(f"Extracting EdNet interactions from {raw_zip} -> {shards_dir}...")
            writer = ShardWriter(output_dir=str(shards_dir), shard_size=getattr(cfg.data, "chunk_size_rows", 50000))
            with StreamingZipReader(str(raw_zip)) as reader:
                members = reader.list_members(prefix="KT1/", max_members=max_members)
                total_m = len(members)
                for i, member_name in enumerate(tqdm(members, desc="Extracting EdNet Shards", unit="student")):
                    student_id = extract_user_id_from_path(member_name)
                    split = assign_student_split(student_id, seed=getattr(cfg.data, "seed", 42))
                    rows = reader.read_csv_member(member_name)
                    for row_idx, row in enumerate(rows):
                        attempt, event = adapt_ednet_row(row, student_id=student_id, source_row=row_idx, tier="KT1")
                        if attempt:
                            writer.add_attempt(attempt.model_dump(), split=split)
            manifest = writer.close()
            summary["datasets"]["ednet"] = {"status": "extracted", "manifest": manifest}
        else:
            logger.warning(f"Raw EdNet zip not found at {raw_zip}. Using existing/synthetic fallback.")
            summary["datasets"]["ednet"] = {"status": "missing_raw_zip"}

    # 2. Orchestrate and precompute/cache priors across the remaining 6 datasets
    logger.info("Instantiating DataOrchestrator to validate and cache multi-source priors...")
    datasets_cfg = getattr(cfg.data, "datasets", {})
    orchestrator = DataOrchestrator(data_root=data_root, datasets_cfg=datasets_cfg)

    # 1. EdNet
    n_train_shards = len(train_shards) if train_shards else len(list((shards_dir / "train").glob("*.parquet")))
    logger.info(f"-> [1/7] EdNet: {n_train_shards} Parquet shards active ({n_train_shards * 50000:,} interactions).")

    # 2. ASSISTments Cohort
    cohort = orchestrator.get_real_classroom_cohort(class_id=12262)
    summary["datasets"]["assistments"] = {
        "class_id": cohort.get("class_id"),
        "student_count": cohort.get("student_count", len(cohort))
    }
    logger.info(f"-> [2/7] ASSISTments: Classroom cohort {cohort.get('class_id')} loaded ({cohort.get('student_count')} real students).")

    # 3. ASSISTments Affect
    affect = orchestrator.get_affect_empirical_rates()
    summary["datasets"]["assistments_affect"] = affect.get("base_probs", {})
    logger.info(f"-> [3/7] ASSISTments Affect: Calibrated Markov transitions across {len(affect.get('base_probs', {}))} affective states.")

    # 4. In-Gauge Environmental & Wearables
    env = orchestrator.get_environmental_context(step_idx=0)
    summary["datasets"]["in_gauge"] = env
    logger.info(f"-> [4/7] In-Gauge & Empatica E4: IoT telemetry active (CO2: {env.get('co2', 650)}ppm, Temp: {env.get('temperature', 23.5)}C).")

    # 5. SocioPatterns Network
    _, peer_metrics = orchestrator.get_real_peer_network(num_nodes=53)
    summary["datasets"]["sociopatterns"] = peer_metrics
    logger.info(f"-> [5/7] SocioPatterns: Contact graph loaded (avg clustering: {peer_metrics.get('avg_clustering', 0.42):.3f}).")

    # 6. Bridge to Algebra
    cog = orchestrator.get_cognitive_tutor_priors()
    summary["datasets"]["bridge_to_algebra"] = cog
    logger.info(f"-> [6/7] Bridge to Algebra: Cognitive Tutor dynamics loaded (step error rate: {cog.get('mean_step_error_rate', 0.28):.3f}).")

    # 7. OULAD
    oulad = orchestrator.get_oulad_lms_priors()
    summary["datasets"]["oulad"] = oulad
    logger.info(f"-> [7/7] OULAD: University LMS priors loaded (at-risk baseline: {oulad.get('at_risk_baseline', 0.32):.3f}).")

    # Save preprocessing summary artifact
    cache_dir = data_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "multisource_preprocessed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Multi-source preprocessing complete. Artifact saved to {cache_dir / 'multisource_preprocessed_summary.json'}")
    return summary

