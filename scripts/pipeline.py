"""
ClassTwin Unified Master Pipeline:
Single flexible CLI entrypoint supporting modular execution across all lifecycle stages:
- Preprocess: Multi-source empirical priors and EdNet shards extraction with caching
- Estimator: GRU-DKT Neural Knowledge Tracing training & ECE calibration
- Policy: Digital Twin simulation & Offline RL (Discrete CQL / MaxEnt-IRL) with transition caching
- Evaluate: Rigorous benchmark against Random, Expert Heuristics, and CQL baselines
- All: End-to-end execution packaging release manifest

Usage Examples:
  python scripts/pipeline.py                               # Run complete end-to-end pipeline
  python scripts/pipeline.py --stage preprocess            # Run only multi-source preprocessing
  python scripts/pipeline.py --stage estimator             # Train only GRU-DKT state estimator
  python scripts/pipeline.py --stage policy                # Train only CQL pedagogical policy
  python scripts/pipeline.py --stage evaluate              # Run evaluation benchmarks only
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root and src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from classtwin.utils.config import load_config
from classtwin.utils.logging import get_logger, set_seed
from classtwin.data.preprocessing import run_multisource_preprocessing
from classtwin.state_estimation.trainer import train_state_estimator
from classtwin.policies.trainer import train_cql_policy
from classtwin.evaluation.benchmark import run_comprehensive_evaluation

logger = get_logger("classtwin.pipeline")

def check_existing_shards(shards_dir: Path):
    """Checks if data/shards already contains train, val, and test Parquet files."""
    train_count = len(list((shards_dir / "train").glob("*.parquet"))) if (shards_dir / "train").exists() else 0
    val_count = len(list((shards_dir / "val").glob("*.parquet"))) if (shards_dir / "val").exists() else 0
    test_count = len(list((shards_dir / "test").glob("*.parquet"))) if (shards_dir / "test").exists() else 0
    return (train_count > 0 and val_count > 0), train_count, val_count, test_count

def package_release_manifest(output_dir: Path, est_meta: Dict[str, Any], pol_meta: Dict[str, Any]) -> Path:
    logger.info("=== STAGE: PACKAGING UNIFIED RELEASE MANIFEST ===")
    manifest = {
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
        json.dump(manifest, f, indent=2)
    logger.info(f"Release Manifest saved to: {manifest_path}")
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="ClassTwin Unified Master Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "preprocess", "estimator", "policy", "evaluate"],
        default="all",
        help="Target execution stage: 'all', 'preprocess', 'estimator', 'policy', or 'evaluate'."
    )
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml", help="Path to config YAML file.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for models and artifacts.")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing datasets (default: from config).")
    parser.add_argument("--max-members", type=int, default=None, help="Max members for preprocessing ZIP archives.")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of Parquet shards for estimator training (default: from config or 20, -1 for all ~76M).")
    parser.add_argument("--max-students", type=int, default=None, help="Subsample max students for CPU training.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--estimator-epochs", type=int, default=None, help="GRU-DKT training epochs override.")
    parser.add_argument("--policy-episodes", type=int, default=None, help="Simulation episodes for policy training (default: from config or 300).")
    parser.add_argument("--policy-steps", type=int, default=None, help="Gradient optimization steps for CQL (default: from config or 500).")
    parser.add_argument("--reward-mode", type=str, choices=["explicit", "irl"], default="explicit", help="Reward mode: 'explicit' or 'irl'.")
    parser.add_argument("--no-cache", action="store_true", help="Force re-simulation and re-extraction without loading cached transitions.")
    parser.add_argument("--force-preprocess", action="store_true", help="Force full raw ZIP extraction even if shards exist.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip data preprocessing.")
    parser.add_argument("--skip-estimator", action="store_true", help="Skip state estimator training.")
    parser.add_argument("--skip-policy", action="store_true", help="Skip policy training.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip benchmark evaluation.")
    args = parser.parse_args()

    set_seed(42)
    cfg = load_config(args.config)
    if args.data_root:
        cfg.data.data_root = args.data_root

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shards_dir = Path(getattr(cfg.data, "shards_dir", "data/shards"))
    data_root = Path(getattr(cfg.data, "data_root", "data"))

    start_t = time.perf_counter()
    logger.info("=================================================================")
    logger.info("   CLASSTWIN UNIFIED PIPELINE (PREPROCESSING + DKT + CQL + EVAL) ")
    logger.info(f"   Target Stage: {args.stage.upper()} | Config: {args.config}")
    logger.info("=================================================================")

    # 1. Stage: Preprocessing (Always validate multi-source priors; reuse Parquet shards if present)
    if args.stage in ["all", "preprocess"] and not args.skip_preprocess:
        run_multisource_preprocessing(cfg=cfg, max_members=args.max_members, force=args.force_preprocess)

    # 2. Stage: Estimator
    est_meta = {}
    if args.stage in ["all", "estimator"] and not args.skip_estimator:
        cfg_shards = getattr(cfg.estimator, "num_shards", 20)
        target_shards = args.num_shards if args.num_shards is not None else cfg_shards
        shards_to_load = None if target_shards == -1 else target_shards
        _, est_meta = train_state_estimator(
            cfg=cfg,
            output_dir=output_path,
            epochs=args.estimator_epochs,
            shards_dir=shards_dir,
            num_shards=shards_to_load,
            max_students=args.max_students,
            batch_size=args.batch_size
        )

    # 3. Stage: Policy
    pol_meta = {}
    if args.stage in ["all", "policy"] and not args.skip_policy:
        cfg_episodes = getattr(cfg.policy, "episodes", 300)
        cfg_steps = getattr(cfg.policy, "train_steps", 500)
        target_episodes = args.policy_episodes if args.policy_episodes is not None else cfg_episodes
        target_steps = args.policy_steps if args.policy_steps is not None else cfg_steps
        _, pol_meta = train_cql_policy(
            cfg=cfg,
            output_dir=output_path,
            episodes=target_episodes,
            train_steps=target_steps,
            reward_mode=args.reward_mode,
            data_root=data_root,
            use_cache=not args.no_cache
        )

    # 4. Stage: Evaluation
    if args.stage in ["all", "evaluate"] and not args.skip_evaluate:
        d_state_dim = getattr(cfg.policy, "d_state", 32)
        run_comprehensive_evaluation(
            output_dir=output_path,
            num_episodes=10,
            d_state=d_state_dim,
            data_root=data_root
        )

    # 5. Packaging Release Manifest
    if args.stage == "all" and est_meta and pol_meta:
        package_release_manifest(output_path, est_meta, pol_meta)

    total_time = time.perf_counter() - start_t
    logger.info("=================================================================")
    logger.info(f"   PIPELINE FINISHED SUCCESSFULLY IN {total_time:.1f}s !        ")
    logger.info(f"   Artifacts and models saved to: {output_path.resolve()}          ")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()
