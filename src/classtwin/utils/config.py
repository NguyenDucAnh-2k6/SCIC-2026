from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    raw_zip_path: str = "data/EdNet-KT1.zip"
    canonical_dir: str = "data/canonical"
    shards_dir: str = "data/shards"
    max_history_len: int = 200
    chunk_size_rows: int = 200_000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

class EstimatorConfig(BaseModel):
    model_type: str = "gru_dkt"  # "bkt" or "gru_dkt"
    d_model: int = 64
    d_hidden: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 5
    probe_sample_size: int = 10

class PolicyConfig(BaseModel):
    policy_type: str = "cql"
    d_state: int = 12
    d_hidden: int = 128
    learning_rate: float = 0.0003
    batch_size: int = 256
    gamma_base: float = 0.99
    tau_seconds: float = 60.0
    cql_alpha: float = 1.0
    workload_budget: float = 10.0
    harm_epsilon: float = 0.05
    polyak_tau: float = 0.005

class TwinConfig(BaseModel):
    default_cohort_size: int = 60
    episode_duration_minutes: int = 60
    mc_rollout_samples: int = 20
    scenarios: list[str] = ["neutral", "high_engagement", "fatigue_prone"]

class ClassTwinConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    twin: TwinConfig = Field(default_factory=TwinConfig)

def load_config(config_path: Optional[str] = None) -> ClassTwinConfig:
    """Loads a YAML configuration file or returns the default config."""
    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
            return ClassTwinConfig(**raw_dict)
    return ClassTwinConfig()
