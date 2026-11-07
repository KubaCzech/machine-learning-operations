"""Training configuration: dataclass + YAML / CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]  # lab1/
REPO_ROOT = ROOT.parent
DATA_DIR = REPO_ROOT / "data"  # shared between labs
MODELS_DIR = REPO_ROOT / "models" / ROOT.name  # shared between labs: models/lab1/<run_name>/
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
MLFLOW_URI = f"file:{ROOT / 'mlruns'}"


@dataclass
class TrainConfig:
    # data
    batch_size: int = 128
    val_size: int = 5000
    augment: bool = False
    num_workers: int = 0
    # model
    width: int = 32  # channels in the first stage (doubled at each stage)
    blocks_per_stage: int = 1
    dropout: float = 0.0
    # optimization
    optimizer: str = "adam"  # adam | adamw | sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # used only by sgd
    scheduler: str = "none"  # none | cosine | onecycle
    label_smoothing: float = 0.0
    max_epochs: int = 20
    # misc
    seed: int = 42
    limit_train_batches: float = 1.0  # <1.0 for quick smoke tests
    limit_val_batches: float = 1.0

    @classmethod
    def from_sources(cls, yaml_paths: list[str] | None = None, overrides: list[str] | None = None) -> "TrainConfig":
        values: dict[str, Any] = {}
        for p in yaml_paths or []:
            with open(p) as f:
                values.update(yaml.safe_load(f) or {})
        for item in overrides or []:
            key, _, raw = item.partition("=")
            values[key] = yaml.safe_load(raw)  # parses ints / floats / bools
        known = {f.name: f.type for f in fields(cls)}
        unknown = set(values) - set(known)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**values)
        # yaml may parse 1e-3 as str in some cases -> coerce numeric fields
        for f in fields(cls):
            v = getattr(cfg, f.name)
            default = getattr(cls, f.name)
            if isinstance(default, bool):
                continue
            if isinstance(default, float) and not isinstance(v, float):
                setattr(cfg, f.name, float(v))
            elif isinstance(default, int) and not isinstance(v, int):
                setattr(cfg, f.name, int(v))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
