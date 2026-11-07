"""Train a single configuration.

Examples:
    python -m src.train --config configs/baseline.yaml --run-name baseline
    python -m src.train --config configs/baseline.yaml --config results/best_params.yaml --run-name best
    python -m src.train --config configs/baseline.yaml --set lr=3e-3 augment=true --run-name my_run
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from sklearn.metrics import confusion_matrix, f1_score

from .config import DATA_DIR, MLFLOW_URI, MODELS_DIR, RUNS_DIR, TrainConfig
from .data import CLASSES, MEAN, STD, CIFAR10DataModule
from .model import CIFAR10Classifier

MONITOR = "val_f1"


class BestMetricTracker(Callback):
    """Keeps the best value of a monitored metric (used as Optuna objective)."""

    def __init__(self, monitor: str = MONITOR):
        self.monitor = monitor
        self.best = float("-inf")
        self.best_epoch = -1

    # on_validation_end runs after LightningModule.on_validation_epoch_end has logged the metric
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val = trainer.callback_metrics.get(self.monitor)
        if val is not None and float(val) > self.best:
            self.best, self.best_epoch = float(val), trainer.current_epoch


class EpochTimer(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._t = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("epoch_time_s", time.perf_counter() - self._t)


def build(cfg: TrainConfig) -> tuple[CIFAR10DataModule, CIFAR10Classifier]:
    dm = CIFAR10DataModule(
        DATA_DIR,
        batch_size=cfg.batch_size,
        val_size=cfg.val_size,
        augment=cfg.augment,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    model = CIFAR10Classifier(
        width=cfg.width,
        blocks_per_stage=cfg.blocks_per_stage,
        dropout=cfg.dropout,
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
        scheduler=cfg.scheduler,
        label_smoothing=cfg.label_smoothing,
    )
    return dm, model


def run_training(
    cfg: TrainConfig,
    run_name: str,
    experiment: str = "cifar10",
    extra_callbacks: list[Callback] | None = None,
    evaluate_test: bool = True,
    save_checkpoint: bool = True,
    tags: dict[str, str] | None = None,
) -> dict:
    """Train one configuration, log to MLflow + CSV, return a summary dict."""
    L.seed_everything(cfg.seed, workers=True)
    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    model_dir = MODELS_DIR / run_name

    dm, model = build(cfg)
    mlf_logger = MLFlowLogger(experiment_name=experiment, run_name=run_name, tracking_uri=MLFLOW_URI, tags=tags)
    csv_logger = CSVLogger(save_dir=str(run_dir), name="", version="")
    mlf_logger.log_hyperparams(cfg.to_dict())
    csv_logger.log_hyperparams(cfg.to_dict())

    tracker = BestMetricTracker()
    callbacks: list[Callback] = [tracker, EpochTimer(), LearningRateMonitor(logging_interval="epoch")]
    ckpt = None
    if save_checkpoint:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        ckpt = ModelCheckpoint(
            dirpath=model_dir, filename="best", monitor=MONITOR, mode="max", save_top_k=1, enable_version_counter=False
        )
        callbacks.append(ckpt)
    callbacks += extra_callbacks or []

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=[mlf_logger, csv_logger],
        callbacks=callbacks,
        enable_checkpointing=save_checkpoint,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        log_every_n_steps=50,
        deterministic=False,
    )
    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    train_time = time.perf_counter() - t0

    summary = {
        "run_name": run_name,
        "config": cfg.to_dict(),
        "best_val_f1": tracker.best,
        "best_epoch": tracker.best_epoch,
        "epochs_run": trainer.current_epoch,
        "train_time_s": train_time,
        "n_params": sum(p.numel() for p in model.parameters()),
        "mlflow_run_id": mlf_logger.run_id,
    }

    if evaluate_test and ckpt is not None and ckpt.best_model_path:
        val_res = trainer.validate(model, datamodule=dm, ckpt_path="best", verbose=False)[0]
        model.test_preds.clear()
        model.test_targets.clear()
        test_res = trainer.test(model, datamodule=dm, ckpt_path="best", verbose=False)[0]
        preds = torch.cat(model.test_preds).numpy()
        targets = torch.cat(model.test_targets).numpy()
        cm = confusion_matrix(targets, preds, labels=list(range(10)))
        np.save(run_dir / "test_confusion_matrix.npy", cm)
        summary["val"] = {k: float(v) for k, v in val_res.items()}
        summary["test"] = {k: float(v) for k, v in test_res.items()}
        summary["test_per_class_f1"] = f1_score(targets, preds, average=None).tolist()
        mlf_logger.log_metrics({f"final_{k}": float(v) for k, v in {**val_res, **test_res}.items()})

    if ckpt is not None and ckpt.best_model_path:
        summary["checkpoint"] = str(Path(ckpt.best_model_path).relative_to(MODELS_DIR.parent.parent))
        model_info = {
            "checkpoint": "best.ckpt",
            "load": "from src.model import CIFAR10Classifier; CIFAR10Classifier.load_from_checkpoint(path)",
            "source": f"{MODELS_DIR.name}/src/model.py",
            "classes": CLASSES,
            "input": {"shape": [3, 32, 32], "normalize_mean": list(MEAN), "normalize_std": list(STD)},
            **{k: summary[k] for k in ("config", "best_val_f1", "best_epoch", "n_params", "mlflow_run_id")},
            **{k: summary[k] for k in ("val", "test") if k in summary},
        }
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", default=[], help="YAML file(s); later ones override earlier")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override single fields")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment", default="cifar10")
    args = parser.parse_args()

    cfg = TrainConfig.from_sources(args.config, args.set)
    summary = run_training(cfg, args.run_name, experiment=args.experiment)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
