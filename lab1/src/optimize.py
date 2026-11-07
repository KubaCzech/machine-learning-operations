"""Hyper-parameter optimization with Optuna (TPE sampler + median pruning).

Every trial is a separate MLflow run in the `cifar10-optuna` experiment.
The study is stored in SQLite (results/optuna.db), so it can be resumed
and analysed in the notebook.

Example:
    python -m src.optimize --config configs/baseline.yaml --n-trials 30 --epochs 10
"""

from __future__ import annotations

import argparse
import json

import optuna
import yaml
from optuna_integration import PyTorchLightningPruningCallback

from .config import RESULTS_DIR, TrainConfig
from .train import MONITOR, run_training

STUDY_DB = f"sqlite:///{RESULTS_DIR / 'optuna.db'}"


def suggest(trial: optuna.Trial) -> dict:
    """Search space."""
    params = {
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
        "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine", "onecycle"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "augment": trial.suggest_categorical("augment", [False, True]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "width": trial.suggest_categorical("width", [32, 48, 64]),
    }
    # learning-rate range depends on the optimizer
    if params["optimizer"] == "sgd":
        params["lr"] = trial.suggest_float("lr_sgd", 1e-2, 3e-1, log=True)
    else:
        params["lr"] = trial.suggest_float("lr_adamw", 1e-4, 1e-2, log=True)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", default=["configs/baseline.yaml"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10, help="epochs per trial (budget)")
    parser.add_argument("--timeout", type=int, default=None, help="seconds")
    parser.add_argument("--study-name", default="cifar10_resnet")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    base = TrainConfig.from_sources(args.config, args.set)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        cfg = TrainConfig(**{**base.to_dict(), **params, "max_epochs": args.epochs})
        summary = run_training(
            cfg,
            run_name=f"trial_{trial.number:03d}",
            experiment="cifar10-optuna",
            extra_callbacks=[PyTorchLightningPruningCallback(trial, monitor=MONITOR)],
            evaluate_test=False,
            save_checkpoint=False,
            tags={"study": args.study_name, "trial": str(trial.number)},
        )
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr("train_time_s", summary["train_time_s"])
        trial.set_user_attr("run_name", summary["run_name"])
        return summary["best_val_f1"]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=STUDY_DB,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=base.seed, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    # enqueue the baseline itself as trial 0 (same epoch budget) as a reference point
    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "optimizer": "adamw",
                "scheduler": base.scheduler,
                "weight_decay": max(base.weight_decay, 1e-6),
                "dropout": base.dropout,
                "label_smoothing": base.label_smoothing,
                "augment": base.augment,
                "batch_size": base.batch_size,
                "width": base.width,
                "lr_adamw": base.lr,
            }
        )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    best = study.best_trial
    best_params = dict(best.params)
    lr_sgd, lr_adamw = best_params.pop("lr_sgd", None), best_params.pop("lr_adamw", None)
    best_params["lr"] = lr_sgd if best_params["optimizer"] == "sgd" else lr_adamw
    with open(RESULTS_DIR / "best_params.yaml", "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=True)
    print(f"Best trial #{best.number}: val_f1={best.value:.4f}")
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    main()
