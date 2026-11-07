#!/usr/bin/env bash
# Full pipeline for HW1. Run from anywhere; uses .venv/bin/python by default.
#   N_TRIALS=40 TRIAL_EPOCHS=12 ./run_all.sh
#   QUICK=1 ./run_all.sh      # few-minute smoke test of the whole pipeline (NOT for real results)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
N_TRIALS=${N_TRIALS:-30}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-10}
SET=()
if [[ "${QUICK:-0}" == "1" ]]; then
  N_TRIALS=4; TRIAL_EPOCHS=2
  SET=(--set max_epochs=3 limit_train_batches=0.05 limit_val_batches=0.3)
fi

# 1. baseline
$PY -m src.train --config configs/baseline.yaml --run-name baseline ${SET[@]+"${SET[@]}"}
# 2. manual change motivated by the baseline curves
$PY -m src.train --config configs/baseline.yaml --config configs/manual_fix.yaml --run-name manual_fix ${SET[@]+"${SET[@]}"}
# 3. hyper-parameter optimization (shorter epoch budget per trial + pruning)
$PY -m src.optimize --config configs/baseline.yaml --n-trials "$N_TRIALS" --epochs "$TRIAL_EPOCHS" ${SET[@]+"${SET[@]}"}
# 4. retrain the best configuration with the full baseline epoch budget
$PY -m src.train --config configs/baseline.yaml --config results/best_params.yaml --run-name best ${SET[@]+"${SET[@]}"}
