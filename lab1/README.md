# Lab 1 (HW1) — PyTorch Lightning + MLflow + Optuna (CIFAR-10)

Polecenie: [`homeworks/HOMEWORK1.md`](../homeworks/HOMEWORK1.md)

Klasyfikacja obrazów CIFAR-10 małym ResNetem. Cały trening odbywa się w skryptach Pythona (`src/`),
notebook `notebooks/analysis.ipynb` jedynie wczytuje wyniki i przeprowadza analizę.

## Wymagania

* **Python 3.12** (testowane na 3.12.13). Python systemowy na macOS (3.9) jest za stary dla
  aktualnych wersji PyTorch / Lightning — dlatego używamy osobnego interpretera i wirtualnego środowiska.
* macOS (Apple Silicon → akceleracja MPS) lub Linux z CUDA / CPU (`accelerator="auto"`).

## Środowisko

Z [`uv`](https://docs.astral.sh/uv/) (instaluje odizolowany Python 3.12, nie dotyka systemowego):

```bash
cd lab1
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name lab1 --display-name "Python 3.12 (lab1)"   # kernel dla notebooka
```

Alternatywnie bez `uv` (wymaga zainstalowanego Pythona 3.12, np. z pyenv / brew):

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Struktura

```
lab1/
├── configs/
│   ├── baseline.yaml      # konfiguracja bazowa (bez tuningu i regularyzacji)
│   └── manual_fix.yaml    # zmiana wprowadzona na podstawie krzywych baseline'u
├── src/
│   ├── config.py          # TrainConfig (dataclass) + wczytywanie YAML / --set
│   ├── data.py            # CIFAR10Dataset (torch Dataset) + CIFAR10DataModule (LightningDataModule)
│   ├── model.py           # ResNetSmall + CIFAR10Classifier (LightningModule)
│   ├── train.py           # trening jednej konfiguracji, logowanie MLflow + CSV, ewaluacja na teście
│   └── optimize.py        # Optuna (TPE + MedianPruner, PyTorchLightningPruningCallback)
├── notebooks/analysis.ipynb   # analiza wyników (bez treningu)
├── run_all.sh             # cały pipeline
├── mlruns/                # MLflow tracking store (gitignored)
└── results/
    ├── runs/<run_name>/   # metrics.csv, summary.json, confusion matrix
    ├── optuna.db          # study Optuny (SQLite)
    └── best_params.yaml   # najlepsze hiperparametry
```

Współdzielone z innymi labami (w katalogu głównym repozytorium):

```
data/cifar-10-batches-py/          # CIFAR-10, pobierany automatycznie przy pierwszym treningu (gitignored)
models/lab1/<run_name>/
    ├── best.ckpt                  # checkpoint z najlepszym val_f1 (baseline, manual_fix, best)
    └── model_info.json            # konfiguracja, metryki val/test, klasy, normalizacja wejścia
```

Wczytanie modelu (np. w kolejnych labach):

```python
import sys; sys.path.insert(0, "lab1")
from src.model import CIFAR10Classifier
model = CIFAR10Classifier.load_from_checkpoint("models/lab1/best/best.ckpt", map_location="cpu").eval()
```

Triale Optuny nie zapisują checkpointów (zapisywany jest tylko finalny retrening `best`).

## Uruchomienie

```bash
source .venv/bin/activate
./run_all.sh                       # baseline -> manual_fix -> Optuna -> retrain best
N_TRIALS=40 TRIAL_EPOCHS=12 ./run_all.sh   # większy budżet HPO
```

Pojedyncze kroki:

```bash
python -m src.train --config configs/baseline.yaml --run-name baseline
python -m src.train --config configs/baseline.yaml --config configs/manual_fix.yaml --run-name manual_fix
python -m src.optimize --config configs/baseline.yaml --n-trials 30 --epochs 10
python -m src.train --config configs/baseline.yaml --config results/best_params.yaml --run-name best
python -m src.train --config configs/baseline.yaml --set lr=3e-3 augment=true --run-name custom   # override
```

Study Optuny jest zapisywane w SQLite, więc ponowne uruchomienie `src.optimize` dokłada kolejne triale.
Aby zacząć od zera: `rm results/optuna.db`.

## Monitoring

```bash
mlflow ui --backend-store-uri ./mlruns     # http://127.0.0.1:5000
```

Eksperymenty: `cifar10` (baseline, manual_fix, best) i `cifar10-optuna` (każdy trial osobno).
Logowane: `train_loss`, `val_loss`, `train/val_acc`, `train/val_f1` (macro), learning rate, czas epoki,
wszystkie hiperparametry oraz finalne metryki val/test najlepszego checkpointu.

## Analiza

```bash
jupyter lab notebooks/analysis.ipynb    # kernel: "Python 3.12 (lab1)"
```
