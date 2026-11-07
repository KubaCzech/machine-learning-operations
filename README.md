# machine-learning-operations

```
homeworks/          # polecenia zadań: HOMEWORK1.md, HOMEWORK2.md, ...
lab1/               # rozwiązanie zadania 1 (własny venv, requirements.txt, README)
lab2/               # rozwiązanie zadania 2
...
data/               # zbiory danych współdzielone między labami (gitignored, pobierane automatycznie)
models/             # wytrenowane modele do ponownego użycia: models/labX/<run_name>/
```

Każdy folder `labX/` jest samodzielny: ma własne `requirements.txt`, wirtualne środowisko `.venv`
(nie jest commitowane) i `README.md` z wymaganą wersją Pythona i instrukcją uruchomienia.
Dane (`data/`) i modele (`models/`) leżą w katalogu głównym, żeby kolejne laby mogły z nich korzystać.

| Lab | Temat | Polecenie | Dane | Modele |
|---|---|---|---|---|
| [lab1](lab1/) | PyTorch Lightning, monitoring (MLflow), HPO (Optuna) | [HOMEWORK1.md](homeworks/HOMEWORK1.md) | `data/cifar-10-batches-py` | `models/lab1/` |
