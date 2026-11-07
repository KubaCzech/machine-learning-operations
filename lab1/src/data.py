"""CIFAR-10: custom PyTorch Dataset + Lightning DataModule.

The Dataset reads the original python-pickle batches directly (no torchvision
dataset class), the DataModule handles download, train/val/test split and
batched on-device augmentation.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR_DIRNAME = "cifar-10-batches-py"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _load_batches(root: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    xs, ys = [], []
    for name in names:
        with open(root / CIFAR_DIRNAME / name, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        xs.append(batch["data"])
        ys.extend(batch["labels"])
    images = np.concatenate(xs).reshape(-1, 3, 32, 32)  # uint8, NCHW
    return images, np.asarray(ys, dtype=np.int64)


class CIFAR10Dataset(Dataset):
    """Holds uint8 images (N, 3, 32, 32) and labels; returns normalized float tensors."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        assert len(images) == len(labels)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx].float().div_(255.0)
        x = (x - self.mean) / self.std
        return x, self.labels[idx]


def batch_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random crop (with reflect padding) + horizontal flip, vectorised over the batch."""
    b, _, h, w = x.shape
    dev = x.device
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    iy = oy[:, None] + torch.arange(h, device=dev)[None, :]  # (B, H)
    ix = ox[:, None] + torch.arange(w, device=dev)[None, :]  # (B, W)
    bi = torch.arange(b, device=dev)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[bi, iy[:, :, None], ix[:, None, :]]  # (B, H, W, C)
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(b, device=dev) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous()


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 128,
        val_size: int = 5000,
        augment: bool = False,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)

    def prepare_data(self) -> None:
        if not (self.data_dir / CIFAR_DIRNAME / "test_batch").exists():
            download_and_extract_archive(CIFAR_URL, str(self.data_dir), md5=CIFAR_MD5)

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "train_ds"):
            return
        x, y = _load_batches(self.data_dir, train=True)
        # fixed random split (seed 0), independent of the training seed
        perm = np.random.default_rng(0).permutation(len(y))
        val_idx, train_idx = perm[: self.hparams.val_size], perm[self.hparams.val_size :]
        self.train_ds = CIFAR10Dataset(x[train_idx], y[train_idx])
        self.val_ds = CIFAR10Dataset(x[val_idx], y[val_idx])
        xt, yt = _load_batches(self.data_dir, train=False)
        self.test_ds = CIFAR10Dataset(xt, yt)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            persistent_workers=nw > 0,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        x, y = batch
        if self.hparams.augment and self.trainer is not None and self.trainer.training:
            x = batch_augment(x)
        return x, y
