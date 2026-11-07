"""Small ResNet for CIFAR-10 wrapped in a LightningModule."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

NUM_CLASSES = 10


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetSmall(nn.Module):
    """3-stage ResNet: widths (w, 2w, 4w), spatial 32 -> 16 -> 8."""

    def __init__(self, width: int = 32, blocks_per_stage: int = 1, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        layers, c_in = [], width
        for i, c_out in enumerate([width, 2 * width, 4 * width]):
            for j in range(blocks_per_stage):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BasicBlock(c_in, c_out, stride))
                c_in = c_out
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(self.stem(x)))


class CIFAR10Classifier(L.LightningModule):
    def __init__(
        self,
        width: int = 32,
        blocks_per_stage: int = 1,
        dropout: float = 0.0,
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        scheduler: str = "none",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = ResNetSmall(width, blocks_per_stage, dropout)
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(NUM_CLASSES, average="micro"),
                "f1": MulticlassF1Score(NUM_CLASSES, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch, metrics: MetricCollection, stage: str, label_smoothing: float = 0.0):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        preds = logits.argmax(1)
        metrics.update(preds, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(y))
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, self.train_metrics, "train", self.hparams.label_smoothing)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.val_metrics, "val")

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        _, preds, y = self._step(batch, self.test_metrics, "test")
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        hp = self.hparams
        # no weight decay on BatchNorm / bias parameters
        decay = [p for n, p in self.named_parameters() if p.ndim > 1]
        no_decay = [p for n, p in self.named_parameters() if p.ndim <= 1]
        groups = [{"params": decay, "weight_decay": hp.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        if hp.optimizer == "adam":
            opt = torch.optim.Adam(groups, lr=hp.lr)
        elif hp.optimizer == "adamw":
            opt = torch.optim.AdamW(groups, lr=hp.lr)
        elif hp.optimizer == "sgd":
            opt = torch.optim.SGD(groups, lr=hp.lr, momentum=hp.momentum, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {hp.optimizer}")

        if hp.scheduler == "none":
            return opt
        if hp.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        if hp.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=hp.lr, total_steps=self.trainer.estimated_stepping_batches
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        raise ValueError(f"Unknown scheduler: {hp.scheduler}")
