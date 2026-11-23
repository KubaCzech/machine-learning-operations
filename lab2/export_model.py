"""Export the HW1 Lightning checkpoint to TorchScript so the servers need only `torch`.

The exported module takes raw uint8 images (N, 32, 32, 3) and returns class
probabilities (N, 10): normalisation and softmax are baked into the graph, so
every serving tool runs exactly the same computation.

    python export_model.py                      # models/lab1/best -> models/lab2/
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab1"))

from src.model import CIFAR10Classifier  # noqa: E402


class ServingModel(nn.Module):
    """uint8 NHWC image batch -> float32 probabilities."""

    def __init__(self, net: nn.Module, mean: list[float], std: list[float]):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        return torch.softmax(self.net(x), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="best", help="run name under models/lab1/")
    parser.add_argument("--out", default=str(ROOT / "models" / "lab2"))
    args = parser.parse_args()

    src = ROOT / "models" / "lab1" / args.run
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "model_info.json").read_text())

    lit = CIFAR10Classifier.load_from_checkpoint(src / "best.ckpt", map_location="cpu").eval()
    model = ServingModel(lit.net, info["input"]["normalize_mean"], info["input"]["normalize_std"]).eval()

    example = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(out / "model.pt"))

    # sanity check on real test images: scripted == eager, and accuracy matches HW1
    with open(ROOT / "data" / "cifar-10-batches-py" / "test_batch", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC uint8
    labels = np.array(batch[b"labels"])
    loaded = torch.jit.load(str(out / "model.pt")).eval()
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(images))
        probs = torch.cat([loaded(x[i : i + 1000]) for i in range(0, len(x), 1000)])
        assert torch.allclose(probs[:64], model(x[:64]), atol=1e-5)
        assert torch.allclose(loaded(x[:1]), probs[:1], atol=1e-5), "batch size 1 differs"
    acc = float((probs.argmax(1).numpy() == labels).mean())

    # a few sample images for the request examples and the load test
    np.save(out / "sample_images.npy", images[:256])
    np.save(out / "sample_labels.npy", labels[:256])
    shutil.copy(src / "model_info.json", out / "model_info.json")

    print(f"exported {src / 'best.ckpt'} -> {out / 'model.pt'}")
    print(f"test accuracy of exported model: {acc:.4f} (HW1: {info['test']['test_acc']:.4f})")


if __name__ == "__main__":
    main()
