"""Code shared by both servers (BentoML and Ray Serve): model loading, payload decoding, inference.

Keeping it in one place guarantees that both tools run exactly the same model code,
so the benchmark compares the serving layers and their batchers, not two implementations.

Configuration (environment variables, read by both servers):
    DEVICE          mps | cpu | cuda          (default: mps if available, else cpu)
    TORCH_THREADS   intra-op threads on CPU   (default: 4)
    BATCHING        1 = dynamic batching on, 0 = off   (default: 1)
    MAX_BATCH_SIZE  upper bound of a batch    (default: 64)
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "lab2"))

BATCHING = os.environ.get("BATCHING", "1") == "1"
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))
IMAGE_SHAPE = (32, 32, 3)
IMAGE_BYTES = 32 * 32 * 3


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_image(image: np.ndarray) -> str:
    """uint8 HWC image (32, 32, 3) -> base64 string of its raw bytes (the wire format)."""
    assert image.shape == IMAGE_SHAPE and image.dtype == np.uint8
    return base64.b64encode(image.tobytes()).decode("ascii")


def decode_images(images: list[str]) -> np.ndarray:
    """List of base64 strings -> uint8 array (N, 32, 32, 3)."""
    raw = b"".join(base64.b64decode(s) for s in images)
    if len(raw) != len(images) * IMAGE_BYTES:
        raise ValueError(f"every image must be {IMAGE_BYTES} raw uint8 bytes (32x32x3, RGB), base64-encoded")
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(images), *IMAGE_SHAPE)


class ModelRunner:
    """Loads the TorchScript model and runs (batched) inference, one call at a time.

    It also records what the batcher really did: a histogram of the batch sizes it
    received and the time spent inside the model, exposed by the servers at /stats.
    """

    def __init__(self) -> None:
        self.device = os.environ.get("DEVICE") or default_device()
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
        # NNPACK convolutions (picked by PyTorch for batches >= 16) are ~10x slower on Apple Silicon
        torch.backends.nnpack.set_flags(False)
        # load on CPU first: the traced graph holds a float64 constant that MPS refuses to load directly
        self.model = torch.jit.load(str(MODEL_DIR / "model.pt"), map_location="cpu").eval().to(self.device)
        self.classes: list[str] = json.loads((MODEL_DIR / "model_info.json").read_text())["classes"]
        self._lock = threading.Lock()
        self.reset_stats()
        # warm-up: first calls pay for JIT optimisation and Metal shader compilation
        for bs in (1, 2, 4, 8, 16, 32, 64):
            for _ in range(3):
                self._forward(np.zeros((bs, *IMAGE_SHAPE), dtype=np.uint8))
        self.reset_stats()

    def _forward(self, images: np.ndarray) -> np.ndarray:
        with self._lock, torch.inference_mode():
            start = time.perf_counter()
            probs = self.model(torch.from_numpy(images).to(self.device)).cpu().numpy()
            self._model_time += time.perf_counter() - start
            self._batch_sizes[len(images)] += 1
        return probs

    def predict(self, images: list[str]) -> list[dict]:
        """One prediction per input image; `images` may come from many batched requests."""
        probs = self._forward(decode_images(images))
        ids = probs.argmax(axis=1)
        return [
            {"label": self.classes[i], "class_id": int(i), "confidence": round(float(p[i]), 4)}
            for i, p in zip(ids, probs)
        ]

    def reset_stats(self) -> None:
        with self._lock:
            self._batch_sizes: Counter[int] = Counter()
            self._model_time = 0.0
            self._since = time.perf_counter()

    def stats(self) -> dict:
        with self._lock:
            calls = sum(self._batch_sizes.values())
            samples = sum(size * n for size, n in self._batch_sizes.items())
            return {
                "device": self.device,
                "batching": BATCHING,
                "max_batch_size": MAX_BATCH_SIZE,
                "model_calls": calls,
                "samples": samples,
                "mean_batch_size": round(samples / calls, 3) if calls else 0.0,
                "batch_size_histogram": {str(k): v for k, v in sorted(self._batch_sizes.items())},
                "model_time_s": round(self._model_time, 4),
                "wall_time_s": round(time.perf_counter() - self._since, 4),
            }
