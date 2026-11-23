"""BentoML service for the HW1 CIFAR-10 classifier, with optional adaptive batching.

    BATCHING=1 bentoml serve service:CifarClassifier --port 3111    # adaptive batching
    BATCHING=0 bentoml serve service:CifarClassifier --port 3111    # no batching

Endpoints:
    POST /predict      {"images": ["<base64 of 3072 raw uint8 bytes>", ...]} -> one prediction per image
    POST /stats        batch-size histogram and time spent in the model
    POST /reset_stats
"""

from __future__ import annotations

import os

import bentoml

from common import BATCHING, MAX_BATCH_SIZE, ModelRunner

# BentoML adaptive batching: `max_latency_ms` is NOT a fixed wait window. It is the latency
# budget of a request; the dispatcher learns how long a batch of a given size takes and decides
# on its own how long it is worth waiting for more requests (requests that cannot meet the
# budget are rejected with 503, so the budget is kept generous).
MAX_LATENCY_MS = int(os.environ.get("MAX_LATENCY_MS", "1000"))

batching_options = (
    {"batchable": True, "batch_dim": 0, "max_batch_size": MAX_BATCH_SIZE, "max_latency_ms": MAX_LATENCY_MS}
    if BATCHING
    else {"batchable": False}
)


# access logging is off: per-request logging would be part of the measured overhead
@bentoml.service(workers=1, traffic={"timeout": 60}, logging={"access": {"enabled": False}})
class CifarClassifier:
    def __init__(self) -> None:
        self.runner = ModelRunner()

    @bentoml.api(**batching_options)
    def predict(self, images: list[str]) -> list[dict]:
        # with batching on, `images` is the concatenation of the lists of many requests
        return self.runner.predict(images)

    @bentoml.api
    def stats(self) -> dict:
        return {**self.runner.stats(), "tool": "bentoml", "max_latency_ms": MAX_LATENCY_MS}

    @bentoml.api
    def reset_stats(self) -> dict:
        self.runner.reset_stats()
        return {"ok": True}
