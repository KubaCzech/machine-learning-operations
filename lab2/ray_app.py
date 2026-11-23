"""Ray Serve application for the HW1 CIFAR-10 classifier, with optional dynamic batching.

    BATCHING=1 serve run ray_serve.yaml      # dynamic batching (@serve.batch), port 8111
    BATCHING=0 serve run ray_serve.yaml      # no batching

Same HTTP contract as the BentoML service (service.py):
    POST /predict      {"images": ["<base64 of 3072 raw uint8 bytes>", ...]} -> one prediction per image
    POST /stats        batch-size histogram and time spent in the model
    POST /reset_stats
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from common import BATCHING, MAX_BATCH_SIZE, ModelRunner

# Ray Serve dynamic batching: a batch is dispatched when it is full (MAX_BATCH_SIZE) or when
# BATCH_WAIT_MS has passed since its first request - a fixed window, whatever the traffic.
BATCH_WAIT_MS = float(os.environ.get("BATCH_WAIT_MS", "5"))


@serve.deployment(num_replicas=1, max_ongoing_requests=1024)
class CifarClassifier:
    def __init__(self) -> None:
        self.runner = ModelRunner()
        # the model runs in one worker thread, so the event loop keeps accepting requests meanwhile
        self.pool = ThreadPoolExecutor(max_workers=1)

    async def _run(self, images: list[str]) -> list[dict]:
        return await asyncio.get_running_loop().run_in_executor(self.pool, self.runner.predict, images)

    @serve.batch(max_batch_size=MAX_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_MS / 1000)
    async def _predict_batched(self, requests: list[list[str]]) -> list[list[dict]]:
        # `requests` holds the image lists of many HTTP requests: flatten, predict once, split back
        flat = [image for images in requests for image in images]
        predictions = await self._run(flat)
        out, start = [], 0
        for images in requests:
            out.append(predictions[start : start + len(images)])
            start += len(images)
        return out

    async def __call__(self, request: Request) -> JSONResponse:
        path = request.url.path.rstrip("/")
        if path.endswith("/stats"):
            return JSONResponse({**self.runner.stats(), "tool": "ray", "batch_wait_ms": BATCH_WAIT_MS})
        if path.endswith("/reset_stats"):
            self.runner.reset_stats()
            return JSONResponse({"ok": True})
        if not path.endswith("/predict"):
            return JSONResponse({"error": "use POST /predict"}, status_code=404)
        try:
            images = (await request.json())["images"]
            predict = self._predict_batched if BATCHING else self._run
            return JSONResponse(await predict(images))
        except (KeyError, ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)


app = CifarClassifier.bind()
