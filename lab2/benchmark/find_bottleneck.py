"""Who is the bottleneck? GPU utilisation under load + a breakdown of what one request costs.

Part 1 (server under load): for every tool x batching mode, at 1 and 32 clients, run a 30 s load test and sample
  * GPU utilisation of the Apple GPU ("Device Utilization %" from `ioreg`, no sudo needed),
  * CPU of every server process (psutil), the share of time inside the model and the batch sizes (/stats).
Part 2 (no server): time the pieces of a request in-process - JSON parsing, base64 decoding, the model at batch
  size 1 and 64 - so the rest of the per-request time can be attributed to the framework (HTTP, routing, Python).

    python benchmark/find_bottleneck.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "benchmark"))
from loadtest import make_payload, run_load_test  # noqa: E402
from run_benchmark import SERVERS, post, server_cpu_seconds, start_server, stop_server, wait_for_power  # noqa: E402

GPU_RE = re.compile(r'"Device Utilization %"=(\d+)')


def gpu_utilisation() -> float | None:
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True)
    values = [int(v) for v in GPU_RE.findall(out.stdout)]
    return float(max(values)) if values else None


class GpuSampler:
    def __init__(self, period: float = 0.5) -> None:
        self.period, self.samples, self._stop = period, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = gpu_utilisation()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.period)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return float(np.mean(self.samples)) if self.samples else float("nan")


def under_load(duration: float) -> list[dict]:
    rows = []
    for tool in SERVERS:
        for batching in (0, 1):
            env = {"BATCHING": str(batching), "DEVICE": "mps", "MAX_BATCH_SIZE": "64"}
            proc = start_server(tool, env, LAB / "results" / "logs" / f"bottleneck_{tool}_{batching}.log")
            try:
                for concurrency in (1, 32):
                    wait_for_power()
                    url = SERVERS[tool]["url"]
                    window: dict = {"gpu": GpuSampler()}

                    def on_start() -> None:
                        post(url + "/reset_stats")
                        window["cpu0"] = server_cpu_seconds(proc)
                        window["gpu"].start()

                    def on_end() -> None:
                        window["gpu_util"] = window["gpu"].stop()
                        window["stats"] = post(url + "/stats")
                        window["cpu1"] = server_cpu_seconds(proc)

                    result = run_load_test(url, concurrency, duration, 10.0, on_start, on_end)
                    cpu: dict[str, float] = {}
                    for key, seconds in window["cpu1"].items():
                        name = key.split(":", 1)[1]
                        cpu[name] = cpu.get(name, 0.0) + (seconds - window["cpu0"].get(key, 0.0)) / duration
                    stats = window["stats"]
                    row = {
                        "tool": tool,
                        "batching": bool(batching),
                        "concurrency": concurrency,
                        "throughput_sps": result["throughput_sps"],
                        "latency_p50_ms": result["latency_p50_ms"],
                        "latency_p95_ms": result["latency_p95_ms"],
                        "gpu_util_pct": window["gpu_util"],
                        "model_time_share": stats["model_time_s"] / stats["wall_time_s"],
                        "model_ms_per_call": 1000 * stats["model_time_s"] / max(stats["model_calls"], 1),
                        "mean_batch_size": stats["mean_batch_size"],
                        "server_cpu_cores": {k: round(v, 2) for k, v in cpu.items() if v >= 0.02},
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            finally:
                stop_server(tool, proc)
    return rows


def per_request_costs() -> dict:
    """In-process timings, no HTTP: what does each step of one request cost?"""
    from common import ModelRunner, decode_images  # imported here: loads torch and the model

    payload = make_payload()
    runner = ModelRunner()
    images = json.loads(payload)["images"]

    def timeit(fn, n: int) -> float:
        for _ in range(min(n, 50)):
            fn()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        return 1000 * (time.perf_counter() - start) / n

    batch64 = images * 64
    return {
        "json_parse_ms": timeit(lambda: json.loads(payload), 5000),
        "base64_decode_ms": timeit(lambda: decode_images(images), 5000),
        "model_batch1_ms": timeit(lambda: runner._forward(decode_images(images)), 300),
        "model_batch64_ms": timeit(lambda: runner._forward(decode_images(batch64)), 100),
        "full_predict_batch1_ms": timeit(lambda: runner.predict(images), 300),
    }


def main() -> None:
    out = LAB / "results" / "bottleneck.json"
    wait_for_power()
    rows = under_load(duration=30.0)
    costs = per_request_costs()
    print(json.dumps(costs, indent=2))
    out.write_text(json.dumps({"under_load": rows, "per_request_costs": costs}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
