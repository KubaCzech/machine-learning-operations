"""Closed-loop HTTP load test: N concurrent clients, each sending the next request as soon as
the previous one returns. One request = one image = one prediction, so requests/s == samples/s.

    python benchmark/loadtest.py --url http://127.0.0.1:3111 --concurrency 32

Methodology (HW2): warm-up traffic is discarded, then throughput and latency percentiles are
measured over a fixed window. The clients are spread over several processes (asyncio + aiohttp
in each) so that the Python client is not the bottleneck; the CPU use of the busiest client
process is reported to prove it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL_DIR, encode_image  # noqa: E402

MAX_CLIENT_PROCESSES = 4


def make_payload(image_index: int = 0) -> bytes:
    image = np.load(MODEL_DIR / "sample_images.npy")[image_index]
    return json.dumps({"images": [encode_image(image)]}).encode()


async def _client(session: aiohttp.ClientSession, url: str, payload: bytes, stop_at: float, out: list) -> None:
    headers = {"content-type": "application/json"}
    while time.time() < stop_at:
        start = time.time()
        try:
            async with session.post(url, data=payload, headers=headers) as response:
                await response.read()
                ok = response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            ok = False
            await asyncio.sleep(0.01)  # do not spin if the server is down
        end = time.time()
        out.append((end, end - start, ok))


async def _run_clients(url: str, payload: bytes, n_clients: int, start_at: float, stop_at: float) -> list:
    out: list = []
    connector = aiohttp.TCPConnector(limit=0)  # one keep-alive connection per client
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        await asyncio.sleep(max(0.0, start_at - time.time()))
        await asyncio.gather(*[_client(session, url, payload, stop_at, out) for _ in range(n_clients)])
    return out


def _worker(url: str, payload: bytes, n_clients: int, start_at: float, stop_at: float, queue: mp.Queue) -> None:
    records = asyncio.run(_run_clients(url, payload, n_clients, start_at, stop_at))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_share = (usage.ru_utime + usage.ru_stime) / (stop_at - start_at)
    queue.put((np.array(records, dtype=np.float64).reshape(-1, 3), cpu_share))


def run_load_test(
    url: str, concurrency: int, duration: float = 60.0, warmup: float = 15.0, on_window_start=None, on_window_end=None
) -> dict:
    """Run one measurement and return its metrics. The optional callbacks fire at the window edges."""
    payload = make_payload()
    n_procs = min(concurrency, MAX_CLIENT_PROCESSES)
    shares = [concurrency // n_procs + (i < concurrency % n_procs) for i in range(n_procs)]

    start_at = time.time() + 1.5  # give the worker processes time to start
    window_start = start_at + warmup
    window_end = window_start + duration
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(f"{url}/predict", payload, n, start_at, window_end, queue)) for n in shares
    ]
    for p in procs:
        p.start()

    time.sleep(max(0.0, window_start - time.time()))
    if on_window_start:
        on_window_start()
    time.sleep(max(0.0, window_end - time.time()))
    if on_window_end:
        on_window_end()

    results = [queue.get(timeout=60) for _ in procs]
    for p in procs:
        p.join()

    records = np.concatenate([r for r, _ in results])
    in_window = records[(records[:, 0] >= window_start) & (records[:, 0] <= window_end)]  # completed in window
    ok = in_window[in_window[:, 2] == 1]
    latency_ms = ok[:, 1] * 1000
    pct = (lambda q: float(np.percentile(latency_ms, q))) if len(ok) else (lambda q: float("nan"))
    return {
        "concurrency": concurrency,
        "duration_s": duration,
        "warmup_s": warmup,
        "client_processes": n_procs,
        "requests_ok": int(len(ok)),
        "requests_failed": int(len(in_window) - len(ok)),
        "throughput_sps": len(ok) / duration,  # 1 sample per request
        "latency_mean_ms": float(latency_ms.mean()) if len(ok) else float("nan"),
        "latency_p50_ms": pct(50),
        "latency_p95_ms": pct(95),
        "latency_p99_ms": pct(99),
        "client_max_cpu_share": max(cpu for _, cpu in results),  # 1.0 = one client process saturates a core
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:3111", help="server base URL")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=60.0, help="measurement window [s]")
    parser.add_argument("--warmup", type=float, default=15.0, help="discarded warm-up [s]")
    args = parser.parse_args()
    print(json.dumps(run_load_test(args.url, args.concurrency, args.duration, args.warmup), indent=2))


if __name__ == "__main__":
    main()
