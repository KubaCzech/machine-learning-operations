"""Does adaptive batching remember past traffic? Measure 1 client before and after a burst of heavy load.

In the main benchmark BentoML with batching gave ~240 samples/s at 1 client on a fresh server, but only
~75-105 samples/s when the 1-client measurement came right after the 128-client one. This script tests that
directly, on one server: 1 client (fresh) -> 128 clients -> 1 client -> ... -> server restart -> 1 client.

    python benchmark/hysteresis.py                      # BentoML, adaptive batching
    python benchmark/hysteresis.py --tool ray           # Ray Serve (fixed window) as a control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "benchmark"))
from loadtest import run_load_test  # noqa: E402
from run_benchmark import SERVERS, post, start_server, stop_server, wait_for_power  # noqa: E402

PHASES = [("fresh", 1), ("burst", 128), ("after burst", 1), ("after burst, later", 1), ("burst", 128), ("after burst", 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tool", default="bentoml", choices=list(SERVERS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--out", default=str(LAB / "results" / "hysteresis.jsonl"))
    args = parser.parse_args()

    wait_for_power()

    env = {"BATCHING": "1", "DEVICE": args.device, "MAX_BATCH_SIZE": "64", "MAX_LATENCY_MS": "1000", "BATCH_WAIT_MS": "5"}
    log = LAB / "results" / "logs" / f"hysteresis_{args.tool}.log"
    url = SERVERS[args.tool]["url"]
    plan = [*PHASES, ("restarted server", 1)]

    proc = start_server(args.tool, env, log)
    try:
        for step, (phase, concurrency) in enumerate(plan, 1):
            if phase == "restarted server":
                stop_server(args.tool, proc)
                proc = start_server(args.tool, env, log)
            wait_for_power()
            stats: dict = {}
            result = run_load_test(
                url,
                concurrency,
                args.duration,
                warmup=3.0,  # short on purpose: we want to see the state the previous phase left behind
                on_window_start=lambda: post(url + "/reset_stats"),
                on_window_end=lambda: stats.update(post(url + "/stats")),
            )
            record = {
                "tool": args.tool,
                "device": args.device,
                "step": step,
                "phase": phase,
                **result,
                "server_mean_batch_size": stats["mean_batch_size"],
                "server_batch_size_histogram": stats["batch_size_histogram"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"{step}. {phase:<20} c={concurrency:<3} {record['throughput_sps']:6.0f} samples/s  "
                f"p50 {record['latency_p50_ms']:6.1f} ms  p95 {record['latency_p95_ms']:6.1f} ms  "
                f"mean batch {record['server_mean_batch_size']}",
                flush=True,
            )
    finally:
        stop_server(args.tool, proc)


if __name__ == "__main__":
    main()
