"""Run the whole benchmark: for every (tool, batching) configuration start the server, sweep the
concurrency levels, repeat every measurement, and append one JSON line per measurement.

    python benchmark/run_benchmark.py                                  # full sweep, both tools, MPS
    python benchmark/run_benchmark.py --tools bentoml --concurrency 1 8 32
    python benchmark/run_benchmark.py --device cpu --concurrency 1 8 32 --out results/raw_cpu.jsonl

Besides the client-side metrics (loadtest.py) every record holds what happened on the server
during the measurement window: the batch sizes the batcher really formed, the share of time
spent inside the model and the CPU used by the server processes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "benchmark"))
from loadtest import run_load_test  # noqa: E402

BIN = Path(sys.executable).parent
SERVERS = {
    "bentoml": {
        "cmd": [str(BIN / "bentoml"), "serve", "service:CifarClassifier", "--port", "3111"],
        "url": "http://127.0.0.1:3111",
    },
    "ray": {"cmd": [str(BIN / "serve"), "run", "ray_serve.yaml"], "url": "http://127.0.0.1:8111"},
}


def power_state() -> dict:
    """Power source and macOS Low Power Mode: both change CPU/GPU clocks, so they are recorded with every
    measurement (the first full run was silently invalidated by battery power and a sleeping laptop)."""
    state = {"on_ac_power": None, "low_power_mode": None}
    if platform.system() != "Darwin":
        return state
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        state["on_ac_power"] = "AC Power" in batt
        settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
        for line in settings.splitlines():
            if line.split()[:1] == ["lowpowermode"]:
                state["low_power_mode"] = line.split()[1] == "1"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return state


MAX_ATTEMPTS = 5


def power_ok(state: dict) -> bool:
    return state["on_ac_power"] is not False and not state["low_power_mode"]


def wait_for_power() -> None:
    """Block until the laptop is on AC power and Low Power Mode is off (laptops slow down on battery)."""
    if power_ok(power_state()):
        return
    print("waiting for AC power (plug in the charger, turn Low Power Mode off)...", flush=True)
    while not power_ok(power_state()):
        time.sleep(10)
    time.sleep(20)  # let clocks settle after the switch
    print("on AC power again", flush=True)


def post(url: str, timeout: float = 5.0) -> dict:
    request = urllib.request.Request(url, data=b"{}", headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def start_server(tool: str, env: dict, log_path: Path) -> subprocess.Popen:
    log = open(log_path, "w")
    proc = subprocess.Popen(
        SERVERS[tool]["cmd"], cwd=LAB, env={**os.environ, **env}, stdout=log, stderr=log, start_new_session=True
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{tool} server exited, see {log_path}")
        try:
            post(SERVERS[tool]["url"] + "/stats")
            return proc
        except Exception:
            time.sleep(1.0)
    stop_server(tool, proc)
    raise RuntimeError(f"{tool} server did not become ready, see {log_path}")


def stop_server(tool: str, proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if tool == "ray":
        subprocess.run([str(BIN / "ray"), "stop", "--force"], capture_output=True)
    time.sleep(3)


def server_cpu_seconds(proc: subprocess.Popen) -> dict[str, float]:
    """CPU seconds used so far by every process of the server, keyed by 'pid:name'."""
    out = {}
    try:
        root = psutil.Process(proc.pid)
        for p in [root, *root.children(recursive=True)]:
            try:
                times = p.cpu_times()
                cmd = " ".join(p.cmdline())
                name = next(
                    (
                        label
                        for key, label in [
                            ("ProxyActor", "ray proxy (HTTP)"),
                            ("ServeReplica", "ray replica (model)"),
                            ("ServeController", "ray controller"),
                            ("raylet", "ray raylet"),
                            ("gcs_server", "ray gcs"),
                            ("worker.service", "bentoml worker (HTTP + model)"),
                        ]
                        if key in cmd or key in p.name()
                    ),
                    "other",
                )
                out[f"{p.pid}:{name}"] = times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except psutil.NoSuchProcess:
        pass
    return out


def measure(tool: str, proc: subprocess.Popen, concurrency: int, duration: float, warmup: float) -> dict:
    url = SERVERS[tool]["url"]
    window: dict = {}

    def on_start() -> None:
        post(url + "/reset_stats")
        window["cpu0"] = server_cpu_seconds(proc)
        window["host_cpu0"] = psutil.cpu_times()

    def on_end() -> None:
        window["stats"] = post(url + "/stats")
        window["cpu1"] = server_cpu_seconds(proc)
        window["host_cpu1"] = psutil.cpu_times()

    result = run_load_test(url, concurrency, duration, warmup, on_start, on_end)

    stats = window["stats"]
    cpu: dict[str, float] = {}
    for key, seconds in window["cpu1"].items():
        name = key.split(":", 1)[1]
        cpu[name] = cpu.get(name, 0.0) + (seconds - window["cpu0"].get(key, 0.0)) / duration
    h0, h1 = window["host_cpu0"], window["host_cpu1"]
    busy = (h1.user - h0.user) + (h1.system - h0.system)
    result.update(
        server_model_time_share=stats["model_time_s"] / stats["wall_time_s"],
        server_mean_batch_size=stats["mean_batch_size"],
        server_batch_size_histogram=stats["batch_size_histogram"],
        server_model_calls=stats["model_calls"],
        server_cpu_cores={k: round(v, 3) for k, v in cpu.items() if v >= 0.005},
        server_cpu_cores_total=round(sum(cpu.values()), 3),
        host_cpu_cores_busy=round(busy / duration, 2),  # all processes on the machine, incl. the clients
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tools", nargs="+", default=["bentoml", "ray"], choices=list(SERVERS))
    parser.add_argument("--batching", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-ids", nargs="+", type=int, help="run only these repeat numbers (re-measuring)")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=15.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-latency-ms", type=int, default=1000, help="BentoML latency budget")
    parser.add_argument("--batch-wait-ms", type=float, default=5, help="Ray Serve batch_wait_timeout")
    parser.add_argument("--tag", default="main", help="label stored with every record")
    parser.add_argument("--out", default=str(LAB / "results" / "raw.jsonl"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (LAB / "results" / "logs").mkdir(parents=True, exist_ok=True)
    repeat_ids = args.repeat_ids or list(range(1, args.repeats + 1))
    total = len(args.tools) * len(args.batching) * len(args.concurrency) * len(repeat_ids)
    eta_min = total * (args.duration + args.warmup + 4) / 60
    print(f"{total} measurements, about {eta_min:.0f} min", flush=True)


    done = 0
    for tool in args.tools:
        for batching in args.batching:
            env = {
                "BATCHING": str(batching),
                "DEVICE": args.device,
                "MAX_BATCH_SIZE": str(args.max_batch_size),
                "MAX_LATENCY_MS": str(args.max_latency_ms),
                "BATCH_WAIT_MS": str(args.batch_wait_ms),
            }
            proc = start_server(tool, env, LAB / "results" / "logs" / f"{tool}_batching{batching}_{args.tag}.log")
            try:
                # repeats are the outer loop, so slow drift of the machine is spread over all levels
                for repeat in repeat_ids:
                    for concurrency in args.concurrency:
                        for attempt in range(1, MAX_ATTEMPTS + 1):
                            wait_for_power()
                            record = {
                                "tag": args.tag,
                                "tool": tool,
                                "batching": bool(batching),
                                "device": args.device,
                                "repeat": repeat,
                                "attempt": attempt,
                                "max_batch_size": args.max_batch_size,
                                "max_latency_ms": args.max_latency_ms if tool == "bentoml" else None,
                                "batch_wait_ms": args.batch_wait_ms if tool == "ray" else None,
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "host": f"{platform.machine()} {platform.system()} {platform.release()}",
                                **power_state(),
                                **measure(tool, proc, concurrency, args.duration, args.warmup),
                            }
                            record["window_end_power"] = power_state()
                            if power_ok(record) and power_ok(record["window_end_power"]):
                                break
                            # power changed during the window: keep the record aside and measure again
                            print(f"power changed during c={concurrency} rep={repeat}, measuring again", flush=True)
                            with open(out.with_name(out.stem + "_invalid_power.jsonl"), "a") as f:
                                f.write(json.dumps(record) + "\n")
                        with open(out, "a") as f:
                            f.write(json.dumps(record) + "\n")
                        done += 1
                        print(
                            f"[{done}/{total}] {tool} batching={batching} c={concurrency} rep={repeat}: "
                            f"{record['throughput_sps']:.0f} samples/s, p95 {record['latency_p95_ms']:.1f} ms, "
                            f"mean batch {record['server_mean_batch_size']}, "
                            f"model {record['server_model_time_share']:.0%}, "
                            f"failed {record['requests_failed']}, client cpu {record['client_max_cpu_share']:.0%}",
                            flush=True,
                        )
            finally:
                stop_server(tool, proc)


if __name__ == "__main__":
    main()
