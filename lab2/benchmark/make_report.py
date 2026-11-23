"""Aggregate results/raw*.jsonl into results/results.md (tables) and results/*.png (plots).

    python benchmark/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
TOOLS = {"bentoml": "BentoML", "ray": "Ray Serve"}
# colour = tool, line style / marker fill = batching (so identity never rests on colour alone)
COLOR = {"bentoml": "#2a78d6", "ray": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#1a1a19", "#6b6a63", "#e4e3de", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
    }
)


def load(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpu_proxy"] = df["server_cpu_cores"].map(lambda d: d.get("ray proxy (HTTP)", 0.0))
    df["failed_share"] = df["requests_failed"] / (df["requests_ok"] + df["requests_failed"]).clip(lower=1)
    agg = (
        df.groupby(["tool", "batching", "concurrency"])
        .agg(
            n=("throughput_sps", "size"),
            thr=("throughput_sps", "mean"),
            thr_std=("throughput_sps", "std"),
            p50=("latency_p50_ms", "mean"),
            p95=("latency_p95_ms", "mean"),
            p95_std=("latency_p95_ms", "std"),
            p99=("latency_p99_ms", "mean"),
            batch=("server_mean_batch_size", "mean"),
            model_share=("server_model_time_share", "mean"),
            server_cpu=("server_cpu_cores_total", "mean"),
            client_cpu=("client_max_cpu_share", "max"),
            failed=("failed_share", "max"),
        )
        .reset_index()
    )
    return agg.fillna({"thr_std": 0.0, "p95_std": 0.0})


def cell(agg: pd.DataFrame, tool: str, batching: bool, concurrency: int) -> pd.Series | None:
    row = agg[(agg.tool == tool) & (agg.batching == batching) & (agg.concurrency == concurrency)]
    return None if row.empty else row.iloc[0]


def pm(mean: float, std: float, digits: int = 0) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def homework_table(agg: pd.DataFrame, tool: str) -> list[str]:
    lines = [
        "| Concurrency | Throughput no batching [samples/s] | Throughput with batching [samples/s] | Speed-up "
        "| p95 latency no batching [ms] | p95 latency with batching [ms] |",
        "|---|---|---|---|---|---|",
    ]
    for c in sorted(agg[agg.tool == tool].concurrency.unique()):
        off, on = cell(agg, tool, False, c), cell(agg, tool, True, c)
        if off is None or on is None:
            continue
        lines.append(
            f"| {c} | {pm(off.thr, off.thr_std)} | {pm(on.thr, on.thr_std)} | {on.thr / off.thr:.2f}× "
            f"| {pm(off.p95, off.p95_std, 1)} | {pm(on.p95, on.p95_std, 1)} |"
        )
    return lines


def comparison_table(agg: pd.DataFrame) -> list[str]:
    tools = [t for t in TOOLS if t in set(agg.tool)]
    head = "| Concurrency | " + " | ".join(f"{TOOLS[t]}, no batching | {TOOLS[t]}, batching" for t in tools) + " |"
    lines = [head, "|---" * (1 + 2 * len(tools)) + "|"]
    for c in sorted(agg.concurrency.unique()):
        cells = []
        for t in tools:
            for b in (False, True):
                r = cell(agg, t, b, c)
                cells.append("–" if r is None else f"{r.thr:.0f} samples/s, p95 {r.p95:.1f} ms")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return lines


def detail_table(agg: pd.DataFrame) -> list[str]:
    lines = [
        "| Tool | Batching | Concurrency | Throughput [samples/s] | p50 [ms] | p95 [ms] | p99 [ms] | Mean batch "
        "| Time in model | Server CPU [cores] | Busiest client process | Failed requests |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in agg.sort_values(["tool", "batching", "concurrency"]).itertuples():
        lines.append(
            f"| {TOOLS[r.tool]} | {'yes' if r.batching else 'no'} | {r.concurrency} | {pm(r.thr, r.thr_std)} "
            f"| {r.p50:.1f} | {r.p95:.1f} | {r.p99:.1f} | {r.batch:.1f} | {r.model_share:.0%} "
            f"| {r.server_cpu:.2f} | {r.client_cpu:.0%} of a core | {r.failed:.2%} |"
        )
    return lines


def series(agg: pd.DataFrame):
    for tool in TOOLS:
        for batching in (False, True):
            part = agg[(agg.tool == tool) & (agg.batching == batching)].sort_values("concurrency")
            if part.empty:
                continue
            style = {
                "color": COLOR[tool],
                "linestyle": "-" if batching else "--",
                "marker": "o",
                "markerfacecolor": COLOR[tool] if batching else SURFACE,
                "markeredgecolor": COLOR[tool],
                "markeredgewidth": 1.6,
                "label": f"{TOOLS[tool]}, {'batching' if batching else 'no batching'}",
            }
            yield part, style


def concurrency_axis(ax, agg: pd.DataFrame) -> None:
    levels = sorted(agg.concurrency.unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, [str(c) for c in levels])
    ax.minorticks_off()
    ax.set_xlabel("concurrent clients")


def plot_sweep(agg: pd.DataFrame, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for part, style in series(agg):
        axes[0].errorbar(part.concurrency, part.thr, yerr=part.thr_std, capsize=2, **style)
        axes[1].errorbar(part.concurrency, part.p95, yerr=part.p95_std, capsize=2, **style)
    axes[0].set_title("Throughput [samples/s]")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("p95 latency [ms], log scale")
    axes[1].set_yscale("log")
    for ax in axes:
        concurrency_axis(ax, agg)
    axes[0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_knee(agg: pd.DataFrame, path: Path) -> None:
    tools = [t for t in TOOLS if t in set(agg.tool)]
    fig, axes = plt.subplots(1, len(tools), figsize=(6 * len(tools), 4.8), sharex=True, sharey=True, squeeze=False)
    for ax, tool in zip(axes[0], tools):
        for part, style in series(agg[agg.tool == tool]):
            ax.plot(part.thr, part.p95, **style)
            for r in part.itertuples():
                ax.annotate(
                    str(r.concurrency), (r.thr, r.p95), textcoords="offset points", xytext=(6, -3), fontsize=8, color=MUTED
                )
        ax.set_yscale("log")
        ax.set_xlim(left=0)
        ax.set_title(f"{TOOLS[tool]}: throughput vs p95 latency")
        ax.set_xlabel("throughput [samples/s]")
        ax.legend(loc="lower right")
    axes[0][0].set_ylabel("p95 latency [ms], log scale")
    fig.suptitle(
        "Finding the knee: every point is one concurrency level (number next to it)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bottleneck(agg: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for part, style in series(agg):
        axes[0].plot(part.concurrency, part.model_share * 100, **style)
        if part.batching.iloc[0]:
            axes[1].plot(part.concurrency, part.batch, **style)
    axes[0].set_title("Share of wall time spent inside the model [%]")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    axes[1].set_title("Mean batch size the server really formed (log scale)")
    levels = sorted(agg.concurrency.unique())
    axes[1].plot(levels, levels, color=MUTED, linewidth=1, linestyle=":", label="ideal: batch = clients")
    axes[1].set_yscale("log", base=2)
    axes[1].set_yticks(levels, [str(c) for c in levels])
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    for ax in axes:
        concurrency_axis(ax, agg)
    fig.suptitle("Who is the bottleneck?", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    main_df, cpu_df = load("raw.jsonl"), load("raw_cpu.jsonl")
    out: list[str] = ["# Benchmark results", "", "Generated by `benchmark/make_report.py` from `results/raw*.jsonl`.", ""]

    if not main_df.empty:
        agg = aggregate(main_df)
        first = main_df.iloc[0]
        out += [
            f"Device: **{first.device.upper()}**, window {first.duration_s:.0f} s after {first.warmup_s:.0f} s warm-up, "
            f"{int(agg.n.max())} repeats per point (mean ± std), 1 image per request. "
            f"`max_batch_size={first.max_batch_size}`; BentoML `max_latency_ms={main_df.max_latency_ms.max():.0f}`; "
            f"Ray Serve `batch_wait_timeout_s={main_df.batch_wait_ms.max() / 1000:g}`.",
            "",
        ]
        if "on_ac_power" in main_df:
            on_ac = int((main_df.on_ac_power == True).sum())  # noqa: E712
            low_power = int((main_df.low_power_mode == True).sum())  # noqa: E712
            out += [f"Power: {on_ac}/{len(main_df)} measurements on AC power, {low_power} in Low Power Mode.", ""]
        for tool in TOOLS:
            if tool in set(agg.tool):
                out += [f"## {TOOLS[tool]}: without vs with dynamic batching", "", *homework_table(agg, tool), ""]
        out += ["## Tool comparison", "", *comparison_table(agg), ""]
        out += ["## All measurements", "", *detail_table(agg), ""]
        plot_sweep(agg, "Concurrency sweep on MPS (mean ± std of 3 runs)", RESULTS / "sweep.png")
        plot_knee(agg, RESULTS / "knee.png")
        plot_bottleneck(agg, RESULTS / "bottleneck.png")
        out += ["![sweep](sweep.png)", "", "![knee](knee.png)", "", "![bottleneck](bottleneck.png)", ""]

    if not cpu_df.empty:
        agg_cpu = aggregate(cpu_df)
        out += ["## CPU instead of MPS (contrast run)", ""]
        for tool in TOOLS:
            if tool in set(agg_cpu.tool):
                out += [f"### {TOOLS[tool]} on CPU", "", *homework_table(agg_cpu, tool), ""]
        out += [*detail_table(agg_cpu), ""]

    hyst = load("hysteresis.jsonl")
    if not hyst.empty:
        out += [
            "## Does adaptive batching remember past traffic? (`benchmark/hysteresis.py`)",
            "",
            "One server, phases run back to back (30 s windows, 3 s warm-up).",
            "",
            "| Tool | Step | Phase | Clients | Throughput [samples/s] | p50 [ms] | p95 [ms] | Mean batch |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in hyst.itertuples():
            out.append(
                f"| {TOOLS[r.tool]} | {r.step} | {r.phase} | {r.concurrency} | {r.throughput_sps:.0f} "
                f"| {r.latency_p50_ms:.1f} | {r.latency_p95_ms:.1f} | {r.server_mean_batch_size:.1f} |"
            )
        out.append("")

    bottleneck = RESULTS / "bottleneck.json"
    if bottleneck.exists():
        data = json.loads(bottleneck.read_text())
        out += [
            "## Who is the bottleneck? (`benchmark/find_bottleneck.py`, 30 s windows on MPS)",
            "",
            "| Tool | Batching | Clients | Throughput [samples/s] | GPU utilisation | Time in model | Model ms/call "
            "| Mean batch | Server CPU by process [cores] |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in data["under_load"]:
            cpu = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["server_cpu_cores"].items(), key=lambda kv: -kv[1]))
            out.append(
                f"| {TOOLS[r['tool']]} | {'yes' if r['batching'] else 'no'} | {r['concurrency']} "
                f"| {r['throughput_sps']:.0f} | {r['gpu_util_pct']:.0f}% | {r['model_time_share']:.0%} "
                f"| {r['model_ms_per_call']:.2f} | {r['mean_batch_size']:.1f} | {cpu} |"
            )
        c = data["per_request_costs"]
        out += [
            "",
            "Cost of the pieces of one request, measured in-process without HTTP:",
            "",
            "| Step | Time [ms] |",
            "|---|---|",
            f"| JSON parsing | {c['json_parse_ms']:.3f} |",
            f"| base64 decoding | {c['base64_decode_ms']:.3f} |",
            f"| model, batch of 1 (incl. copy to and from the GPU) | {c['model_batch1_ms']:.2f} |",
            f"| model, batch of 64 | {c['model_batch64_ms']:.2f} ({c['model_batch64_ms'] / 64:.3f} per image) |",
            f"| whole `predict` for 1 image | {c['full_predict_batch1_ms']:.2f} |",
            "",
        ]

    (RESULTS / "results.md").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'results.md'}")


if __name__ == "__main__":
    main()
