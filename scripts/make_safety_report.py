from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from _path import ROOT


def main(path: str = "reports/safety_suite.json"):
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(path).read_text())

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "lines.linewidth": 2.4,
            "lines.markersize": 5,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )

    metrics = data["metrics"]
    labels = ["input_only", "output_only", "exchange", "probe", "cascade"]
    miss_rates = [metrics[k]["miss_rate"] for k in labels]
    colors = ["#9E9E9E", "#6C8EBF", "#3D9970", "#B07AA1", "#E07A5F"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, miss_rates, color=colors)
    ax.set_ylabel("Attack miss rate (↓)")
    ax.set_title("Exchange‑Aware Classifiers Reduce Misses")
    fig.tight_layout()
    fig.savefig(out_dir / "safety_exchange.png", dpi=180)
    plt.close(fig)

    pareto = data["pareto"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([p["cost"] for p in pareto], [p["miss_rate"] for p in pareto], color="#E07A5F")
    ax.scatter(
        [metrics["exchange"]["fpr"]],
        [metrics["exchange"]["miss_rate"]],
        color="#3D9970",
        label="exchange",
    )
    ax.set_xlabel("Escalation fraction (cost)")
    ax.set_ylabel("Attack miss rate (↓)")
    ax.set_title("Probe Cascade Pareto")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "safety_pareto.png", dpi=180)
    plt.close(fig)

    latency = data["latency_ms"]
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["input_only", "output_only", "exchange", "cascade"]
    ax.bar(keys, [latency[k] for k in keys], color=["#9E9E9E", "#6C8EBF", "#3D9970", "#E07A5F"])
    ax.set_ylabel("Latency per batch (ms)")
    ax.set_title("Safety Latency Cost")
    fig.tight_layout()
    fig.savefig(out_dir / "safety_latency.png", dpi=180)
    plt.close(fig)

    # Simple poster
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(labels, miss_rates, color=colors)
    ax1.set_title("Miss Rate @ Fixed FPR")
    ax1.set_ylabel("Miss rate (↓)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([p["cost"] for p in pareto], [p["miss_rate"] for p in pareto], color="#E07A5F")
    ax2.set_xlabel("Escalation cost")
    ax2.set_ylabel("Miss rate (↓)")
    ax2.set_title("Cascade Pareto")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(keys, [latency[k] for k in keys], color=["#9E9E9E", "#6C8EBF", "#3D9970", "#E07A5F"])
    ax3.set_title("Latency per Batch")
    ax3.set_ylabel("ms")

    ax4 = fig.add_subplot(gs[1, 1])
    throughput = data["throughput"]
    ax4.bar(keys, [throughput[k] for k in keys], color=["#9E9E9E", "#6C8EBF", "#3D9970", "#E07A5F"])
    ax4.set_title("Throughput")
    ax4.set_ylabel("Tokens/s")

    fig.suptitle("Safety‑Aware Inference: Probe‑Cascade vs Exchange", fontsize=15, y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_poster.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
