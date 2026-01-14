from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from _path import ROOT


PALETTE = {
    "dense": "#222222",
    "moe": "#4C78A8",
    "engram": "#59A14F",
    "hc": "#E15759",
    "hybrid_nvfp4": "#B07AA1",
}


def style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def load_json(path: Path):
    return json.loads(path.read_text())


def prefix_path(reports: Path, prefix: str, name: str) -> Path:
    if prefix:
        return reports / f"{prefix}_{name}"
    return reports / name


def plot_loss(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, res in data["results"].items():
        hist = res["history"]
        ax.plot(
            hist["step"],
            hist["val_loss"],
            label=name,
            color=PALETTE.get(name, "#888888"),
            marker="o",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Hybrid Suite: Val Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_throughput(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(data["results"].keys())
    tps = [data["results"][n]["tokens_per_s"] for n in names]
    colors = [PALETTE.get(n, "#888888") for n in names]
    ax.bar(names, tps, color=colors)
    ax.set_ylabel("Tokens/s")
    ax.set_title("Training Throughput")
    for i, val in enumerate(tps):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_alloc(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    rho = [r["rho"] for r in data["alloc_results"]]
    loss = [r["val_loss"] for r in data["alloc_results"]]
    ax.plot(rho, loss, marker="o", color="#4C78A8")
    ax.set_xlabel("Allocation ratio (rho)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Sparsity Allocation Sweep")
    best_idx = min(range(len(loss)), key=lambda i: loss[i])
    ax.scatter([rho[best_idx]], [loss[best_idx]], color="#E15759", s=50, zorder=5)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

def plot_memory(data: dict, out: Path):
    mem = data.get("memory_results", [])
    if not mem:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    buckets = [r["buckets"] for r in mem]
    loss = [r["val_loss"] for r in mem]
    ax.plot(buckets, loss, marker="o", color="#59A14F")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Engram buckets (log scale)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Engram Memory Scaling")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_long_context(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    long_ctx = data.get("long_context", {})
    if isinstance(long_ctx, dict):
        for name, entries in long_ctx.items():
            seqs = [r["seq_len"] for r in entries]
            lat = [r["latency_ms"] for r in entries]
            ax.plot(seqs, lat, marker="o", label=name, color=PALETTE.get(name))
        ax.legend()
        ax.set_title("Long‑Context Latency (Dense vs Hybrid)")
    else:
        seqs = [r["seq_len"] for r in long_ctx]
        lat = [r["latency_ms"] for r in long_ctx]
        ax.plot(seqs, lat, marker="o", color="#4C78A8")
        ax.set_title("Long‑Context Latency")
    ax.set_xlabel("Seq length")
    ax.set_ylabel("Latency (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_gate(data: dict, out: Path):
    gate_series = []
    for name in ("hybrid_nvfp4", "engram", "hc"):
        sample = data["results"].get(name, {}).get("gates_sample")
        if sample:
            gate_series.append((name, sample))
    if not gate_series:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    for name, series in gate_series:
        ax.plot(series, label=name, color=PALETTE.get(name))
    ax.set_xlabel("Token index")
    ax.set_ylabel("Gate α")
    ax.set_title("Engram Gate Activation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_ablation(data: dict, out: Path):
    ablation = data.get("ablation_results", {})
    if not ablation:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(ablation.keys())
    vals = [
        ablation[k]["history"]["val_loss"][-1]
        if ablation[k]["history"]["val_loss"]
        else None
        for k in labels
    ]
    ax.bar(labels, vals, color="#4C78A8")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Hybrid Ablation Matrix")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_stability(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ["hc", "hybrid_nvfp4"]:
        hist = data["results"][name]["history"]
        ax.plot(hist["step"], hist["grad_norm"], label=f"{name} grad norm", color=PALETTE.get(name))
    ax.set_xlabel("Step")
    ax.set_ylabel("Grad Norm")
    ax.set_title("Stability: HC vs mHC+NVFP4")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_table(data: dict, out: Path, out_md: Path):
    rows = []
    for name, res in data["results"].items():
        hist = res["history"]
        val_loss = hist["val_loss"][-1] if hist["val_loss"] else None
        rows.append(
            {
                "Model": name,
                "Params": res["params"],
                "ValLoss": val_loss,
                "PPL": math.exp(val_loss) if val_loss else None,
                "Tokens/s": res["tokens_per_s"],
                "Infer ms": res["infer_ms"],
                "Peak GB": res["peak_mem_gb"],
                "Amax": hist["amax"][-1] if hist["amax"] else None,
                "Bigram Acc": res.get("bigram_acc"),
                "Gate Mean": hist["gate_mean"][-1] if hist["gate_mean"] else None,
            }
        )

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        out_md.write_text(df.to_markdown(index=False))

        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.axis("off")
        tbl = ax.table(
            cellText=df.round(4).values,
            colLabels=df.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="reports/hybrid_suite.json")
    parser.add_argument("--prefix", default="FINAL")
    args = parser.parse_args()

    reports = ROOT / "reports"
    data = load_json(reports / args.json)

    style()

    plot_loss(data, prefix_path(reports, args.prefix, "loss.png"))
    plot_throughput(data, prefix_path(reports, args.prefix, "throughput.png"))
    plot_alloc(data, prefix_path(reports, args.prefix, "alloc.png"))
    plot_memory(data, prefix_path(reports, args.prefix, "memory.png"))
    plot_long_context(data, prefix_path(reports, args.prefix, "long_context.png"))
    plot_gate(data, prefix_path(reports, args.prefix, "gate.png"))
    plot_ablation(data, prefix_path(reports, args.prefix, "ablation.png"))
    plot_stability(data, prefix_path(reports, args.prefix, "stability.png"))
    plot_table(
        data,
        prefix_path(reports, args.prefix, "table.png"),
        prefix_path(reports, args.prefix, "table.md"),
    )

    print("Wrote FINAL report assets with prefix", args.prefix)


if __name__ == "__main__":
    main()
