from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _path import ROOT


def load_img(path: Path):
    if path.exists():
        return plt.imread(path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    reports = ROOT / "reports"
    prefix = args.prefix

    def pref(name: str, fallback: str):
        path = reports / f"{prefix}_{name}" if prefix else reports / fallback
        if not path.exists():
            path = reports / fallback
        return load_img(path)

    imgs = {
        "arch": load_img(reports / "arch_hybrid.png"),
        "arch_engram": load_img(reports / "arch_engram.png"),
        "arch_mhc": load_img(reports / "arch_mhc.png"),
        "loss": pref("loss.png", "hybrid_loss.png"),
        "throughput": pref("throughput.png", "hybrid_throughput.png"),
        "table": pref("table.png", "hybrid_table.png"),
        "alloc": pref("alloc.png", "hybrid_alloc.png"),
        "memory": pref("memory.png", "hybrid_memory.png"),
        "long": pref("long_context.png", "hybrid_long_context.png"),
        "stability": pref("stability.png", "hybrid_stability.png"),
        "ablation": pref("ablation.png", "hybrid_ablation.png"),
        "bench": load_img(reports / "part2_benchmark.png"),
        "sweep": load_img(reports / "part2_sweep.png"),
        "roundtrip": load_img(reports / "part1_roundtrip.png"),
        "part1_ablation": load_img(reports / "part1_ablation_report.png"),
        "engram_gate": pref("gate.png", "hybrid_gate.png"),
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "lines.linewidth": 2.0,
            "font.family": "DejaVu Sans",
        }
    )

    fig = plt.figure(figsize=(16, 30))
    gs = fig.add_gridspec(7, 3, height_ratios=[0.15, 1, 1, 1, 1, 1, 1])

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.01, 0.7, "Hybrid Efficient LM: NVFP4 + Engram + mHC + MoE", fontsize=20, fontweight="bold")
    ax_title.text(
        0.01,
        0.2,
        "GPU‑verified on B200. Dataset: CodeParrot (byte‑level). Metrics: loss, throughput, stability, allocation.",
        fontsize=11,
    )

    panel_defs = [
        (1, 0, imgs["arch"], "Architecture"),
        (1, 1, imgs["loss"], "Pretraining Loss"),
        (1, 2, imgs["throughput"], "Training Throughput"),
        (2, 0, imgs["table"], "Comparison Table"),
        (2, 1, imgs["alloc"], "Sparsity Allocation"),
        (2, 2, imgs["memory"], "Engram Memory Scaling"),
        (3, 0, imgs["long"], "Long‑Context Latency"),
        (3, 1, imgs["stability"], "Stability (HC vs mHC)"),
        (3, 2, imgs["engram_gate"], "Engram Gate Trace"),
        (4, 0, imgs["bench"], "NVFP4 GEMM Benchmark"),
        (4, 1, imgs["sweep"], "NVFP4 Crossover"),
        (4, 2, imgs["ablation"], "Hybrid Ablation Matrix"),
        (5, 0, imgs["arch_engram"], "Engram Detail"),
        (5, 1, imgs["arch_mhc"], "mHC Detail"),
        (5, 2, imgs["roundtrip"], "NVFP4 Round‑Trip"),
    ]

    for row, col, img, title in panel_defs:
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    out_name = f"{prefix}_POSTER.png" if prefix else "POSTER.png"
    fig.savefig(reports / out_name, dpi=180)
    print("Wrote", reports / out_name)


if __name__ == "__main__":
    main()
