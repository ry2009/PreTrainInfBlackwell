from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _path import ROOT


def load_img(path: Path):
    if not path.exists():
        return None
    return plt.imread(path)


def main():
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    bench = json.loads((reports / "part2_benchmark.json").read_text())
    sweep = json.loads((reports / "part2_sweep.json").read_text()) if (reports / "part2_sweep.json").exists() else []

    img_bench = load_img(reports / "part2_benchmark.png")
    img_sweep = load_img(reports / "part2_sweep.png")
    img_roundtrip = load_img(reports / "part1_roundtrip.png")
    img_ablation = load_img(reports / "part1_ablation_report.png")

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.25, 1, 1])

    # Title block
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    title = "NVFP4 Pretraining Showcase"
    subtitle = (
        f"GEMM 8192³ speedup: {bench['speedup']:.2f}× (NVFP4 {bench['nvfp4_ms']:.3f} ms vs BF16 {bench['bf16_ms']:.3f} ms)"
    )
    ax_title.text(0.01, 0.7, title, fontsize=20, fontweight="bold")
    ax_title.text(0.01, 0.2, subtitle, fontsize=12)

    # Row 1
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    for ax, img, label in [
        (ax1, img_bench, "Part 2: NVFP4 vs BF16 GEMM"),
        (ax2, img_sweep, "Twist: NVFP4 Crossover vs Size"),
    ]:
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        ax.set_title(label, fontsize=11)

    # Row 2
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    for ax, img, label in [
        (ax3, img_roundtrip, "Part 1: Round‑Trip Error"),
        (ax4, img_ablation, "Part 1: Ablations"),
    ]:
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        ax.set_title(label, fontsize=11)

    fig.tight_layout()
    fig.savefig(reports / "SHOWCASE.png", dpi=160)

    # Markdown summary
    md = reports / "SHOWCASE.md"
    lines = [
        "# NVFP4 Pretraining Showcase",
        "",
        f"**GEMM 8192³ speedup:** `{bench['speedup']:.2f}×` (NVFP4 {bench['nvfp4_ms']:.3f} ms vs BF16 {bench['bf16_ms']:.3f} ms)",
        "",
        "## Highlights",
        "- Part 1: NVFP4 quantization, RHT, SR, and ablations (see plots).",
        "- Part 2: TransformerEngine NVFP4 on B200 with real benchmarks.",
        "- Twist: Crossover curve shows when NVFP4 wins vs BF16.",
        "",
        "![Showcase](SHOWCASE.png)",
    ]
    md.write_text("\n".join(lines))

    print("Wrote", md)


if __name__ == "__main__":
    main()
