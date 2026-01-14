from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _path import ROOT


def box(ax, xy, text, w=0.22, h=0.08, color="#e8f0f0"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1,
        edgecolor="#2f3b3b",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    return (x, y, w, h)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, lw=1.2, color="#34495e"))


def draw_hybrid(path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    b1 = box(ax, (0.05, 0.85), "Input IDs", w=0.18)
    b2 = box(ax, (0.30, 0.85), "Embedding", w=0.18)
    b3 = box(ax, (0.55, 0.85), "Engram\nLookup", w=0.18, color="#e7f4ff")
    b4 = box(ax, (0.05, 0.65), "mHC\nStreams", w=0.18, color="#f6e8ff")
    b5 = box(ax, (0.30, 0.65), "Attention", w=0.18)
    b6 = box(ax, (0.55, 0.65), "MoE FFN\n(NVFP4)", w=0.18, color="#fff3e6")
    b7 = box(ax, (0.30, 0.45), "Output", w=0.18)

    arrow(ax, (0.23, 0.89), (0.30, 0.89))
    arrow(ax, (0.48, 0.89), (0.55, 0.89))
    arrow(ax, (0.39, 0.85), (0.14, 0.69))
    arrow(ax, (0.64, 0.85), (0.64, 0.73))
    arrow(ax, (0.23, 0.69), (0.30, 0.69))
    arrow(ax, (0.48, 0.69), (0.55, 0.69))
    arrow(ax, (0.39, 0.65), (0.39, 0.53))

    ax.text(0.05, 0.32, "Hybrid block: Engram + mHC + MoE + NVFP4", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_engram(path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    b1 = box(ax, (0.05, 0.6), "Tokens", w=0.18)
    b2 = box(ax, (0.30, 0.6), "N‑gram\nHash", w=0.18, color="#e7f4ff")
    b3 = box(ax, (0.55, 0.6), "Embedding\nLookup", w=0.18, color="#e7f4ff")
    b4 = box(ax, (0.30, 0.35), "Context\nGate", w=0.18, color="#f6e8ff")
    b5 = box(ax, (0.55, 0.35), "Conv +\nResidual", w=0.18, color="#fef6e7")
    arrow(ax, (0.23, 0.64), (0.30, 0.64))
    arrow(ax, (0.48, 0.64), (0.55, 0.64))
    arrow(ax, (0.64, 0.6), (0.64, 0.45))
    arrow(ax, (0.39, 0.6), (0.39, 0.45))
    arrow(ax, (0.48, 0.39), (0.55, 0.39))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_mhc(path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    b1 = box(ax, (0.05, 0.6), "Streams", w=0.18, color="#f6e8ff")
    b2 = box(ax, (0.30, 0.6), "H_pre\n(mix)", w=0.18)
    b3 = box(ax, (0.55, 0.6), "Layer F", w=0.18)
    b4 = box(ax, (0.30, 0.35), "H_post", w=0.18)
    b5 = box(ax, (0.55, 0.35), "H_res\n(Sinkhorn)", w=0.18, color="#f6e8ff")
    arrow(ax, (0.23, 0.64), (0.30, 0.64))
    arrow(ax, (0.48, 0.64), (0.55, 0.64))
    arrow(ax, (0.64, 0.6), (0.64, 0.45))
    arrow(ax, (0.39, 0.6), (0.39, 0.45))
    arrow(ax, (0.48, 0.39), (0.55, 0.39))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    draw_hybrid(reports / "arch_hybrid.png")
    draw_engram(reports / "arch_engram.png")
    draw_mhc(reports / "arch_mhc.png")
    print("Wrote architecture diagrams")


if __name__ == "__main__":
    main()
