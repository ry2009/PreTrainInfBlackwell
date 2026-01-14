from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _path import ROOT
from nvfp4 import nvfp4_quantize, nvfp4_dequantize, error_metrics, tensor_stats


def should_apply_rht(stats: dict, threshold: float = 6.0) -> bool:
    return stats["max_over_mean_abs"] >= threshold


def evaluate(name: str, x: torch.Tensor, rht: bool, sr: bool):
    rng = torch.Generator().manual_seed(123) if sr else None
    q = nvfp4_quantize(
        x,
        block_size=16,
        stochastic_rounding=sr,
        rng=rng,
        rht=rht,
    )
    x_hat = nvfp4_dequantize(q, apply_inverse_rht=rht)
    return error_metrics(x, x_hat)


def main():
    torch.manual_seed(0)
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Three regimes: normal, heavy-tail, mixed
    normal = torch.randn(512, 512) * 0.5
    heavy = torch.randn(512, 512) * 0.5
    heavy.view(-1)[::2048] *= 12.0
    mixed = torch.cat([normal[:256], heavy[256:]], dim=0)

    tensors = {
        "normal": normal,
        "heavy_tail": heavy,
        "mixed": mixed,
    }

    report = {}
    mse_baseline = []
    mse_always_rht = []
    mse_router = []

    for name, x in tensors.items():
        stats = tensor_stats(x)
        baseline = evaluate(name, x, rht=False, sr=False)
        always_rht = evaluate(name, x, rht=True, sr=True)
        use_rht = should_apply_rht(stats, threshold=6.0)
        routed = evaluate(name, x, rht=use_rht, sr=use_rht)

        report[name] = {
            "stats": stats,
            "baseline": baseline,
            "always_rht_sr": always_rht,
            "router": {"use_rht": use_rht, **routed},
        }

        mse_baseline.append(baseline["mse"])
        mse_always_rht.append(always_rht["mse"])
        mse_router.append(routed["mse"])

    # Plot
    labels = list(tensors.keys())
    x_idx = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x_idx], mse_baseline, width, label="Baseline (RTNE)")
    ax.bar([i for i in x_idx], mse_always_rht, width, label="Always RHT+SR")
    ax.bar([i + width for i in x_idx], mse_router, width, label="Outlier Router")

    ax.set_xticks(list(x_idx))
    ax.set_xticklabels(labels)
    ax.set_ylabel("MSE")
    ax.set_title("Twist: Outlier‑Aware RHT Router")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "twist_outlier_router.png", dpi=160)

    with (out_dir / "twist_outlier_router.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
