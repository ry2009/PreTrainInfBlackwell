from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _path import ROOT
from nvfp4 import nvfp4_quantize, nvfp4_dequantize, error_metrics


def main():
    torch.manual_seed(123)
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.randn(256, 256) * 0.5
    x.view(-1)[::2048] *= 12.0

    configs = [
        {"name": "max_rtne", "rht": False, "sr": False, "global": "max"},
        {"name": "max_rht_rtne", "rht": True, "sr": False, "global": "max"},
        {"name": "max_rht_sr", "rht": True, "sr": True, "global": "max"},
        {"name": "p999_rtne", "rht": False, "sr": False, "global": "percentile"},
        {"name": "p999_rht_rtne", "rht": True, "sr": False, "global": "percentile"},
        {"name": "p999_rht_sr", "rht": True, "sr": True, "global": "percentile"},
    ]

    results = {}
    mse_vals = []
    labels = []

    for cfg in configs:
        rng = torch.Generator().manual_seed(321) if cfg["sr"] else None
        q = nvfp4_quantize(
            x,
            block_size=16,
            stochastic_rounding=cfg["sr"],
            rng=rng,
            rht=cfg["rht"],
            global_amax_mode=cfg["global"],
            percentile=0.999,
        )
        x_hat = nvfp4_dequantize(q, apply_inverse_rht=cfg["rht"])
        metrics = error_metrics(x, x_hat)
        results[cfg["name"]] = metrics
        labels.append(cfg["name"])
        mse_vals.append(metrics["mse"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, mse_vals, color="#2a6d62")
    ax.set_ylabel("MSE")
    ax.set_title("NVFP4 Ablations (Lower is Better)")
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "part1_ablation_report.png", dpi=160)

    with (out_dir / "part1_ablation_report.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
