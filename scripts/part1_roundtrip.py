from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _path import ROOT
from nvfp4 import nvfp4_quantize, nvfp4_dequantize, error_metrics


def main():
    torch.manual_seed(42)
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy‑tailed synthetic data
    x = torch.randn(512, 512) * 0.6
    x.view(-1)[::4096] *= 10.0

    configs = [
        {"name": "rtne", "rht": False, "sr": False},
        {"name": "rht_rtne", "rht": True, "sr": False},
        {"name": "rht_sr", "rht": True, "sr": True},
    ]

    all_metrics = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    for cfg in configs:
        rng = torch.Generator().manual_seed(123) if cfg["sr"] else None
        q = nvfp4_quantize(
            x,
            block_size=16,
            stochastic_rounding=cfg["sr"],
            rng=rng,
            rht=cfg["rht"],
        )
        x_hat = nvfp4_dequantize(q, apply_inverse_rht=cfg["rht"])
        metrics = error_metrics(x, x_hat)
        all_metrics[cfg["name"]] = metrics

        err = (x - x_hat).abs().flatten().numpy()
        ax.hist(err, bins=120, alpha=0.4, label=cfg["name"], density=True)

    ax.set_title("NVFP4 Round‑Trip Absolute Error")
    ax.set_xlabel("|x - x̂|")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "part1_roundtrip.png", dpi=160)

    with (out_dir / "part1_roundtrip.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
