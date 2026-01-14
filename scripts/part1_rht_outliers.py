from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _path import ROOT
from nvfp4 import rht_forward, tensor_stats


def main():
    torch.manual_seed(7)
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 64
    x = torch.randn(n) * 0.5
    x[7] = 8.0
    x[23] = -7.0
    x[51] = 6.0

    stats_before = tensor_stats(x)
    x_rht, sign = rht_forward(x, block_size=16)
    stats_after = tensor_stats(x_rht)

    report = {
        "before": stats_before,
        "after": stats_after,
    }

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(x.numpy(), bins=24, color="#2a6d62")
    ax[0].set_title("Original")
    ax[1].hist(x_rht.numpy(), bins=24, color="#9bd650")
    ax[1].set_title("After RHT")
    fig.suptitle("Outlier Redistribution via RHT")
    fig.tight_layout()

    fig.savefig(out_dir / "part1_rht_outliers.png", dpi=160)
    with (out_dir / "part1_rht_outliers.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
