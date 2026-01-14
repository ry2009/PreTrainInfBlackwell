from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from _path import ROOT
from mhc import MHCConfig, HyperConnectionNet


def composite_amax(model: HyperConnectionNet) -> float:
    mat = model.composite_residual()
    row_sum = mat.abs().sum(dim=-1).max().item()
    col_sum = mat.abs().sum(dim=-2).max().item()
    return max(row_sum, col_sum)


def train_loop(model, steps: int, batch: int, device: torch.device, lr: float = 5e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"step": [], "loss": [], "amax": []}
    d = model.config.d_model
    rng = torch.Generator(device=device).manual_seed(0)
    w_target = torch.randn(d, d, generator=rng, device=device) / (d**0.5)
    b_target = torch.randn(d, generator=rng, device=device) * 0.1

    for step in range(steps):
        x = torch.randn(batch, d, generator=rng, device=device)
        y = torch.tanh(x @ w_target + b_target)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():
                amax = composite_amax(model)
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["amax"].append(float(amax))
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = MHCConfig(d_model=64, streams=4, layers=16, sinkhorn_iters=15)

    hc = HyperConnectionNet(cfg, mhc=False)
    mhc = HyperConnectionNet(cfg, mhc=True)

    hist_hc = train_loop(hc, args.steps, args.batch, device, lr=8e-3)
    hist_mhc = train_loop(mhc, args.steps, args.batch, device, lr=8e-3)

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "hc": hist_hc,
        "mhc": hist_mhc,
        "config": {
            "steps": args.steps,
            "batch": args.batch,
            "d_model": cfg.d_model,
            "streams": cfg.streams,
            "layers": cfg.layers,
        },
    }
    (reports / "mhc_toy.json").write_text(json.dumps(payload, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist_hc["step"], hist_hc["loss"], label="HC")
    axes[0].plot(hist_mhc["step"], hist_mhc["loss"], label="mHC")
    axes[0].set_title("Toy HC vs mHC: Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(hist_hc["step"], hist_hc["amax"], label="HC Amax")
    axes[1].plot(hist_mhc["step"], hist_mhc["amax"], label="mHC Amax")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Composite Residual Gain")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Amax")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(reports / "mhc_toy.png", dpi=160)
    print("Wrote", reports / "mhc_toy.png")


if __name__ == "__main__":
    main()
