from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import trange

from _path import ROOT
from nvfp4.nn import NVFP4Linear, NVFP4Config


def build_data(
    n_samples: int,
    in_dim: int,
    out_dim: int,
    device: str,
    scale: float = 0.35,
    noise: float = 0.02,
    standardize: bool = True,
):
    torch.manual_seed(0)
    X = torch.randn(n_samples, in_dim, device=device) * scale
    true_w = torch.randn(out_dim, in_dim, device=device) / (in_dim ** 0.5)
    y = X @ true_w.t() + noise * torch.randn(n_samples, out_dim, device=device)
    if standardize:
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-6)
    return X, y


def run_trial(
    label: str,
    steps: int,
    cooldown_frac: float,
    cfg: NVFP4Config,
    device: str,
    scale: float,
    noise: float,
    standardize: bool,
):
    X, y = build_data(1536, 256, 64, device, scale=scale, noise=noise, standardize=standardize)
    l1 = NVFP4Linear(256, 128, cfg=cfg).to(device)
    l2 = NVFP4Linear(128, 64, cfg=cfg).to(device)
    optim = torch.optim.AdamW(list(l1.parameters()) + list(l2.parameters()), lr=3e-3)

    losses = []
    cooldown_start = int(steps * (1.0 - cooldown_frac))

    for step in trange(steps, desc=label):
        optim.zero_grad(set_to_none=True)
        quantize = step < cooldown_start

        if quantize:
            h = l1(X)
            h = F.relu(h)
            out = l2(h)
        else:
            h = F.linear(X, l1.weight, l1.bias)
            h = F.relu(h)
            out = F.linear(h, l2.weight, l2.bias)

        loss = F.mse_loss(out, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--cooldown-frac", type=float, default=0.15)
    parser.add_argument("--pct", type=float, default=0.999)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_cfg = NVFP4Config(global_amax_mode="max", stochastic_rounding=False)
    twist_cfg = NVFP4Config(global_amax_mode="percentile", percentile=args.pct, stochastic_rounding=True)

    losses_base = run_trial(
        "baseline",
        args.steps,
        0.0,
        baseline_cfg,
        device,
        scale=args.scale,
        noise=args.noise,
        standardize=not args.no_standardize,
    )
    losses_twist = run_trial(
        "twist",
        args.steps,
        args.cooldown_frac,
        twist_cfg,
        device,
        scale=args.scale,
        noise=args.noise,
        standardize=not args.no_standardize,
    )

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses_base, label="baseline (max + RTNE)")
    ax.plot(losses_twist, label="twist (percentile + SR + cooldown)")
    ax.axvline(int(args.steps * (1.0 - args.cooldown_frac)), color="#9bd650", linestyle="--", label="cooldown")
    ax.set_title("Twist Experiment: Adaptive Scaling + Cooldown")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "twist_adaptive_scaling.png", dpi=160)

    print(f"Baseline final loss: {losses_base[-1]:.6f}")
    print(f"Twist final loss: {losses_twist[-1]:.6f}")


if __name__ == "__main__":
    main()
