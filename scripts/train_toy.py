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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--precision", choices=["nvfp4", "bf16"], default="nvfp4")
    parser.add_argument("--cooldown-frac", type=float, default=0.0, help="Fraction of steps to run in BF16 at end")
    parser.add_argument("--global-amax", choices=["max", "percentile"], default="max")
    parser.add_argument("--pct", type=float, default=0.999)
    parser.add_argument("--sr", action="store_true")
    parser.add_argument("--rht", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = build_data(1536, 256, 64, device, scale=args.scale, noise=args.noise, standardize=not args.no_standardize)

    cfg = NVFP4Config(
        stochastic_rounding=args.sr,
        rht=args.rht,
        global_amax_mode=args.global_amax,
        percentile=args.pct,
    )

    l1 = NVFP4Linear(256, args.hidden, cfg=cfg).to(device)
    l2 = NVFP4Linear(args.hidden, 64, cfg=cfg).to(device)

    # Simple SGD
    params = list(l1.parameters()) + list(l2.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr)

    losses = []
    cooldown_start = int(args.steps * (1.0 - args.cooldown_frac))

    for step in trange(args.steps):
        optim.zero_grad(set_to_none=True)

        quantize = args.precision == "nvfp4" and (step < cooldown_start)

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

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, label="train")
    if args.cooldown_frac > 0:
        ax.axvline(cooldown_start, color="#9bd650", linestyle="--", label="cooldown")
    ax.set_title("Toy NVFP4 Training")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_toy_loss.png", dpi=160)

    print(f"Final loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
