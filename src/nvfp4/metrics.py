"""Metrics for quantization error and tensor stats."""

from __future__ import annotations

import torch


def tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().float()
    abs_x = x.abs()
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "mean_abs": abs_x.mean().item(),
        "max_abs": abs_x.max().item(),
        "max_over_mean_abs": (abs_x.max() / (abs_x.mean() + 1e-12)).item(),
    }


def error_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> dict:
    x = x.detach().float()
    x_hat = x_hat.detach().float()
    diff = x - x_hat
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(diff.abs()).item()
    max_abs = diff.abs().max().item()
    rel_l2 = (diff.norm() / (x.norm() + 1e-12)).item()
    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
    }
