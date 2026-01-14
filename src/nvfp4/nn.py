"""Toy NVFP4-aware modules for demo/training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .nvfp4 import nvfp4_quantize, nvfp4_dequantize


GlobalAmaxMode = Literal["max", "percentile"]


@dataclass
class NVFP4Config:
    block_size: int = 16
    weight_block_shape: tuple[int, int] | None = (16, 16)
    stochastic_rounding: bool = False
    rht: bool = False
    global_amax_mode: GlobalAmaxMode = "max"
    percentile: float = 0.999


class NVFP4Linear(nn.Module):
    """Reference linear layer that simulates NVFP4 quantization (slow, CPU‑friendly)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, cfg: NVFP4Config | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.cfg = cfg or NVFP4Config()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        rng = None
        qx = nvfp4_quantize(
            x,
            block_size=cfg.block_size,
            stochastic_rounding=cfg.stochastic_rounding,
            rng=rng,
            rht=cfg.rht,
            global_amax_mode=cfg.global_amax_mode,
            percentile=cfg.percentile,
        )
        x_hat = nvfp4_dequantize(qx, apply_inverse_rht=cfg.rht)
        # Straight‑through estimator for activations
        x_ste = x + (x_hat - x).detach()

        qw = nvfp4_quantize(
            self.weight,
            block_size=cfg.block_size,
            block_shape=cfg.weight_block_shape,
            stochastic_rounding=False,
            rht=False,
            global_amax_mode=cfg.global_amax_mode,
            percentile=cfg.percentile,
        )
        w_hat = nvfp4_dequantize(qw)
        # Straight‑through estimator for weights
        w_ste = self.weight + (w_hat - self.weight).detach()

        y = x_ste @ w_ste.t()
        if self.bias is not None:
            y = y + self.bias
        return y
