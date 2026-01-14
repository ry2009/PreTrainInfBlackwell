"""FP4 E2M1 quantization utilities."""

from __future__ import annotations

import torch

from .constants import FP4_E2M1_MAX, FP4_VALUES


def quantize_fp4_rtne(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest FP4 E2M1 value using RTNE-style thresholds.

    Expects input already scaled + clamped to [-6, 6].
    Returns float tensor with FP4 representable values.
    """
    abs_x = x.abs()
    q = torch.zeros_like(abs_x)

    # Positive bins (RTNE thresholds)
    q = torch.where((abs_x > 0.25) & (abs_x < 0.75), torch.tensor(0.5, device=x.device, dtype=x.dtype), q)
    q = torch.where((abs_x >= 0.75) & (abs_x <= 1.25), torch.tensor(1.0, device=x.device, dtype=x.dtype), q)
    q = torch.where((abs_x > 1.25) & (abs_x < 1.75), torch.tensor(1.5, device=x.device, dtype=x.dtype), q)
    q = torch.where((abs_x >= 1.75) & (abs_x <= 2.5), torch.tensor(2.0, device=x.device, dtype=x.dtype), q)
    q = torch.where((abs_x > 2.5) & (abs_x < 3.5), torch.tensor(3.0, device=x.device, dtype=x.dtype), q)
    q = torch.where((abs_x >= 3.5) & (abs_x <= 5.0), torch.tensor(4.0, device=x.device, dtype=x.dtype), q)
    q = torch.where(abs_x > 5.0, torch.tensor(6.0, device=x.device, dtype=x.dtype), q)

    # Restore sign (keeps +0 / -0 where applicable)
    return q * torch.sign(x)


def quantize_fp4_sr(x: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
    """Stochastic rounding to FP4 E2M1 values.

    Expects input already scaled + clamped to [-6, 6].
    """
    fp4_vals = FP4_VALUES.to(device=x.device, dtype=x.dtype)
    flat = x.reshape(-1, 1)

    # Distances to all FP4 values
    dist = (flat - fp4_vals.reshape(1, -1)).abs()

    # Two nearest neighbors per value
    nearest_dist, nearest_idx = dist.topk(k=2, largest=False)

    # Probability: inverse distance weighting
    d0 = nearest_dist[:, 0]
    d1 = nearest_dist[:, 1]
    denom = d0 + d1
    denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)

    p0 = d1 / denom_safe
    p1 = d0 / denom_safe
    probs = torch.stack([p0, p1], dim=1)

    # If exactly on a representable value, pick it deterministically
    exact_mask = denom == 0
    if exact_mask.any():
        probs[exact_mask, 0] = 1.0
        probs[exact_mask, 1] = 0.0

    # Sample
    choices = torch.multinomial(probs, num_samples=1, generator=rng)
    selected = nearest_idx.gather(1, choices).squeeze(1)

    out = fp4_vals[selected].reshape(x.shape)
    return out


def clamp_fp4_range(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -FP4_E2M1_MAX, FP4_E2M1_MAX)
