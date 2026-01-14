"""Approximate FP8 E4M3 quantization (for scale factors)."""

from __future__ import annotations

import torch

from .constants import (
    FP8_E4M3_MAX,
    E4M3_MANTISSA_BITS,
    E4M3_MIN_EXP,
    E4M3_MAX_EXP,
    EPS,
)


def quantize_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize to approximate FP8 E4M3 (finite) values.

    This is a reference implementation for scale factor quantization.
    It does not encode actual FP8 bits; it returns float32 values.
    """
    x = x.to(torch.float32)
    sign = torch.sign(x)
    ax = torch.clamp(x.abs(), min=0.0)

    # Clamp to max representable
    ax = torch.clamp(ax, max=FP8_E4M3_MAX)

    # Handle zeros
    is_zero = ax < EPS

    # Normal range threshold (min normal)
    min_normal = 2.0 ** E4M3_MIN_EXP

    # Subnormals (no implicit 1)
    sub_mask = (ax < min_normal) & (~is_zero)

    # Compute exponent for normals
    log2 = torch.log2(ax + EPS)
    exp = torch.floor(log2)
    exp = torch.clamp(exp, E4M3_MIN_EXP, E4M3_MAX_EXP)

    # Mantissa for normals
    mant = ax / (2.0 ** exp) - 1.0
    mant_q = torch.round(mant * (2 ** E4M3_MANTISSA_BITS)) / (2 ** E4M3_MANTISSA_BITS)

    # Handle mantissa overflow (carry)
    overflow = mant_q >= 1.0
    exp = torch.where(overflow, torch.clamp(exp + 1, E4M3_MIN_EXP, E4M3_MAX_EXP), exp)
    mant_q = torch.where(overflow, torch.zeros_like(mant_q), mant_q)

    normal_val = (2.0 ** exp) * (1.0 + mant_q)

    # Subnormal quantization
    sub_mant = torch.round(ax / min_normal * (2 ** E4M3_MANTISSA_BITS)) / (2 ** E4M3_MANTISSA_BITS)
    sub_val = min_normal * sub_mant

    out = torch.where(sub_mask, sub_val, normal_val)
    out = torch.where(is_zero, torch.zeros_like(out), out)
    out = torch.clamp(out, max=FP8_E4M3_MAX)

    return out * sign
