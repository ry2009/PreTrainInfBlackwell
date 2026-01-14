"""Reference NVFP4 quantization pipeline (global + block scaling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .constants import FP4_E2M1_MAX, FP8_E4M3_MAX, NVFP4_BLOCK_SIZE, EPS
from .fp4 import quantize_fp4_rtne, quantize_fp4_sr, clamp_fp4_range
from .fp8 import quantize_fp8_e4m3
from .hadamard import rht_forward, rht_inverse


GlobalAmaxMode = Literal["max", "percentile"]


@dataclass
class NVFP4Quantized:
    q: torch.Tensor
    encode_scales: torch.Tensor
    decode_scales_fp8: torch.Tensor
    global_amax: torch.Tensor
    global_encode_scale: torch.Tensor
    global_decode_scale: torch.Tensor
    pad: int | tuple[int, int]
    block_size: int
    block_shape: tuple[int, int] | None
    rht_sign: torch.Tensor | None


def compute_global_amax(x: torch.Tensor, mode: GlobalAmaxMode = "max", percentile: float = 0.999) -> torch.Tensor:
    abs_x = x.abs().float().flatten()
    if mode == "max":
        return abs_x.max()
    if mode == "percentile":
        return torch.quantile(abs_x, percentile)
    raise ValueError(f"Unknown global_amax mode: {mode}")


def _blockwise_view_1d(x: torch.Tensor, block_size: int):
    pad = (block_size - (x.shape[-1] % block_size)) % block_size
    if pad:
        x = F.pad(x, (0, pad))
    view = x.reshape(*x.shape[:-1], -1, block_size)
    return view, pad


def _blockwise_view_2d(x: torch.Tensor, block_shape: tuple[int, int]):
    if x.dim() != 2:
        raise ValueError("2D block scaling expects a 2D tensor.")
    br, bc = block_shape
    pad_r = (br - (x.shape[0] % br)) % br
    pad_c = (bc - (x.shape[1] % bc)) % bc
    if pad_r or pad_c:
        x = F.pad(x, (0, pad_c, 0, pad_r))
    m_blocks = x.shape[0] // br
    n_blocks = x.shape[1] // bc
    view = x.view(m_blocks, br, n_blocks, bc)
    return view, (pad_r, pad_c)


def nvfp4_quantize(
    x: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    block_shape: tuple[int, int] | None = None,
    stochastic_rounding: bool = False,
    rng: torch.Generator | None = None,
    rht: bool = False,
    rht_sign: torch.Tensor | None = None,
    global_amax_mode: GlobalAmaxMode = "max",
    percentile: float = 0.999,
) -> NVFP4Quantized:
    """Quantize to NVFP4 with two‑level scaling.

    Returns an NVFP4Quantized object containing q (float values in FP4 domain),
    scales, and metadata. This is a reference implementation, not a kernel.
    """
    orig_dtype = x.dtype
    x = x.float()

    # Optional RHT (applied before quantization)
    rht_sign_used = None
    if rht:
        x, rht_sign_used = rht_forward(x, block_size=block_size, sign=rht_sign)

    # Global amax + scales
    global_amax = compute_global_amax(x, mode=global_amax_mode, percentile=percentile)
    global_encode_scale = (FP4_E2M1_MAX * FP8_E4M3_MAX) / (global_amax + EPS)
    global_encode_scale = torch.clamp(global_encode_scale, max=torch.finfo(torch.float32).max)
    if global_amax.item() == 0.0 or global_encode_scale.item() == 0.0:
        global_encode_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    global_decode_scale = 1.0 / global_encode_scale

    # Blockwise scaling
    if block_shape is not None:
        view, pads = _blockwise_view_2d(x, block_shape)
        block_amax = view.abs().amax(dim=(1, 3), keepdim=True)
        decode_scales = block_amax / FP4_E2M1_MAX
        decode_scales = decode_scales * global_encode_scale
        decode_scales = torch.clamp(decode_scales, min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
        decode_scales_fp8 = quantize_fp8_e4m3(decode_scales)
        encode_scales = 1.0 / (decode_scales_fp8.float() * global_decode_scale + EPS)

        # Quantize
        scaled = view * encode_scales
        scaled = clamp_fp4_range(scaled)
        if stochastic_rounding:
            q = quantize_fp4_sr(scaled, rng=rng)
        else:
            q = quantize_fp4_rtne(scaled)
        # Restore shape
        q = q.view(x.shape[0] + pads[0], x.shape[1] + pads[1]) if (pads[0] or pads[1]) else q.view_as(x)
        if pads[0] or pads[1]:
            q = q[: x.shape[0], : x.shape[1]]

        return NVFP4Quantized(
            q=q.to(orig_dtype),
            encode_scales=encode_scales,
            decode_scales_fp8=decode_scales_fp8,
            global_amax=global_amax,
            global_encode_scale=global_encode_scale,
            global_decode_scale=global_decode_scale,
            pad=pads,
            block_size=block_size,
            block_shape=block_shape,
            rht_sign=rht_sign_used,
        )

    # 1D blocks along last dim
    view, pad = _blockwise_view_1d(x, block_size)
    block_amax = view.abs().amax(dim=-1, keepdim=True)

    decode_scales = block_amax / FP4_E2M1_MAX
    decode_scales = decode_scales * global_encode_scale
    decode_scales = torch.clamp(decode_scales, min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    decode_scales_fp8 = quantize_fp8_e4m3(decode_scales)

    encode_scales = 1.0 / (decode_scales_fp8.float() * global_decode_scale + EPS)

    scaled = view * encode_scales
    scaled = clamp_fp4_range(scaled)
    if stochastic_rounding:
        q = quantize_fp4_sr(scaled, rng=rng)
    else:
        q = quantize_fp4_rtne(scaled)

    q = q.reshape(*x.shape[:-1], -1)
    if pad:
        q = q[..., : x.shape[-1]]

    return NVFP4Quantized(
        q=q.to(orig_dtype),
        encode_scales=encode_scales,
        decode_scales_fp8=decode_scales_fp8,
        global_amax=global_amax,
        global_encode_scale=global_encode_scale,
        global_decode_scale=global_decode_scale,
        pad=pad,
        block_size=block_size,
        block_shape=None,
        rht_sign=rht_sign_used,
    )


def nvfp4_dequantize(q: NVFP4Quantized, apply_inverse_rht: bool = False) -> torch.Tensor:
    """Dequantize NVFP4 back to FP32 (approx).

    If apply_inverse_rht=True, applies the inverse RHT using stored sign.
    """
    xq = q.q.float()

    if q.block_shape is not None:
        # 2D block scaling
        block_decode = q.decode_scales_fp8.float() * q.global_decode_scale
        x_view, pads = _blockwise_view_2d(xq, q.block_shape)
        x_hat = (x_view * block_decode).view(x_view.shape[0] * q.block_shape[0], x_view.shape[2] * q.block_shape[1])
        if isinstance(q.pad, tuple) and (q.pad[0] or q.pad[1]):
            x_hat = x_hat[: xq.shape[0], : xq.shape[1]]
    else:
        # 1D block scaling
        x_view, pad = _blockwise_view_1d(xq, q.block_size)
        block_decode = q.decode_scales_fp8.float() * q.global_decode_scale
        x_hat = (x_view * block_decode).reshape(*xq.shape[:-1], -1)
        if pad:
            x_hat = x_hat[..., : xq.shape[-1]]

    if apply_inverse_rht and q.rht_sign is not None:
        x_hat = rht_inverse(x_hat, sign=q.rht_sign, block_size=q.block_size)

    return x_hat
