"""NVFP4 reference utilities (CPU‑safe, educational)."""

from .constants import (
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    NVFP4_BLOCK_SIZE,
    FP4_VALUES,
)
from .fp4 import quantize_fp4_rtne, quantize_fp4_sr
from .fp8 import quantize_fp8_e4m3
from .hadamard import rht_forward, rht_inverse
from .nvfp4 import nvfp4_quantize, nvfp4_dequantize, NVFP4Quantized
from .metrics import tensor_stats, error_metrics
from .nn import NVFP4Linear, NVFP4Config

__all__ = [
    "NVFP4Linear",
    "NVFP4Config",
    "FP4_E2M1_MAX",
    "FP8_E4M3_MAX",
    "NVFP4_BLOCK_SIZE",
    "FP4_VALUES",
    "quantize_fp4_rtne",
    "quantize_fp4_sr",
    "quantize_fp8_e4m3",
    "rht_forward",
    "rht_inverse",
    "nvfp4_quantize",
    "nvfp4_dequantize",
    "NVFP4Quantized",
    "tensor_stats",
    "error_metrics",
]
