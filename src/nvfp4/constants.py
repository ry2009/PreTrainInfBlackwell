"""Format constants for NVFP4 reference quantization."""

import torch

# FP4 E2M1
FP4_E2M1_MAX = 6.0

# FP8 E4M3 (finite numbers, max 448 per NVFP4 recipe)
FP8_E4M3_MAX = 448.0

# NVFP4 block size (1x16 by default)
NVFP4_BLOCK_SIZE = 16

# FP4 representable values (E2M1) including sign
FP4_VALUES = torch.tensor(
    [
        0.0,   # 0000
        0.5,   # 0001
        1.0,   # 0010
        1.5,   # 0011
        2.0,   # 0100
        3.0,   # 0101
        4.0,   # 0110
        6.0,   # 0111
        -0.0,  # 1000
        -0.5,  # 1001
        -1.0,  # 1010
        -1.5,  # 1011
        -2.0,  # 1100
        -3.0,  # 1101
        -4.0,  # 1110
        -6.0,  # 1111
    ],
    dtype=torch.float32,
)

# E4M3 params for reference quantization (approximate)
E4M3_MANTISSA_BITS = 3
E4M3_MIN_EXP = -6
E4M3_MAX_EXP = 8

# Small epsilon for numeric stability
EPS = 1e-12
