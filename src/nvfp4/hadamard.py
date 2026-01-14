"""Randomized Hadamard Transform (RHT) utilities."""

from __future__ import annotations

import torch


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    """Generate an n x n normalized Hadamard matrix (n must be power of 2)."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")
    if n == 1:
        return torch.tensor([[1.0]], dtype=dtype, device=device)

    h = hadamard_matrix(n // 2, dtype=dtype, device=device)
    top = torch.cat([h, h], dim=1)
    bottom = torch.cat([h, -h], dim=1)
    out = torch.cat([top, bottom], dim=0) / (2 ** 0.5)
    return out


def rht_forward(x: torch.Tensor, block_size: int = 16, sign: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply randomized Hadamard transform along the last dimension in blocks.

    Returns (transformed, sign_vector).
    """
    if x.shape[-1] % block_size != 0:
        raise ValueError("Last dimension must be divisible by block_size for RHT.")

    device = x.device
    dtype = x.dtype

    if sign is None:
        sign = torch.randint(0, 2, (block_size,), device=device, dtype=dtype) * 2 - 1

    h = hadamard_matrix(block_size, dtype=dtype, device=device)

    # Reshape into blocks on last dim
    x_view = x.reshape(*x.shape[:-1], -1, block_size)
    x_signed = x_view * sign

    # Apply hadamard
    y = torch.einsum("...bd,df->...bf", x_signed, h)
    y = y.reshape_as(x)
    return y, sign


def rht_inverse(y: torch.Tensor, sign: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Invert RHT: apply Hadamard then sign (since H is orthogonal and D^2=I)."""
    if y.shape[-1] % block_size != 0:
        raise ValueError("Last dimension must be divisible by block_size for RHT.")

    device = y.device
    dtype = y.dtype

    h = hadamard_matrix(block_size, dtype=dtype, device=device)

    y_view = y.reshape(*y.shape[:-1], -1, block_size)
    x = torch.einsum("...bd,df->...bf", y_view, h)
    x = x * sign
    return x.reshape_as(y)
