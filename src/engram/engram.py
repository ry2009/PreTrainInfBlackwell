from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class EngramConfig:
    vocab_size: int = 128
    d_model: int = 64
    ngram_orders: Tuple[int, ...] = (2, 3)
    num_heads: int = 4
    num_buckets: int = 2048
    conv_kernel: int = 4
    conv_dilation: int = 3
    use_conv: bool = True
    eps: float = 1e-6


def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _hash_ngram(ngram: torch.Tensor, seed: int, num_buckets: int) -> torch.Tensor:
    # ngram: (..., n)
    h = torch.zeros(ngram.shape[:-1], dtype=torch.int64, device=ngram.device)
    prime = 1_000_003
    for i in range(ngram.shape[-1]):
        h = (h * prime + ngram[..., i].to(torch.int64) + seed) % num_buckets
    return h


class EngramMemory(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        ngram_orders: Sequence[int] = (2, 3),
        num_heads: int = 4,
        num_buckets: int = 2048,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ngram_orders = tuple(ngram_orders)
        self.num_heads = num_heads
        self.num_buckets = num_buckets

        parts = len(self.ngram_orders) * self.num_heads
        if d_model % parts != 0:
            raise ValueError("d_model must be divisible by num_heads * num_orders")
        head_dim = d_model // parts
        self.head_dim = head_dim

        tables: List[nn.Embedding] = []
        seeds: List[int] = []
        seed_base = 1337
        for order in self.ngram_orders:
            for head in range(self.num_heads):
                tables.append(nn.Embedding(num_buckets, head_dim))
                seeds.append(seed_base + 97 * order + 17 * head)
        self.tables = nn.ModuleList(tables)
        self.register_buffer("_seeds", torch.tensor(seeds, dtype=torch.int64), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T)
        batch, seq_len = tokens.shape
        device = tokens.device

        mem_parts: List[torch.Tensor] = []
        table_idx = 0
        for order in self.ngram_orders:
            pad = torch.zeros((batch, order - 1), dtype=tokens.dtype, device=device)
            padded = torch.cat([pad, tokens], dim=1)
            ngrams = torch.stack(
                [padded[:, i : i + seq_len] for i in range(order)], dim=-1
            )  # (B, T, order)
            for head in range(self.num_heads):
                seed = int(self._seeds[table_idx].item())
                idx = _hash_ngram(ngrams, seed, self.num_buckets)
                emb = self.tables[table_idx](idx)
                mem_parts.append(emb)
                table_idx += 1
        mem = torch.cat(mem_parts, dim=-1)
        return mem


class EngramGate(nn.Module):
    def __init__(self, d_model: int, use_conv: bool = True, conv_kernel: int = 4, conv_dilation: int = 3) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.use_conv = use_conv
        self.conv_kernel = conv_kernel
        self.conv_dilation = conv_dilation
        if use_conv:
            padding = (conv_kernel - 1) * conv_dilation
            self.conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel,
                dilation=conv_dilation,
                padding=padding,
                groups=d_model,
            )

    def forward(self, h: torch.Tensor, mem: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        # h, mem: (B, T, D)
        k = self.key(mem)
        v = self.value(mem)
        h_norm = rmsnorm(h, eps=eps)
        k_norm = rmsnorm(k, eps=eps)
        scale = 1.0 / math.sqrt(h.shape[-1])
        alpha = torch.sigmoid((h_norm * k_norm).sum(dim=-1, keepdim=True) * scale)
        out = v * alpha
        if self.use_conv:
            out_t = out.transpose(1, 2)
            out_t = self.conv(out_t)
            out_t = out_t[..., : out.shape[1]]
            out = out_t.transpose(1, 2)
        return out, alpha


class EngramModule(nn.Module):
    def __init__(self, config: EngramConfig) -> None:
        super().__init__()
        self.config = config
        self.memory = EngramMemory(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            ngram_orders=config.ngram_orders,
            num_heads=config.num_heads,
            num_buckets=config.num_buckets,
        )
        self.gate = EngramGate(
            d_model=config.d_model,
            use_conv=config.use_conv,
            conv_kernel=config.conv_kernel,
            conv_dilation=config.conv_dilation,
        )

    def forward(self, h: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.memory(tokens)
        delta, alpha = self.gate(h, mem, eps=self.config.eps)
        return h + delta, alpha
