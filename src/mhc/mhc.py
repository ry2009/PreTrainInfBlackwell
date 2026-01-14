from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


def sinkhorn(matrix: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    # Project to doubly stochastic via Sinkhorn-Knopp.
    m = torch.exp(matrix)
    for _ in range(iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


@dataclass
class MHCConfig:
    d_model: int = 64
    streams: int = 4
    layers: int = 12
    sinkhorn_iters: int = 10


class HyperConnectionLayer(nn.Module):
    def __init__(self, d_model: int, streams: int, mhc: bool = False, sinkhorn_iters: int = 10) -> None:
        super().__init__()
        self.d_model = d_model
        self.streams = streams
        self.mhc = mhc
        self.sinkhorn_iters = sinkhorn_iters

        self.h_pre = nn.Parameter(torch.zeros(1, streams))
        self.h_post = nn.Parameter(torch.zeros(1, streams))
        self.h_res = nn.Parameter(torch.zeros(streams, streams))

        self.layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

        with torch.no_grad():
            self.h_res.copy_(torch.eye(streams))

    def _mix_pre(self) -> torch.Tensor:
        if self.mhc:
            return torch.sigmoid(self.h_pre)
        return self.h_pre

    def _mix_post(self) -> torch.Tensor:
        if self.mhc:
            return torch.sigmoid(self.h_post)
        return self.h_post

    def _mix_res(self) -> torch.Tensor:
        if self.mhc:
            return sinkhorn(self.h_res, iters=self.sinkhorn_iters)
        return self.h_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        w_pre = self._mix_pre()
        w_post = self._mix_post()
        h_res = self._mix_res()

        x_in = torch.einsum("bs,bsd->bd", w_pre, x)
        out = self.layer(x_in)
        out_streams = torch.einsum("bs,bd->bsd", w_post, out)
        res = torch.einsum("ij,bsd->bid", h_res, x)
        return res + out_streams

    def res_matrix(self) -> torch.Tensor:
        return self._mix_res()


class HyperConnectionNet(nn.Module):
    def __init__(self, config: MHCConfig, mhc: bool = False) -> None:
        super().__init__()
        self.config = config
        self.mhc = mhc
        self.layers = nn.ModuleList(
            [
                HyperConnectionLayer(
                    config.d_model,
                    config.streams,
                    mhc=mhc,
                    sinkhorn_iters=config.sinkhorn_iters,
                )
                for _ in range(config.layers)
            ]
        )
        self.head = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        streams = x.unsqueeze(1).repeat(1, self.config.streams, 1)
        for layer in self.layers:
            streams = layer(streams)
        pooled = streams.mean(dim=1)
        return self.head(pooled)

    def composite_residual(self) -> torch.Tensor:
        mat = torch.eye(self.config.streams, device=next(self.parameters()).device)
        for layer in self.layers:
            mat = layer.res_matrix() @ mat
        return mat
