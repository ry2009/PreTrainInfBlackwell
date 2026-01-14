from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _path import ROOT
from engram import EngramConfig, EngramModule
from mhc import sinkhorn
from nvfp4.nn import NVFP4Config, NVFP4Linear


def generate_sequences(num_sequences: int, seq_len: int, vocab_size: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    seq = np.zeros((num_sequences, seq_len), dtype=np.int64)
    seq[:, :2] = rng.integers(0, vocab_size, size=(num_sequences, 2))
    p1, p2, p3, p4 = 131, 193, 337, 521
    for t in range(2, seq_len):
        a = seq[:, t - 2]
        b = seq[:, t - 1]
        c = seq[:, t - 1] ^ seq[:, t - 2]
        seq[:, t] = (a * p1 + b * p2 + c * p3 + p4) % vocab_size
    return torch.tensor(seq, dtype=torch.long)


class StreamBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        streams: int,
        use_mhc: bool = False,
        use_nvfp4: bool = False,
        sinkhorn_iters: int = 10,
    ) -> None:
        super().__init__()
        self.streams = streams
        self.use_mhc = use_mhc
        self.sinkhorn_iters = sinkhorn_iters

        self.h_pre = nn.Parameter(torch.zeros(streams))
        self.h_post = nn.Parameter(torch.zeros(streams))
        self.h_res = nn.Parameter(torch.eye(streams))

        if use_nvfp4:
            cfg = NVFP4Config(stochastic_rounding=False, rht=False)
            self.ff = NVFP4Linear(d_model, d_model, cfg=cfg)
        else:
            self.ff = nn.Linear(d_model, d_model)
        self.act = nn.Tanh()

    def _mix_pre(self) -> torch.Tensor:
        return torch.softmax(self.h_pre, dim=0)

    def _mix_post(self) -> torch.Tensor:
        return torch.softmax(self.h_post, dim=0)

    def _mix_res(self) -> torch.Tensor:
        if self.use_mhc:
            return sinkhorn(self.h_res, iters=self.sinkhorn_iters)
        return self.h_res

    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        # x_streams: (B, T, S, D)
        w_pre = self._mix_pre()
        w_post = self._mix_post()
        h_res = self._mix_res()

        x_in = (x_streams * w_pre.view(1, 1, -1, 1)).sum(dim=2)
        out = self.act(self.ff(x_in))
        out_streams = out.unsqueeze(2) * w_post.view(1, 1, -1, 1)
        res = torch.einsum("ij,btjd->btid", h_res, x_streams)
        return res + out_streams

    def res_matrix(self) -> torch.Tensor:
        return self._mix_res()


class HybridLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        layers: int,
        streams: int,
        use_engram: bool = False,
        use_mhc: bool = False,
        use_nvfp4: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.layers = layers
        self.streams = streams
        self.use_engram = use_engram

        self.embed = nn.Embedding(vocab_size, d_model)
        if use_engram:
            self.engram = EngramModule(
                EngramConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    ngram_orders=(2, 3),
                    num_heads=4,
                    num_buckets=2048,
                    conv_kernel=4,
                    conv_dilation=3,
                    use_conv=True,
                )
            )
        else:
            self.engram = None

        self.blocks = nn.ModuleList(
            [
                StreamBlock(
                    d_model,
                    streams,
                    use_mhc=use_mhc,
                    use_nvfp4=use_nvfp4,
                    sinkhorn_iters=10,
                )
                for _ in range(layers)
            ]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor):
        h = self.embed(tokens)
        gates = None
        if self.engram is not None:
            h, gates = self.engram(h, tokens)
        streams = h.unsqueeze(2).repeat(1, 1, self.streams, 1)
        for block in self.blocks:
            streams = block(streams)
        h_out = streams.mean(dim=2)
        logits = self.head(h_out)
        return logits, gates

    def composite_amax(self) -> float:
        mat = torch.eye(self.streams, device=next(self.parameters()).device)
        for block in self.blocks:
            mat = block.res_matrix() @ mat
        row_sum = mat.abs().sum(dim=-1).max().item()
        col_sum = mat.abs().sum(dim=-2).max().item()
        return max(row_sum, col_sum)


def sample_batch(data: torch.Tensor, batch: int) -> torch.Tensor:
    idx = torch.randint(0, data.shape[0], (batch,))
    return data[idx]


def train_model(model: HybridLM, train: torch.Tensor, val: torch.Tensor, steps: int, batch: int, lr: float, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"step": [], "train_loss": [], "val_loss": []}
    for step in range(steps):
        model.train()
        batch_data = sample_batch(train, batch).to(device)
        inputs = batch_data[:, :-1]
        targets = batch_data[:, 1:]
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                v_inputs = val[:, :-1].to(device)
                v_targets = val[:, 1:].to(device)
                v_logits, _ = model(v_inputs)
                v_loss = F.cross_entropy(v_logits.reshape(-1, v_logits.shape[-1]), v_targets.reshape(-1))
            history["step"].append(step)
            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(v_loss.item()))
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train = generate_sequences(2048, args.seq, args.vocab, seed=3)
    val = generate_sequences(256, args.seq, args.vocab, seed=4)

    configs = {
        "baseline": dict(use_engram=False, use_mhc=False, use_nvfp4=False),
        "engram": dict(use_engram=True, use_mhc=False, use_nvfp4=False),
        "engram_mhc": dict(use_engram=True, use_mhc=True, use_nvfp4=False),
        "engram_mhc_nvfp4": dict(use_engram=True, use_mhc=True, use_nvfp4=True),
    }

    histories = {}
    amax = {}
    for name, cfg in configs.items():
        model = HybridLM(
            vocab_size=args.vocab,
            d_model=args.d_model,
            layers=args.layers,
            streams=args.streams,
            **cfg,
        )
        histories[name] = train_model(model, train, val, args.steps, args.batch, 2e-3, device)
        with torch.no_grad():
            amax[name] = float(model.composite_amax())

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {"histories": histories, "amax": amax, "config": vars(args)}
    (reports / "hybrid_efficiency.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, hist in histories.items():
        ax.plot(hist["step"], hist["val_loss"], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Hybrid Efficient LM: Engram + mHC + NVFP4 (toy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "hybrid_efficiency.png", dpi=160)

    # Gate heatmap (from the strongest model)
    model = HybridLM(
        vocab_size=args.vocab,
        d_model=args.d_model,
        layers=args.layers,
        streams=args.streams,
        use_engram=True,
        use_mhc=True,
        use_nvfp4=True,
    ).to(device)
    model.eval()
    sample = val[:1].to(device)
    with torch.no_grad():
        _, gates = model(sample[:, :-1])
    if gates is not None:
        gate_vals = gates.squeeze(0).squeeze(-1).cpu().numpy()
        fig2, ax2 = plt.subplots(figsize=(7, 1.8))
        ax2.imshow(gate_vals[None, :], aspect="auto", cmap="magma")
        ax2.set_yticks([])
        ax2.set_xlabel("Token position")
        ax2.set_title("Hybrid Gate Activations (toy)")
        fig2.tight_layout()
        fig2.savefig(reports / "hybrid_gate.png", dpi=160)

    print("Wrote", reports / "hybrid_efficiency.png")


if __name__ == "__main__":
    main()
