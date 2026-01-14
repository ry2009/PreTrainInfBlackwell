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


def generate_sequences(
    num_sequences: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 0,
) -> torch.Tensor:
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


class BaselineLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor):
        h = self.embed(tokens)
        logits = self.head(h)
        return logits, None


class EngramLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, config: EngramConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.engram = EngramModule(config)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor):
        h = self.embed(tokens)
        h, gates = self.engram(h, tokens)
        logits = self.head(h)
        return logits, gates


def sample_batch(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    idx = torch.randint(0, data.shape[0], (batch_size,))
    return data[idx]


def run_train(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    eval_every: int = 20,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"step": [], "train_loss": [], "val_loss": []}
    for step in range(steps):
        model.train()
        batch = sample_batch(train_data, batch_size).to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                val = val_data.to(device)
                v_inputs = val[:, :-1]
                v_targets = val[:, 1:]
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
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train = generate_sequences(2048, args.seq, args.vocab, seed=1)
    val = generate_sequences(256, args.seq, args.vocab, seed=2)

    baseline = BaselineLM(args.vocab, args.d_model)
    eng_cfg = EngramConfig(
        vocab_size=args.vocab,
        d_model=args.d_model,
        ngram_orders=(2, 3),
        num_heads=4,
        num_buckets=1024,
        conv_kernel=4,
        conv_dilation=3,
        use_conv=True,
    )
    engram = EngramLM(args.vocab, args.d_model, eng_cfg)

    hist_base = run_train(baseline, train, val, args.steps, args.batch, 2e-3, device)
    hist_eng = run_train(engram, train, val, args.steps, args.batch, 2e-3, device)

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": hist_base,
        "engram": hist_eng,
        "config": {
            "steps": args.steps,
            "batch": args.batch,
            "vocab": args.vocab,
            "seq": args.seq,
            "d_model": args.d_model,
        },
    }
    (reports / "engram_toy.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_base["step"], hist_base["val_loss"], label="Baseline (no memory)")
    ax.plot(hist_eng["step"], hist_eng["val_loss"], label="Engram (conditional memory)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Toy Engram: N-gram Memory vs Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "engram_toy.png", dpi=160)

    # Gate heatmap
    eng_model = engram.to(device)
    eng_model.eval()
    sample = val[:1].to(device)
    with torch.no_grad():
        _, gates = eng_model(sample[:, :-1])
    if gates is not None:
        gate_vals = gates.squeeze(0).squeeze(-1).cpu().numpy()
        fig2, ax2 = plt.subplots(figsize=(7, 1.8))
        ax2.imshow(gate_vals[None, :], aspect="auto", cmap="magma")
        ax2.set_yticks([])
        ax2.set_xlabel("Token position")
        ax2.set_title("Engram Gate Activations (toy)")
        fig2.tight_layout()
        fig2.savefig(reports / "engram_gate.png", dpi=160)

    print("Wrote", reports / "engram_toy.png")


if __name__ == "__main__":
    main()
