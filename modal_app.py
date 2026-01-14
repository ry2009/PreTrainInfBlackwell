import os
import time

import modal

# Pick an NGC PyTorch image that already has CUDA libs.
IMAGE = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.01-py3").run_commands(
    "pip install --no-build-isolation 'transformer-engine[pytorch]==2.11.0' tqdm"
)

app = modal.App("nvfp4-pretraining-demo", image=IMAGE)


def resolve_gpu():
    name = os.environ.get("MODAL_GPU", "B200").upper()
    # Return string to request exact GPU type (e.g., B200)
    return name


def ensure_te_compat():
    """Patch missing distributed symbols for TE in minimal single‑GPU runs."""
    import sys
    import types
    import enum
    import torch

    # FSDP stub for TrainingState
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_common  # noqa: F401
    except Exception:
        parent_name = "torch.distributed.fsdp._fully_shard"
        common_name = "torch.distributed.fsdp._fully_shard._fsdp_common"

        if parent_name not in sys.modules:
            parent_mod = types.ModuleType(parent_name)
            parent_mod.__path__ = []
            sys.modules[parent_name] = parent_mod

        common_mod = types.ModuleType(common_name)

        class TrainingState(enum.Enum):
            IDLE = 0
            FORWARD = 1
            BACKWARD = 2

        common_mod.TrainingState = TrainingState
        sys.modules[common_name] = common_mod
    # DTensor stub for torch builds without DTensor
    try:
        from torch.distributed.tensor import DTensor  # noqa: F401
    except Exception:
        try:
            import torch.distributed.tensor as tdt

            class DTensor:  # minimal stub
                pass

            tdt.DTensor = DTensor
        except Exception:
            # If torch.distributed.tensor itself is missing, create a stub module
            tensor_mod = types.ModuleType("torch.distributed.tensor")

            class DTensor:  # minimal stub
                pass

            tensor_mod.DTensor = DTensor
            sys.modules["torch.distributed.tensor"] = tensor_mod


@app.function(gpu=resolve_gpu(), timeout=60 * 30)
def nvfp4_benchmark(
    m: int = 2048,
    k: int = 2048,
    n: int = 2048,
    steps: int = 50,
    warmup: int = 10,
    use_fp4: bool = True,
):
    import torch
    ensure_te_compat()
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    linear = te.Linear(k, n, params_dtype=torch.bfloat16, bias=False, device=device)

    recipe = NVFP4BlockScaling()

    def run_loop(enable_nvfp4: bool):
        for _ in range(warmup):
            linear.zero_grad(set_to_none=True)
            with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
                out = linear(x)
                loss = out.float().mean()
            loss.backward()
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(steps):
            linear.zero_grad(set_to_none=True)
            with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
                out = linear(x)
                loss = out.float().mean()
            loss.backward()
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000 / steps
        return elapsed

    nvfp4_ms = run_loop(True) if use_fp4 else None
    bf16_ms = run_loop(False)

    return {
        "m": m,
        "k": k,
        "n": n,
        "steps": steps,
        "nvfp4_ms": nvfp4_ms,
        "bf16_ms": bf16_ms,
        "speedup": (bf16_ms / nvfp4_ms) if nvfp4_ms else None,
    }


@app.function(gpu=resolve_gpu(), timeout=60 * 60)
def sweep_gemm_sizes(
    sizes: list[int] = [1024, 2048, 4096, 8192],
    steps: int = 20,
    warmup: int = 5,
):
    import torch
    ensure_te_compat()
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    device = "cuda"
    torch.manual_seed(0)
    recipe = NVFP4BlockScaling()

    results = []

    for n in sizes:
        x = torch.randn(n, n, device=device, dtype=torch.bfloat16, requires_grad=True)
        linear = te.Linear(n, n, params_dtype=torch.bfloat16, bias=False, device=device)

        def run_loop(enable_nvfp4: bool):
            for _ in range(warmup):
                linear.zero_grad(set_to_none=True)
                with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
                    out = linear(x)
                    loss = out.float().mean()
                loss.backward()
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(steps):
                linear.zero_grad(set_to_none=True)
                with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
                    out = linear(x)
                    loss = out.float().mean()
                loss.backward()
            torch.cuda.synchronize()
            return (time.time() - start) * 1000 / steps

        nvfp4_ms = run_loop(True)
        bf16_ms = run_loop(False)
        results.append(
            {
                "n": n,
                "nvfp4_ms": nvfp4_ms,
                "bf16_ms": bf16_ms,
                "speedup": bf16_ms / nvfp4_ms,
            }
        )

    return results


@app.function(gpu=resolve_gpu(), timeout=60 * 60)
def train_tiny_transformer(
    precision: str = "nvfp4",
    steps: int = 120,
    vocab: int = 2048,
    seq_len: int = 128,
    batch_size: int = 8,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    ffn_mult: int = 4,
    lr: float = 3e-4,
    cooldown_frac: float = 0.15,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ensure_te_compat()
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    device = "cuda"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model, device=device, dtype=torch.bfloat16)
            self.qkv = te.Linear(d_model, 3 * d_model, params_dtype=torch.bfloat16, bias=False, device=device)
            self.proj = te.Linear(d_model, d_model, params_dtype=torch.bfloat16, bias=False, device=device)
            self.ln2 = nn.LayerNorm(d_model, device=device, dtype=torch.bfloat16)
            self.ff1 = te.Linear(d_model, ffn_mult * d_model, params_dtype=torch.bfloat16, bias=False, device=device)
            self.ff2 = te.Linear(ffn_mult * d_model, d_model, params_dtype=torch.bfloat16, bias=False, device=device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.to(torch.bfloat16)
            h = self.ln1(x)
            qkv = self.qkv(h)
            q, k, v = qkv.chunk(3, dim=-1)
            b, s, _ = q.shape
            head_dim = d_model // n_heads
            q = q.view(b, s, n_heads, head_dim).transpose(1, 2)
            k = k.view(b, s, n_heads, head_dim).transpose(1, 2)
            v = v.view(b, s, n_heads, head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(b, s, d_model).to(torch.bfloat16)
            x = x + self.proj(attn)
            f = self.ln2(x.to(torch.bfloat16))
            f = self.ff2(F.gelu(self.ff1(f))).to(torch.bfloat16)
            return (x + f).to(torch.bfloat16)

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model, dtype=torch.bfloat16, device=device)
            self.pos = nn.Parameter(torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16) * 0.01)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model, device=device, dtype=torch.bfloat16)
            self.head = te.Linear(d_model, vocab, params_dtype=torch.bfloat16, bias=False, device=device)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            x = (self.embed(idx) + self.pos[:, : idx.shape[1], :]).to(torch.bfloat16)
            for blk in self.blocks:
                x = blk(x)
            x = self.ln_f(x.to(torch.bfloat16))
            return self.head(x)

    model = TinyTransformer().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    recipe = NVFP4BlockScaling()
    losses = []

    cooldown_start = int(steps * (1.0 - cooldown_frac)) if precision == "cooldown" else steps + 1

    torch.cuda.synchronize()
    start_time = time.time()
    for step in range(steps):
        tokens = torch.randint(0, vocab, (batch_size, seq_len + 1), device=device)
        x = tokens[:, :-1]
        y = tokens[:, 1:]

        optim.zero_grad(set_to_none=True)

        enable_nvfp4 = precision == "nvfp4" or (precision == "cooldown" and step < cooldown_start)
        with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab), y.reshape(-1))
        loss.backward()
        optim.step()

        losses.append(loss.item())

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_step_ms = (total_time * 1000) / steps
    tokens_per_s = (batch_size * seq_len) / (avg_step_ms / 1000.0)

    return {
        "precision": precision,
        "steps": steps,
        "vocab": vocab,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ffn_mult": ffn_mult,
        "lr": lr,
        "cooldown_frac": cooldown_frac,
        "avg_step_ms": avg_step_ms,
        "tokens_per_s": tokens_per_s,
        "losses": losses,
    }


@app.function(gpu=resolve_gpu(), timeout=60 * 60)
def train_mlp_only(
    precision: str = "nvfp4",
    steps: int = 200,
    dim: int = 2048,
    seq_len: int = 256,
    batch_size: int = 8,
    depth: int = 4,
    lr: float = 3e-4,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ensure_te_compat()
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    device = "cuda"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    class MLPBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = te.Linear(dim, 4 * dim, params_dtype=torch.bfloat16, bias=False, device=device)
            self.fc2 = te.Linear(4 * dim, dim, params_dtype=torch.bfloat16, bias=False, device=device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            return x

    class MLPStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([MLPBlock() for _ in range(depth)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                x = blk(x)
            return x

    model = MLPStack().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    recipe = NVFP4BlockScaling()
    losses = []

    tokens = batch_size * seq_len

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(steps):
        x = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
        optim.zero_grad(set_to_none=True)

        enable_nvfp4 = precision == "nvfp4"
        with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
            out = model(x)
            loss = out.float().mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_step_ms = (total_time * 1000) / steps
    tokens_per_s = tokens / (avg_step_ms / 1000.0)

    return {
        "precision": precision,
        "steps": steps,
        "dim": dim,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "depth": depth,
        "lr": lr,
        "avg_step_ms": avg_step_ms,
        "tokens_per_s": tokens_per_s,
        "losses": losses,
    }


@app.local_entrypoint()
def main(
    runs: str = "nvfp4,cooldown,bf16",
    steps: int = 120,
    benchmark: bool = True,
    bench_m: int = 8192,
    bench_k: int = 8192,
    bench_n: int = 8192,
    bench_steps: int = 30,
    bench_warmup: int = 10,
    model_d_model: int = 512,
    model_layers: int = 6,
    model_heads: int = 8,
    model_seq: int = 256,
    model_batch: int = 8,
    ffn_mult: int = 4,
    lr: float = 3e-4,
    cooldown_frac: float = 0.15,
    mlp_dim: int = 2048,
    mlp_seq: int = 256,
    mlp_batch: int = 8,
    mlp_depth: int = 4,
    sweep: bool = True,
):
    import json
    from pathlib import Path

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    if benchmark:
        bench = nvfp4_benchmark.remote(
            m=bench_m,
            k=bench_k,
            n=bench_n,
            steps=bench_steps,
            warmup=bench_warmup,
        )
        (out_dir / "part2_benchmark.json").write_text(json.dumps(bench, indent=2))
        print("Benchmark:", bench)

    if sweep:
        sweep_data = sweep_gemm_sizes.remote()
        (out_dir / "part2_sweep.json").write_text(json.dumps(sweep_data, indent=2))
        print("Sweep:", sweep_data)

        try:
            import matplotlib.pyplot as plt

            sizes = [d["n"] for d in sweep_data]
            speedups = [d["speedup"] for d in sweep_data]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(sizes, speedups, marker="o", color="#2a6d62")
            ax.axhline(1.0, color="#999", linestyle="--", linewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("GEMM size (N x N)")
            ax.set_ylabel("BF16 / NVFP4 speedup")
            ax.set_title("NVFP4 Crossover vs GEMM Size")
            fig.tight_layout()
            fig.savefig(out_dir / "part2_sweep.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            print("Sweep plot skipped:", exc)

    run_list = [r.strip() for r in runs.split(",") if r.strip()]
    for precision in run_list:
        if precision in {"nvfp4", "bf16", "cooldown"}:
            result = train_tiny_transformer.remote(
                precision=precision,
                steps=steps,
                seq_len=model_seq,
                batch_size=model_batch,
                d_model=model_d_model,
                n_layers=model_layers,
                n_heads=model_heads,
                ffn_mult=ffn_mult,
                lr=lr,
                cooldown_frac=cooldown_frac,
            )
            out_path = out_dir / f"part2_train_{precision}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(
                f"Train {precision}:",
                {k: result[k] for k in ("precision", "avg_step_ms", "tokens_per_s", "losses") if k in result},
            )

            # Local plot (if matplotlib installed)
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(result["losses"])
                ax.set_title(f"Part 2: Tiny Transformer ({precision})")
                ax.set_xlabel("step")
                ax.set_ylabel("loss")
                fig.tight_layout()
                fig.savefig(out_dir / f"part2_train_{precision}.png", dpi=160)
                plt.close(fig)
            except Exception as exc:
                print("Plot skipped:", exc)
        elif precision in {"mlp_nvfp4", "mlp_bf16"}:
            mlp_precision = precision.split("_", 1)[1]
            result = train_mlp_only.remote(
                precision=mlp_precision,
                steps=steps,
                dim=mlp_dim,
                seq_len=mlp_seq,
                batch_size=mlp_batch,
                depth=mlp_depth,
                lr=lr,
            )
            out_path = out_dir / f"part2_mlp_{mlp_precision}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(
                f"MLP {mlp_precision}:",
                {k: result[k] for k in ("precision", "avg_step_ms", "tokens_per_s", "losses") if k in result},
            )
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(result["losses"])
                ax.set_title(f"Part 2: MLP‑Only ({mlp_precision})")
                ax.set_xlabel("step")
                ax.set_ylabel("loss")
                fig.tight_layout()
                fig.savefig(out_dir / f"part2_mlp_{mlp_precision}.png", dpi=160)
                plt.close(fig)
            except Exception as exc:
                print("Plot skipped:", exc)
        else:
            print(f"Skipping unknown precision: {precision}")
