import os
import time

import modal

# Pick an NGC PyTorch image that already has CUDA libs.
IMAGE = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.01-py3").run_commands(
    "pip install --no-build-isolation 'transformer-engine[pytorch]==2.11.0' tqdm "
    "datasets transformers sentencepiece einops"
)

app = modal.App("nvfp4-pretraining-demo", image=IMAGE)

try:
    GPU_2X = modal.gpu.B200(count=2)
except Exception:
    GPU_2X = modal.gpu.A100(count=2)


def resolve_gpu():
    name = os.environ.get("MODAL_GPU", "B200").upper()
    # Return string to request exact GPU type (e.g., B200)
    return name


def _ddp_worker(rank: int, world_size: int, cfg: dict, out_dict):
    import os
    import time
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    vocab = cfg["vocab"]
    d_model = cfg["d_model"]
    seq_len = cfg["seq_len"]
    batch_size = cfg["batch_size"]
    steps = cfg["steps"]
    warmup = cfg["warmup"]

    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab, d_model),
        torch.nn.Linear(d_model, d_model),
        torch.nn.GELU(),
        torch.nn.Linear(d_model, vocab),
    ).to(device)
    ddp = DDP(model, device_ids=[rank])
    optim = torch.optim.AdamW(ddp.parameters(), lr=3e-4)

    def step_once():
        x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        optim.zero_grad(set_to_none=True)
        logits = ddp(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        optim.step()
        return loss

    for _ in range(warmup):
        step_once()
    torch.cuda.synchronize()

    start = time.time()
    loss = None
    for _ in range(steps):
        loss = step_once()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / steps
    tokens = batch_size * seq_len * world_size
    tokens_per_s = tokens / (elapsed / 1000.0)

    if rank == 0:
        out_dict["world_size"] = world_size
        out_dict["avg_step_ms"] = elapsed
        out_dict["tokens_per_s"] = tokens_per_s
        out_dict["loss"] = float(loss.item()) if loss is not None else None

    dist.destroy_process_group()


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


@app.function(gpu=GPU_2X, timeout=60 * 60)
def ddp_smoke(
    steps: int = 100,
    warmup: int = 10,
    vocab: int = 2048,
    seq_len: int = 128,
    batch_size: int = 8,
    d_model: int = 256,
):
    import torch
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()
    if world_size < 2:
        return {
            "world_size": world_size,
            "avg_step_ms": None,
            "tokens_per_s": None,
            "loss": None,
        }

    manager = mp.Manager()
    out_dict = manager.dict()
    cfg = {
        "steps": steps,
        "warmup": warmup,
        "vocab": vocab,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
    }
    mp.spawn(_ddp_worker, args=(world_size, cfg, out_dict), nprocs=world_size)
    return dict(out_dict)


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


@app.function(gpu=resolve_gpu(), timeout=60 * 120)
def run_hybrid_suite(
    dataset_name: str = "wikitext",
    dataset_subset: str = "wikitext-103-raw-v1",
    text_field: str = "text",
    tokenizer_name: str = "gpt2",
    tokenization: str = "bpe",  # "bpe" or "byte"
    train_samples: int = 2000,
    val_samples: int = 200,
    seq_len: int = 256,
    batch_size: int = 8,
    steps: int = 300,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    ffn_mult: int = 4,
    streams: int = 4,
    num_experts: int = 4,
    engram_buckets: int = 8192,
    engram_heads: int = 8,
    engram_orders: list[int] = [2, 3],
    engram_use_conv: bool = True,
    nvfp4_cooldown_frac: float = 0.0,
    lr: float = 3e-4,
    eval_every: int = 25,
    alloc_ratios: list[float] = [0.0, 0.3, 0.6, 1.0],
    size_sweep: list[int] = [256, 384, 512],
    memory_sweep: list[int] = [4096, 8192, 16384, 32768, 65536],
    long_context: list[int] = [256, 512, 1024],
    sweep_steps: int = 80,
    memory_sweep_steps: int = 60,
    ablation_steps: int = 120,
):
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ensure_te_compat()
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    device = "cuda"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    max_seq = max([seq_len] + list(long_context))

    def load_split(split: str):
        if dataset_subset:
            return load_dataset(dataset_name, dataset_subset, split=split)
        return load_dataset(dataset_name, split=split)

    def load_tokens(split: str, max_samples: int, fallback_train=None):
        try:
            ds = load_split(split)
        except Exception:
            ds = fallback_train if fallback_train is not None else load_split("train")
        texts = []
        # For validation fallback, take tail to avoid overlap with train head.
        if split != "train" and fallback_train is not None:
            start = max(0, len(ds) - max_samples)
            ds = ds.select(range(start, min(len(ds), start + max_samples)))
        else:
            ds = ds.select(range(min(max_samples, len(ds))))
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            text = row.get(text_field, "")
            if text:
                texts.append(text)
        if tokenization == "byte":
            blob = b"".join([t.encode("utf-8", errors="ignore") for t in texts])
            if len(blob) == 0:
                blob = b"empty"
            tokens = torch.tensor(list(blob), dtype=torch.long)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            tokenizer.model_max_length = max(max_seq * 2, 1024)
            tokens = tokenizer("\n\n".join(texts), return_tensors="pt")["input_ids"][0]
        # reshape into sequences
        if tokens.numel() < (max_seq + 1):
            reps = (max_seq + 1) // max(1, tokens.numel()) + 1
            tokens = tokens.repeat(reps)
        total = (tokens.numel() // (max_seq + 1)) * (max_seq + 1)
        tokens = tokens[:total].view(-1, max_seq + 1)
        return tokens

    train_ds = load_split("train")
    train_tokens = load_tokens("train", train_samples, fallback_train=train_ds)
    val_tokens = load_tokens("validation", val_samples, fallback_train=train_ds)
    vocab_size = int(max(train_tokens.max().item(), val_tokens.max().item()) + 1)

    # Build frequent bigram set on CPU for evaluation
    def build_bigram_set(tokens_cpu: torch.Tensor, top_k: int = 500):
        from collections import Counter

        flat = tokens_cpu.flatten().tolist()
        counts = Counter()
        for i in range(len(flat) - 1):
            counts[(flat[i], flat[i + 1])] += 1
        return set([bg for bg, _ in counts.most_common(top_k)])

    bigram_set = build_bigram_set(train_tokens[:, : seq_len + 1].cpu(), top_k=500)

    def sinkhorn(m: torch.Tensor, iters: int = 10, eps: float = 1e-6):
        m = torch.exp(m)
        for _ in range(iters):
            m = m / (m.sum(dim=-1, keepdim=True) + eps)
            m = m / (m.sum(dim=-2, keepdim=True) + eps)
        return m

    class Linear(nn.Module):
        def __init__(self, in_f: int, out_f: int, use_te: bool):
            super().__init__()
            self.use_te = use_te
            if use_te:
                self.lin = te.Linear(in_f, out_f, params_dtype=torch.bfloat16, bias=False, device=device)
            else:
                self.lin = nn.Linear(in_f, out_f, bias=False, device=device, dtype=torch.bfloat16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    class EngramMemory(nn.Module):
        def __init__(
            self,
            d_model_local: int,
            num_buckets: int = 4096,
            num_heads: int = 4,
            n_orders: tuple[int, ...] = (2, 3),
        ):
            super().__init__()
            self.num_buckets = num_buckets
            self.num_heads = num_heads
            self.n_orders = n_orders
            parts = len(n_orders) * num_heads
            self.head_dim = d_model_local // parts
            tables = []
            seeds = []
            seed_base = 1337
            for order in n_orders:
                for head in range(num_heads):
                    tables.append(nn.Embedding(num_buckets, self.head_dim, device=device))
                    seeds.append(seed_base + 97 * order + 17 * head)
            self.tables = nn.ModuleList(tables)
            self.register_buffer("_seeds", torch.tensor(seeds, dtype=torch.int64), persistent=False)

        def _hash(self, ngram: torch.Tensor, seed: int) -> torch.Tensor:
            h = torch.zeros(ngram.shape[:-1], dtype=torch.int64, device=ngram.device)
            prime = 1_000_003
            for i in range(ngram.shape[-1]):
                h = (h * prime + ngram[..., i].to(torch.int64) + seed) % self.num_buckets
            return h

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            # tokens: (B, T)
            b, t = tokens.shape
            device = tokens.device
            parts = []
            table_idx = 0
            for order in self.n_orders:
                pad = torch.zeros((b, order - 1), dtype=tokens.dtype, device=device)
                padded = torch.cat([pad, tokens], dim=1)
                ngrams = torch.stack([padded[:, i : i + t] for i in range(order)], dim=-1)
                for head in range(self.num_heads):
                    seed = int(self._seeds[table_idx].item())
                    idx = self._hash(ngrams, seed)
                    emb = self.tables[table_idx](idx)
                    parts.append(emb)
                    table_idx += 1
            return torch.cat(parts, dim=-1)

    class EngramGate(nn.Module):
        def __init__(self, d_model_local: int, use_conv: bool = True):
            super().__init__()
            self.key = nn.Linear(d_model_local, d_model_local, bias=False, device=device, dtype=torch.bfloat16)
            self.value = nn.Linear(d_model_local, d_model_local, bias=False, device=device, dtype=torch.bfloat16)
            self.use_conv = use_conv
            if use_conv:
                self.conv = nn.Conv1d(
                    d_model_local,
                    d_model_local,
                    kernel_size=4,
                    dilation=3,
                    padding=9,
                    groups=d_model_local,
                    device=device,
                    dtype=torch.bfloat16,
                )

        def forward(self, h: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            k = self.key(mem)
            v = self.value(mem)
            h_norm = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            k_norm = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            alpha = torch.sigmoid((h_norm * k_norm).sum(dim=-1, keepdim=True) / math.sqrt(h.shape[-1]))
            out = (v * alpha).to(torch.bfloat16)
            if self.use_conv:
                out = self.conv(out.transpose(1, 2))[..., : h.shape[1]].transpose(1, 2)
            return out, alpha

    class Engram(nn.Module):
        def __init__(
            self,
            d_model_local: int,
            num_buckets: int,
            num_heads: int,
            n_orders: tuple[int, ...],
            use_conv: bool,
        ):
            super().__init__()
            self.mem = EngramMemory(
                d_model_local,
                num_buckets=num_buckets,
                num_heads=num_heads,
                n_orders=n_orders,
            )
            self.gate = EngramGate(d_model_local, use_conv=use_conv)

        def forward(self, h: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mem = self.mem(tokens).to(torch.bfloat16)
            delta, alpha = self.gate(h, mem)
            return h + delta, alpha

    class SelfAttention(nn.Module):
        def __init__(self, d_model_local: int, n_heads_local: int, use_te: bool):
            super().__init__()
            self.d_model = d_model_local
            self.n_heads = n_heads_local
            self.qkv = Linear(d_model_local, 3 * d_model_local, use_te)
            self.proj = Linear(d_model_local, d_model_local, use_te)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            b, s, _ = q.shape
            head_dim = self.d_model // self.n_heads
            q = q.view(b, s, self.n_heads, head_dim).transpose(1, 2)
            k = k.view(b, s, self.n_heads, head_dim).transpose(1, 2)
            v = v.view(b, s, self.n_heads, head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(b, s, self.d_model)
            return self.proj(attn)

    class MLP(nn.Module):
        def __init__(self, d_model_local: int, use_te: bool):
            super().__init__()
            self.fc1 = Linear(d_model_local, ffn_mult * d_model_local, use_te)
            self.fc2 = Linear(ffn_mult * d_model_local, d_model_local, use_te)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(x)))

    class MoE(nn.Module):
        def __init__(self, d_model_local: int, num_experts_local: int):
            super().__init__()
            self.num_experts = num_experts_local
            self.router = nn.Linear(
                d_model_local, num_experts_local, bias=False, device=device, dtype=torch.bfloat16
            )
            self.experts = nn.ModuleList([MLP(d_model_local, use_te=False) for _ in range(num_experts_local)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.router(x)
            top = logits.argmax(dim=-1)
            out = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                mask = top == i
                if mask.any():
                    out[mask] = expert(x[mask])
            return out

    class StreamBlock(nn.Module):
        def __init__(
            self,
            d_model_local: int,
            n_heads_local: int,
            num_experts_local: int,
            use_engram: bool,
            use_moe: bool,
            use_mhc: bool,
            use_te: bool,
            engram_buckets_local: int,
            engram_heads_local: int,
            engram_orders_local: tuple[int, ...],
            engram_use_conv_local: bool,
        ):
            super().__init__()
            self.use_engram = use_engram
            self.use_mhc = use_mhc
            self.h_pre = nn.Parameter(torch.zeros(streams, device=device))
            self.h_post = nn.Parameter(torch.zeros(streams, device=device))
            self.h_res = nn.Parameter(torch.eye(streams, device=device))
            self.ln1 = nn.LayerNorm(d_model_local, device=device)
            self.attn = SelfAttention(d_model_local, n_heads_local, use_te=use_te)
            self.ln2 = nn.LayerNorm(d_model_local, device=device)
            self.mlp = MoE(d_model_local, num_experts_local) if use_moe else MLP(d_model_local, use_te=use_te)
            self.engram = (
                Engram(
                    d_model_local,
                    num_buckets=engram_buckets_local,
                    num_heads=engram_heads_local,
                    n_orders=engram_orders_local,
                    use_conv=engram_use_conv_local,
                )
                if use_engram
                else None
            )

        def _mix_pre(self):
            return torch.softmax(self.h_pre, dim=0)

        def _mix_post(self):
            return torch.softmax(self.h_post, dim=0)

        def _mix_res(self):
            return sinkhorn(self.h_res, iters=10) if self.use_mhc else self.h_res

        def res_matrix(self) -> torch.Tensor:
            return self._mix_res()

        def forward(self, x_streams: torch.Tensor, tokens: torch.Tensor):
            w_pre = self._mix_pre()
            w_post = self._mix_post()
            h_res = self._mix_res()
            x_in = (x_streams * w_pre.view(1, 1, -1, 1)).sum(dim=2)

            if self.engram is not None:
                x_in, gates = self.engram(x_in, tokens)
            else:
                gates = None

            h = self.ln1(x_in.float()).to(torch.bfloat16)
            h = x_in + self.attn(h)
            f = self.ln2(h.float()).to(torch.bfloat16)
            f = h + self.mlp(f)

            out_streams = f.unsqueeze(2) * w_post.view(1, 1, -1, 1)
            res = torch.einsum("ij,btsd->btid", h_res, x_streams)
            return res + out_streams, gates

    class HybridTransformer(nn.Module):
        def __init__(
            self,
            d_model_local: int,
            n_layers_local: int,
            n_heads_local: int,
            num_experts_local: int,
            use_engram: bool,
            use_moe: bool,
            use_mhc: bool,
            use_te: bool,
            engram_buckets_local: int,
            engram_heads_local: int,
            engram_orders_local: tuple[int, ...],
            engram_use_conv_local: bool,
        ):
            super().__init__()
            self.d_model = d_model_local
            self.embed = nn.Embedding(vocab_size, d_model_local, device=device, dtype=torch.bfloat16)
            self.pos = nn.Parameter(torch.randn(1, max_seq, d_model_local, device=device, dtype=torch.bfloat16) * 0.01)
            self.blocks = nn.ModuleList(
                [
                    StreamBlock(
                        d_model_local,
                        n_heads_local,
                        num_experts_local,
                        use_engram,
                        use_moe,
                        use_mhc,
                        use_te,
                        engram_buckets_local,
                        engram_heads_local,
                        engram_orders_local,
                        engram_use_conv_local,
                    )
                    for _ in range(n_layers_local)
                ]
            )
            self.ln_f = nn.LayerNorm(d_model_local, device=device)
            self.head = nn.Linear(d_model_local, vocab_size, bias=False, device=device, dtype=torch.bfloat16)

        def forward(self, tokens: torch.Tensor):
            x = (self.embed(tokens) + self.pos[:, : tokens.shape[1], :]).to(torch.bfloat16)
            streams_t = x.unsqueeze(2).repeat(1, 1, streams, 1)
            gates = None
            for block in self.blocks:
                streams_t, gates = block(streams_t, tokens)
            x = streams_t.mean(dim=2)
            x = self.ln_f(x.float()).to(torch.bfloat16)
            return self.head(x), gates

        def composite_amax(self) -> float:
            mat = torch.eye(streams, device=device)
            for blk in self.blocks:
                mat = blk.res_matrix() @ mat
            row = mat.abs().sum(dim=-1).max().item()
            col = mat.abs().sum(dim=-2).max().item()
            return max(row, col)

    recipe = NVFP4BlockScaling()

    def run_train(
        name: str,
        use_engram: bool,
        use_moe: bool,
        use_mhc: bool,
        use_nvfp4: bool,
        experts: int = num_experts,
        d_model_local: int = d_model,
        n_layers_local: int = n_layers,
        n_heads_local: int = n_heads,
        engram_buckets_local: int | None = None,
        engram_heads_local: int | None = None,
        engram_orders_local: tuple[int, ...] | None = None,
        engram_use_conv_local: bool | None = None,
        steps_override: int | None = None,
        long_context_eval: list[int] | None = None,
    ):
        buckets_local = engram_buckets_local if engram_buckets_local is not None else engram_buckets
        heads_local = engram_heads_local if engram_heads_local is not None else engram_heads
        orders_local = engram_orders_local if engram_orders_local is not None else tuple(engram_orders)
        use_conv_local = engram_use_conv_local if engram_use_conv_local is not None else engram_use_conv
        model = HybridTransformer(
            d_model_local,
            n_layers_local,
            n_heads_local,
            experts,
            use_engram,
            use_moe,
            use_mhc,
            use_te=use_nvfp4,
            engram_buckets_local=buckets_local,
            engram_heads_local=heads_local,
            engram_orders_local=orders_local,
            engram_use_conv_local=use_conv_local,
        )
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        history = {"step": [], "train_loss": [], "val_loss": [], "amax": [], "grad_norm": [], "gate_mean": []}

        run_steps = steps_override or steps
        cooldown_start = int(run_steps * (1.0 - nvfp4_cooldown_frac)) if use_nvfp4 else run_steps + 1

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        for step in range(run_steps):
            model.train()
            batch = train_tokens[torch.randint(0, train_tokens.shape[0], (batch_size,))].to(device)
            batch = batch[:, : seq_len + 1]
            x = batch[:, :-1]
            y = batch[:, 1:]

            optim.zero_grad(set_to_none=True)
            enable_nvfp4 = use_nvfp4 and step < cooldown_start
            with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            if step % eval_every == 0 or step == run_steps - 1:
                model.eval()
                with torch.no_grad():
                    vbatch = val_tokens[: batch_size].to(device)
                    vbatch = vbatch[:, : seq_len + 1]
                    vx = vbatch[:, :-1]
                    vy = vbatch[:, 1:]
                    enable_nvfp4 = use_nvfp4 and step < cooldown_start
                    with te.autocast(enabled=enable_nvfp4, recipe=recipe if enable_nvfp4 else None):
                        v_logits, gates = model(vx)
                        v_loss = F.cross_entropy(v_logits.float().view(-1, vocab_size), vy.reshape(-1))
                history["step"].append(step)
                history["train_loss"].append(float(loss.item()))
                history["val_loss"].append(float(v_loss.item()))
                history["grad_norm"].append(float(grad_norm))
                history["amax"].append(float(model.composite_amax()))
                history["gate_mean"].append(float(gates.mean().item()) if gates is not None else None)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_step_ms = (elapsed * 1000) / run_steps
        tokens_per_s = (batch_size * seq_len) / (avg_step_ms / 1000.0)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        # Inference latency
        model.eval()
        with torch.no_grad():
            xb = val_tokens[:batch_size, : seq_len + 1].to(device)
            xb = xb[:, :-1]
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                with te.autocast(enabled=use_nvfp4, recipe=recipe if use_nvfp4 else None):
                    _ = model(xb)
            torch.cuda.synchronize()
            infer_ms = ((time.time() - t0) * 1000) / 10

        # Bigram accuracy on a small eval slice
        bigram_acc = None
        with torch.no_grad():
            eval_batch = val_tokens[: batch_size * 2, : seq_len + 1].to(device)
            x_eval = eval_batch[:, :-1]
            y_eval = eval_batch[:, 1:]
            with te.autocast(enabled=use_nvfp4, recipe=recipe if use_nvfp4 else None):
                logits, _ = model(x_eval)
            preds = logits.argmax(dim=-1)
            mask = torch.zeros_like(y_eval, dtype=torch.bool)
            x_cpu = x_eval.cpu()
            for i in range(x_cpu.shape[0]):
                for t in range(1, x_cpu.shape[1]):
                    if (int(x_cpu[i, t - 1]), int(x_cpu[i, t])) in bigram_set:
                        mask[i, t] = True
            if mask.any():
                correct = (preds == y_eval) & mask.to(device)
                bigram_acc = float(correct.sum().item() / mask.sum().item())

        long_ctx_metrics = []
        if long_context_eval:
            for sl in long_context_eval:
                vbatch = val_tokens[: batch_size, : sl + 1].to(device)
                vx = vbatch[:, :-1]
                vy = vbatch[:, 1:]
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    with te.autocast(enabled=use_nvfp4, recipe=recipe if use_nvfp4 else None):
                        v_logits, _ = model(vx)
                        v_loss = F.cross_entropy(v_logits.float().view(-1, vocab_size), vy.reshape(-1))
                torch.cuda.synchronize()
                ms = (time.time() - t0) * 1000
                long_ctx_metrics.append(
                    {
                        "seq_len": sl,
                        "val_loss": float(v_loss.item()),
                        "latency_ms": ms,
                        "tokens_per_s": (batch_size * sl) / (ms / 1000.0),
                    }
                )

        gates_sample = None
        if use_engram:
            with torch.no_grad():
                sample = val_tokens[:1, : seq_len + 1].to(device)
                sx = sample[:, :-1]
                with te.autocast(enabled=use_nvfp4, recipe=recipe if use_nvfp4 else None):
                    _, gates = model(sx)
                if gates is not None:
                    gates_sample = gates[0, : min(128, gates.shape[1]), 0].float().cpu().tolist()

        return {
            "name": name,
            "avg_step_ms": avg_step_ms,
            "tokens_per_s": tokens_per_s,
            "peak_mem_gb": peak_mem,
            "infer_ms": infer_ms,
            "history": history,
            "params": sum(p.numel() for p in model.parameters()),
            "steps": run_steps,
            "bigram_acc": bigram_acc,
            "long_context": long_ctx_metrics,
            "gates_sample": gates_sample,
        }

    variants = {
        "dense": dict(use_engram=False, use_moe=False, use_mhc=False, use_nvfp4=False),
        "moe": dict(use_engram=False, use_moe=True, use_mhc=False, use_nvfp4=False),
        "engram": dict(use_engram=True, use_moe=False, use_mhc=False, use_nvfp4=False),
        "hc": dict(use_engram=True, use_moe=True, use_mhc=False, use_nvfp4=False),
        "hybrid_nvfp4": dict(use_engram=True, use_moe=True, use_mhc=True, use_nvfp4=True),
    }

    results = {}
    for name, cfg in variants.items():
        long_eval = long_context if name in {"dense", "hybrid_nvfp4"} else None
        results[name] = run_train(name, **cfg, long_context_eval=long_eval)

    # Ablation matrix around hybrid config
    ablation_grid = {
        "base": dict(use_engram=True, use_moe=True, use_mhc=True, use_nvfp4=True),
        "no_engram": dict(use_engram=False, use_moe=True, use_mhc=True, use_nvfp4=True),
        "no_mhc": dict(use_engram=True, use_moe=True, use_mhc=False, use_nvfp4=True),
        "no_moe": dict(use_engram=True, use_moe=False, use_mhc=True, use_nvfp4=True),
        "no_nvfp4": dict(use_engram=True, use_moe=True, use_mhc=True, use_nvfp4=False),
    }
    ablation_results = {}
    for name, cfg in ablation_grid.items():
        ablation_results[name] = run_train(
            name=f"ablation_{name}",
            **cfg,
            steps_override=ablation_steps,
            long_context_eval=None,
        )

    # Allocation sweep (memory vs experts) quick pass
    alloc_results = []
    for rho in alloc_ratios:
        experts = max(1, int(num_experts * max(rho, 0.05)))
        buckets = max(256, int(engram_buckets * max(0.1, 1.0 - rho + 0.1)))
        use_moe = rho > 0.0
        model = HybridTransformer(
            d_model,
            n_layers,
            n_heads,
            experts,
            use_engram=True,
            use_moe=use_moe,
            use_mhc=True,
            use_te=False,
            engram_buckets_local=buckets,
            engram_heads_local=engram_heads,
            engram_orders_local=tuple(engram_orders),
            engram_use_conv_local=engram_use_conv,
        )
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(50):
            batch = train_tokens[torch.randint(0, train_tokens.shape[0], (batch_size,))].to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            optim.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optim.step()
        with torch.no_grad():
            vbatch = val_tokens[: batch_size].to(device)
            vx = vbatch[:, :-1]
            vy = vbatch[:, 1:]
            v_logits, _ = model(vx)
            v_loss = F.cross_entropy(v_logits.float().view(-1, vocab_size), vy.reshape(-1))
        alloc_results.append({"rho": rho, "val_loss": float(v_loss.item()), "buckets": buckets, "experts": experts})

    def pick_heads(size: int, base: int) -> int:
        if size % base == 0:
            return base
        for h in range(base, 0, -1):
            if size % h == 0:
                return h
        return 1

    # Size sweep
    size_results = []
    for size in size_sweep:
        layers_local = max(2, int(n_layers * size / d_model))
        heads_local = pick_heads(size, n_heads)
        size_result = run_train(
            name=f"size_{size}",
            use_engram=True,
            use_moe=False,
            use_mhc=True,
            use_nvfp4=False,
            experts=num_experts,
            d_model_local=size,
            n_layers_local=layers_local,
            n_heads_local=heads_local,
            steps_override=sweep_steps,
        )
        size_results.append(
            {
                "d_model": size,
                "layers": layers_local,
                "heads": heads_local,
                "params": size_result["params"],
                "val_loss": size_result["history"]["val_loss"][-1] if size_result["history"]["val_loss"] else None,
                "tokens_per_s": size_result["tokens_per_s"],
            }
        )

    # Memory sweep (Engram buckets)
    memory_results = []
    for buckets in memory_sweep:
        mem_result = run_train(
            name=f"memory_{buckets}",
            use_engram=True,
            use_moe=False,
            use_mhc=True,
            use_nvfp4=False,
            experts=num_experts,
            engram_buckets_local=buckets,
            engram_heads_local=engram_heads,
            engram_orders_local=tuple(engram_orders),
            engram_use_conv_local=engram_use_conv,
            steps_override=memory_sweep_steps,
        )
        memory_results.append(
            {
                "buckets": buckets,
                "params": mem_result["params"],
                "val_loss": mem_result["history"]["val_loss"][-1] if mem_result["history"]["val_loss"] else None,
                "tokens_per_s": mem_result["tokens_per_s"],
            }
        )

    long_ctx = {
        name: res.get("long_context", []) for name, res in results.items() if res.get("long_context")
    }

    return {
        "config": {
            "dataset": f"{dataset_name}/{dataset_subset}",
            "tokenization": tokenization,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "steps": steps,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ffn_mult": ffn_mult,
            "streams": streams,
            "num_experts": num_experts,
            "engram_buckets": engram_buckets,
            "engram_heads": engram_heads,
            "engram_orders": engram_orders,
            "engram_use_conv": engram_use_conv,
            "memory_sweep": memory_sweep,
            "memory_sweep_steps": memory_sweep_steps,
        },
        "vocab_size": vocab_size,
        "results": results,
        "ablation_results": ablation_results,
        "alloc_results": alloc_results,
        "size_results": size_results,
        "memory_results": memory_results,
        "long_context": long_ctx,
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
    sweep_sizes: str = "1024,2048,4096,8192",
    sweep_steps: int = 20,
    sweep_warmup: int = 5,
    hybrid: bool = True,
    distributed: bool = False,
    dataset_name: str = "wikitext",
    dataset_subset: str = "wikitext-103-raw-v1",
    text_field: str = "text",
    tokenization: str = "bpe",
    engram_buckets: int = 8192,
    engram_heads: int = 8,
    engram_orders: str = "2,3",
    engram_use_conv: bool = True,
    alloc_ratios: str = "0.0,0.3,0.6,1.0",
    size_sweep: str = "256,384,512",
    size_sweep_steps: int = 80,
    ablation_steps: int = 120,
    memory_sweep: str = "4096,8192,16384,32768,65536",
    memory_sweep_steps: int = 60,
    nvfp4_cooldown_frac: float = 0.0,
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
        sizes = [int(x) for x in sweep_sizes.split(",") if x.strip()]
        sweep_data = sweep_gemm_sizes.remote(sizes=sizes, steps=sweep_steps, warmup=sweep_warmup)
        (out_dir / "part2_sweep.json").write_text(json.dumps(sweep_data, indent=2))
        print("Sweep:", sweep_data)

        try:
            import matplotlib.pyplot as plt

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams.update(
                {
                    "axes.titlesize": 12,
                    "axes.labelsize": 11,
                    "lines.linewidth": 2.2,
                    "lines.markersize": 5,
                    "legend.fontsize": 9,
                    "font.family": "DejaVu Sans",
                }
            )

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

    if distributed:
        ddp = ddp_smoke.remote()
        (out_dir / "ddp_smoke.json").write_text(json.dumps(ddp, indent=2))
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["ddp"], [ddp["tokens_per_s"]], color="#4C78A8")
            ax.set_ylabel("Tokens/s")
            ax.set_title(f"DDP Smoke (world={ddp['world_size']})")
            fig.tight_layout()
            fig.savefig(out_dir / "ddp_smoke.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            print("DDP plot skipped:", exc)
    if hybrid or "hybrid" in run_list:
        orders = [int(x) for x in engram_orders.split(",") if x.strip()]
        memory_sweep_list = [int(x) for x in memory_sweep.split(",") if x.strip()]
        alloc_ratio_list = [float(x) for x in alloc_ratios.split(",") if x.strip()]
        size_sweep_list = [int(x) for x in size_sweep.split(",") if x.strip()]
        hybrid_result = run_hybrid_suite.remote(
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            text_field=text_field,
            tokenization=tokenization,
            seq_len=model_seq,
            batch_size=model_batch,
            steps=steps,
            d_model=model_d_model,
            n_layers=model_layers,
            n_heads=model_heads,
            ffn_mult=ffn_mult,
            engram_buckets=engram_buckets,
            engram_heads=engram_heads,
            engram_orders=orders,
            engram_use_conv=engram_use_conv,
            alloc_ratios=alloc_ratio_list,
            size_sweep=size_sweep_list,
            sweep_steps=size_sweep_steps,
            memory_sweep=memory_sweep_list,
            memory_sweep_steps=memory_sweep_steps,
            ablation_steps=ablation_steps,
            nvfp4_cooldown_frac=nvfp4_cooldown_frac,
        )
        (out_dir / "hybrid_suite.json").write_text(json.dumps(hybrid_result, indent=2))
        print("Hybrid suite complete.")

        try:
            import matplotlib.pyplot as plt

            # Loss curves
            fig, ax = plt.subplots(figsize=(7, 4))
            palette = {
                "dense": "#222222",
                "moe": "#4C78A8",
                "engram": "#59A14F",
                "hc": "#E15759",
                "hybrid_nvfp4": "#B07AA1",
            }
            for name, res in hybrid_result["results"].items():
                hist = res["history"]
                ax.plot(
                    hist["step"],
                    hist["val_loss"],
                    label=name,
                    color=palette.get(name),
                    marker="o",
                )
            ax.set_xlabel("Step")
            ax.set_ylabel("Validation Loss")
            ax.set_title("Hybrid Suite: Val Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "hybrid_loss.png", dpi=160)
            plt.close(fig)

            # Throughput bar
            fig, ax = plt.subplots(figsize=(7, 4))
            names = list(hybrid_result["results"].keys())
            tps = [hybrid_result["results"][n]["tokens_per_s"] for n in names]
            colors = [palette.get(n, "#888888") for n in names]
            ax.bar(names, tps, color=colors)
            ax.set_ylabel("Tokens/s")
            ax.set_title("Training Throughput")
            fig.tight_layout()
            fig.savefig(out_dir / "hybrid_throughput.png", dpi=160)
            plt.close(fig)

            # Allocation sweep
            fig, ax = plt.subplots(figsize=(6, 4))
            rho = [r["rho"] for r in hybrid_result["alloc_results"]]
            loss = [r["val_loss"] for r in hybrid_result["alloc_results"]]
            ax.plot(rho, loss, marker="o", color="#4C78A8")
            ax.set_xlabel("Allocation ratio (rho)")
            ax.set_ylabel("Validation Loss")
            ax.set_title("Sparsity Allocation Sweep")
            fig.tight_layout()
            fig.savefig(out_dir / "hybrid_alloc.png", dpi=160)
            plt.close(fig)

            # Memory sweep
            mem = hybrid_result.get("memory_results", [])
            if mem:
                fig, ax = plt.subplots(figsize=(6, 4))
                buckets = [m["buckets"] for m in mem]
                m_loss = [m["val_loss"] for m in mem]
                ax.plot(buckets, m_loss, marker="o", color="#59A14F")
                ax.set_xscale("log", base=2)
                ax.set_xlabel("Engram buckets (log scale)")
                ax.set_ylabel("Validation Loss")
                ax.set_title("Engram Memory Scaling")
                fig.tight_layout()
                fig.savefig(out_dir / "hybrid_memory.png", dpi=160)
                plt.close(fig)

            # Long context latency
            fig, ax = plt.subplots(figsize=(6, 4))
            long_ctx = hybrid_result["long_context"]
            if isinstance(long_ctx, dict):
                for name, entries in long_ctx.items():
                    seqs = [r["seq_len"] for r in entries]
                    lat = [r["latency_ms"] for r in entries]
                    ax.plot(seqs, lat, marker="o", label=name, color=palette.get(name))
                ax.legend()
                ax.set_title("Long‑Context Latency (Dense vs Hybrid)")
            else:
                seqs = [r["seq_len"] for r in long_ctx]
                lat = [r["latency_ms"] for r in long_ctx]
                ax.plot(seqs, lat, marker="o")
                ax.set_title("Long‑Context Latency (Hybrid)")
            ax.set_xlabel("Seq length")
            ax.set_ylabel("Latency (ms)")
            fig.tight_layout()
            fig.savefig(out_dir / "hybrid_long_context.png", dpi=160)
            plt.close(fig)

            # Gating sample plot if present
            gate_series = []
            for name in ("hybrid_nvfp4", "engram", "hc"):
                sample = hybrid_result["results"].get(name, {}).get("gates_sample")
                if sample:
                    gate_series.append((name, sample))
            if gate_series:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                for name, series in gate_series:
                    ax.plot(series, label=name, color=palette.get(name))
                ax.set_xlabel("Token index")
                ax.set_ylabel("Gate α")
                ax.set_title("Engram Gate Activation")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "hybrid_gate.png", dpi=160)
                plt.close(fig)

            # Ablation matrix plot
            ablation = hybrid_result.get("ablation_results", {})
            if ablation:
                fig, ax = plt.subplots(figsize=(7, 4))
                labels = list(ablation.keys())
                vals = [
                    ablation[k]["history"]["val_loss"][-1]
                    if ablation[k]["history"]["val_loss"]
                    else None
                    for k in labels
                ]
                ax.bar(labels, vals, color="#4C78A8")
                ax.set_ylabel("Validation Loss")
                ax.set_title("Hybrid Ablation Matrix")
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                fig.savefig(out_dir / "hybrid_ablation.png", dpi=160)
                plt.close(fig)
        except Exception as exc:
            print("Hybrid plotting skipped:", exc)
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
