# NVFP4 Pretraining Playground (Part 1 + Part 2)

A practical, show‑your‑work demo of NVFP4 mixed‑precision training. This repo has:

- **Part 1 (Theory → Practice):** A reference NVFP4 quantization pipeline (global + block scaling), Random Hadamard Transform (RHT), and Stochastic Rounding (SR), with correctness + error metrics and plots.
- **Part 2 (Systems):** A **Modal** app that runs **TransformerEngine** NVFP4 on a Blackwell GPU, with benchmarks and profiling hooks.
- **Twist:** Adaptive global scaling + precision cooldown schedule (NVFP4 → BF16) to demonstrate a custom recipe.
- **DeepSeek Add‑ons (Toy):** Engram‑style conditional memory + mHC (manifold‑constrained hyper‑connections) with small‑scale repro plots.
- **Hybrid Suite (GPU):** Engram + mHC + MoE + NVFP4 on a real dataset (byte‑level WikiText), with loss/throughput/latency/stability plots.

This is not a production kernel implementation — it’s a **reproducible, explainable demo** that lets you show you understand both **numerics** and **systems**.

---

## Quickstart (Local, Part 1)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Sanity check: NVFP4 round‑trip + metrics
python scripts/part1_roundtrip.py

# RHT outlier demo
python scripts/part1_rht_outliers.py

# Quantization ablations + plots
python scripts/part1_ablation_report.py
```

Outputs are saved to `reports/` as PNGs + JSON metrics.

---

## Part 2 (Modal + Blackwell)

1. Install Modal locally:
   ```bash
   pip install modal
   python3 -m modal setup
   ```

2. Run the remote NVFP4 benchmark + tiny transformer training on GPU:
   ```bash
   modal run modal_app.py
   ```

If your Modal account uses a specific Blackwell GPU name (e.g. `B200`), set:

```bash
export MODAL_GPU=B200
```

The Modal app includes:

- **TE NVFP4 Linear** forward/backward
- **benchmark** vs BF16
- **tiny transformer training** (NVFP4 / BF16 / cooldown)

Outputs are saved locally to `reports/`:

- `part2_benchmark.json`
- `part2_train_nvfp4.json` + `part2_train_nvfp4.png`
- `part2_train_bf16.json` + `part2_train_bf16.png`
- `part2_train_cooldown.json` + `part2_train_cooldown.png`

Customize runs:

```bash
modal run modal_app.py --runs nvfp4,bf16 --steps 200
```

---

## Hybrid Suite (Engram + mHC + MoE + NVFP4 on GPU)

Runs a **byte‑level** WikiText subset to highlight Engram’s N‑gram memory in a real corpus.

```bash
modal run modal_app.py --runs=hybrid --steps=800 --tokenization=byte --no-benchmark --no-sweep
```

Outputs:

- `reports/hybrid_suite.json`
- `reports/hybrid_loss.png`
- `reports/hybrid_throughput.png`
- `reports/hybrid_alloc.png`
- `reports/hybrid_long_context.png`
- `reports/hybrid_gate.png`
- `reports/hybrid_table.png`
- `reports/hybrid_stability.png`
- `reports/POSTER.png`

You can tune memory size / gating:

```bash
modal run modal_app.py --runs=hybrid --steps=1200 --tokenization=byte --engram-buckets=16384 --engram-heads=8
```

---

## Safety Suite (Exchange‑Aware + Probe‑Cascade)

Demonstrates **exchange‑aware safety** with a **probe‑cascade** that preserves throughput. Uses a safe synthetic dataset (no harmful content) to model:

- **Reconstruction attacks** (fragments assembled across context)
- **Obfuscation attacks** (rule‑based encoding)

Run on GPU:

```bash
modal run modal_app.py --runs=safety --no-benchmark --no-sweep --no-hybrid --safety --safety-samples=5000 --safety-steps=600 --safety-probe-steps=200 --safety-seq-len=256
```

Outputs:

- `reports/safety_suite.json`
- `reports/safety_exchange.png`
- `reports/safety_pareto.png`
- `reports/safety_latency.png`
- `reports/safety_poster.png`

---

## Twist Experiments

```bash
# Adaptive global scaling (percentile) + cooldown schedule
python scripts/twist_adaptive_scaling.py --steps 200 --cooldown-frac 0.15 --global-amax percentile --pct 0.999
```

This demonstrates a simple custom recipe on top of NVFP4 while staying faithful to the core protocol.

---

## DeepSeek Add‑ons (Toy)

These are **small, CPU‑safe reproductions** to show the ideas, not full‑scale training runs.

```bash
# Engram‑style conditional memory (hashed N‑gram lookup + gating)
python scripts/engram_toy.py

# mHC vs HC stability toy demo (Sinkhorn‑constrained residual mixing)
python scripts/mhc_toy.py

# Hybrid efficiency demo: Engram + mHC + NVFP4 in one toy LM
python scripts/hybrid_efficiency.py
```

Outputs:

- `reports/engram_toy.png` + `reports/engram_gate.png`
- `reports/mhc_toy.png`
- `reports/hybrid_efficiency.png` + `reports/hybrid_gate.png`

---

## Project Layout

```
src/nvfp4/            Core NVFP4 utilities (quantization, RHT, SR, metrics)
src/engram/           Engram toy module (conditional memory)
src/mhc/              mHC toy module (manifold‑constrained HC)
scripts/             Demos + reports
modal_app.py          TransformerEngine NVFP4 on Modal
reports/             Saved plots + JSON metrics
```

---

## Notes

- **CPU‑safe:** Part 1 runs on CPU and does not require float8 hardware.
- **GPU‑optional:** Part 2 requires CUDA + Blackwell for true NVFP4 throughput.
- **Educational > production:** The reference quantizer is intentionally explicit.

---

## Hiring Manager TL;DR

This repo demonstrates end‑to‑end, GPU‑verified work across **numerics**, **systems**, and **model design**:

- **Low‑precision training:** NVFP4 + TE on Blackwell with real benchmarks and crossover plots.
- **Hybrid architecture design:** Engram (hashed N‑gram memory) + mHC stability + MoE.
- **Hardware‑aware metrics:** throughput, latency, long‑context scaling, and stability (Amax/grad‑norm).
- **Ablation + scaling:** allocation sweeps, component ablation matrix, and code‑domain wins.

One‑liner: **GPU‑verified hybrid LM system coupling novel memory/gating with stable residual mixing and NVFP4 throughput.**

Artifacts to review:
- Poster (core run): `artifacts/final_codeparrot_2500_mem_POSTER.png`
- Table: `artifacts/final_codeparrot_2500_mem_table.md`
- Poster (bigger model run): `artifacts/final_codeparrot_big_POSTER.png`
- DDP smoke test: `artifacts/ddp_smoke.json` + `artifacts/ddp_smoke.png`
- NVFP4 TE benchmarks: `artifacts/part2_benchmark.png` + `artifacts/part2_sweep.png`
- Safety poster: `artifacts/safety_poster.png`
- Safety metrics + plots: `artifacts/safety_suite.json`, `artifacts/safety_exchange.png`, `artifacts/safety_pareto.png`, `artifacts/safety_latency.png`, `artifacts/safety_budget.png`

---

## License

MIT (demo code). Adjust as needed.
