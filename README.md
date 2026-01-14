# NVFP4 Pretraining Playground (Part 1 + Part 2)

A practical, show‑your‑work demo of NVFP4 mixed‑precision training. This repo has:

- **Part 1 (Theory → Practice):** A reference NVFP4 quantization pipeline (global + block scaling), Random Hadamard Transform (RHT), and Stochastic Rounding (SR), with correctness + error metrics and plots.
- **Part 2 (Systems):** A **Modal** app that runs **TransformerEngine** NVFP4 on a Blackwell GPU, with benchmarks and profiling hooks.
- **Twist:** Adaptive global scaling + precision cooldown schedule (NVFP4 → BF16) to demonstrate a custom recipe.

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

## Twist Experiments

```bash
# Adaptive global scaling (percentile) + cooldown schedule
python scripts/twist_adaptive_scaling.py --steps 200 --cooldown-frac 0.15 --global-amax percentile --pct 0.999
```

This demonstrates a simple custom recipe on top of NVFP4 while staying faithful to the core protocol.

---

## Project Layout

```
src/nvfp4/            Core NVFP4 utilities (quantization, RHT, SR, metrics)
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

## License

MIT (demo code). Adjust as needed.
