from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _path import ROOT


def main():
    reports = ROOT / "reports"
    data = json.loads((reports / "hybrid_suite.json").read_text())

    rows = []
    for name, res in data["results"].items():
        hist = res["history"]
        rows.append(
            {
                "Model": name,
                "Params": res["params"],
                "ValLoss": hist["val_loss"][-1] if hist["val_loss"] else None,
                "PPL": math.exp(hist["val_loss"][-1]) if hist["val_loss"] else None,
                "Tokens/s": res["tokens_per_s"],
                "Step ms": res["avg_step_ms"],
                "Infer ms": res["infer_ms"],
                "Peak GB": res["peak_mem_gb"],
                "Amax": hist["amax"][-1] if hist["amax"] else None,
                "Bigram Acc": res.get("bigram_acc"),
                "Gate Mean": hist["gate_mean"][-1] if hist["gate_mean"] else None,
            }
        )

    df = pd.DataFrame(rows)
    df_path = reports / "hybrid_table.md"
    df_path.write_text(df.to_markdown(index=False))

    # Render table image
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "lines.linewidth": 2.2,
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(reports / "hybrid_table.png", dpi=160)

    # Stability plot
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for name in ["hc", "hybrid_nvfp4"]:
        hist = data["results"][name]["history"]
        ax2.plot(hist["step"], hist["grad_norm"], label=f"{name} grad norm")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Grad Norm")
    ax2.set_title("Stability: HC vs mHC+NVFP4")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(reports / "hybrid_stability.png", dpi=160)

    print("Wrote hybrid_table and stability plot")


if __name__ == "__main__":
    main()
