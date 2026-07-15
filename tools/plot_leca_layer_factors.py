#!/usr/bin/env python3
"""绘制 LECA 八位置学习标量与 Hard Test 实际动态因子。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root / "reports/leca_layer_factor_summary_seed42.csv").open(newline="")))
    layers = ["model.2.eca", "model.4.eca", "model.6.eca", "model.8.eca",
              "model.13.eca", "model.16.eca", "model.19.eca", "model.22.eca"]
    by_key = {(row["layer"], row["factor"]): row for row in rows}
    labels = [layer.replace("model.", "L").replace(".eca", "") for layer in layers]
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9), constrained_layout=True)
    x = np.arange(len(layers)); width = .24
    first = [by_key[(layer, "w_sup")] for layer in layers]
    for offset, key, color in [(-width, "alpha", "#3b82f6"), (0, "beta", "#10b981"), (width, "gamma", "#f59e0b")]:
        axes[0].bar(x + offset, [float(row[key]) for row in first], width, label=key, color=color)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("学习后的标量值")
    axes[0].set_title("每个 LECA 位置独立学习的 alpha / beta / gamma")
    axes[0].legend(ncol=3)
    axes[0].grid(axis="y", alpha=.25)

    for factor, color, marker in [("w_sup", "#ef4444", "o"), ("w_rec", "#3b82f6", "s"),
                                   ("w_bri", "#10b981", "^"), ("w_stat", "#7c3aed", "D")]:
        selected = [by_key[(layer, factor)] for layer in layers]
        mean = np.array([float(row["mean"]) for row in selected])
        low = np.array([float(row["p05"]) for row in selected])
        high = np.array([float(row["p95"]) for row in selected])
        axes[1].errorbar(x, mean, yerr=np.vstack((mean - low, high - mean)), label=factor,
                         color=color, marker=marker, capsize=3, linewidth=1.8)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1, label="中性值 1")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("实际乘性因子（均值与 P05–P95）")
    axes[1].set_title("全部 146 张 Hard Test 上的逐层动态因子")
    axes[1].legend(ncol=5, fontsize=9)
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("LECA 八个 C3k2 位置的学习参数与实际作用", fontsize=17, weight="bold")

    output = root / "artifacts/visualizations/leca_layer_factors/leca_layer_factors.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
