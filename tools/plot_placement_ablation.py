#!/usr/bin/env python3
"""将小型位置消融 CSV 绘制为本地答辩图；图片目录由 .gitignore 排除。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "reports/placement_ablation_results.csv"
    output = root / "artifacts/visualizations/placement_ablation/placement_ablation_metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as file:
        rows = list(csv.DictReader(file))

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    cjk_font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = ["无 LECA", "仅 Backbone", "仅融合路径", "仅 P3/P4/P5", "全部 8 处"]
    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP@0.5"),
        ("map50_95", "mAP@0.5:0.95"),
    ]
    colors = ["#9ca3af", "#60a5fa", "#34d399", "#f59e0b", "#ef4444"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_ylim(max(0.45, min(values) - 0.06), 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=18)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("LECA 插入位置消融（Hard Test，seed=42）", fontsize=17, weight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
