#!/usr/bin/env python3
"""在 Hard Test 上临时中性化已训练 LECA 分支；仅作推理敏感性分析。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

from ultralytics import YOLO


CONDITIONS = {"full": None, "var_off": "beta", "rec_off": "alpha", "bri_off": "gamma"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    checkpoint = ROOT / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt"
    output = ROOT / "runs_repro/hard_branch_neutralization"
    rows = []
    for condition, scalar in CONDITIONS.items():
        model = YOLO(str(checkpoint))
        if scalar:
            with torch.no_grad():
                for module in model.model.modules():
                    if module.__class__.__name__ == "LECA":
                        getattr(module, scalar).zero_()
        metrics = model.val(
            data=str(ROOT / "ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml"),
            split="test", imgsz=640, batch=args.batch, device=args.device, conf=.001, iou=.7,
            plots=True, project=str(output), name=condition, exist_ok=False,
        )
        values = metrics.results_dict
        rows.append({
            "condition": condition, "neutralized_scalar": scalar or "none", "seed": 42,
            "precision": values["metrics/precision(B)"], "recall": values["metrics/recall(B)"],
            "map50": values["metrics/mAP50(B)"], "map50_95": values["metrics/mAP50-95(B)"],
            "status": "inference_sensitivity_not_independent_retraining",
        })
    tracked = ROOT / "reports/hard_branch_neutralization_seed42.csv"
    local = output / "hard_branch_neutralization_seed42.csv"
    for path in (tracked, local):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
