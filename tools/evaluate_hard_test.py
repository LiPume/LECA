#!/usr/bin/env python3
"""Evaluate matched controlled models on the local hard test set only."""

from __future__ import annotations

import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoints = {
        "baseline": root / "runs_repro/mechanism_smoke/baseline_seed42/weights/best.pt",
        "eca": root / "runs_repro/mechanism_smoke/eca_seed42/weights/best.pt",
        "leca": root / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt",
    }
    output = root / "runs_repro/hard_test_controlled"
    rows = []
    for name, checkpoint in checkpoints.items():
        metrics = YOLO(str(checkpoint)).val(
            data=str(root / "ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml"),
            split="test",
            imgsz=640,
            batch=16,
            device=0,
            conf=0.001,
            iou=0.7,
            plots=True,
            project=str(output),
            name=name,
            exist_ok=False,
        )
        values = metrics.results_dict
        rows.append({
            "model": name,
            "precision": values["metrics/precision(B)"],
            "recall": values["metrics/recall(B)"],
            "map50": values["metrics/mAP50(B)"],
            "map50_95": values["metrics/mAP50-95(B)"],
            "status": "controlled_one_seed_hard_test_observation",
        })
    with (output / "hard_test_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
