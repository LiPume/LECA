#!/usr/bin/env python3
"""Inference-only LECA branch-neutralization sensitivity analysis.

This tool reloads the checkpoint for every condition and changes scalar values in
memory only. It never edits a checkpoint. Results from the historical duplicated
validation split are marked non-reportable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ultralytics import YOLO


CONDITIONS = {"full": None, "var_off": "beta", "rec_off": "alpha", "bri_off": "gamma"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "runs_repro/mechanism_sensitivity"
    rows = []
    for name, scalar in CONDITIONS.items():
        model = YOLO(str(args.checkpoint))
        if scalar:
            with torch.no_grad():
                for module in model.model.modules():
                    if module.__class__.__name__ == "LECA" and hasattr(module, scalar):
                        getattr(module, scalar).zero_()
        metrics = model.val(
            data=str(root / "ultralytics-main/ultralytics/cfg/models/datasets/screw.yaml"),
            split="val",
            imgsz=640,
            batch=16,
            device=args.device,
            plots=True,
            project=str(output),
            name=name,
            exist_ok=False,
        )
        values = metrics.results_dict
        rows.append(
            {
                "condition": name,
                "neutralized_scalar": scalar or "none",
                "precision": values["metrics/precision(B)"],
                "recall": values["metrics/recall(B)"],
                "map50": values["metrics/mAP50(B)"],
                "map50_95": values["metrics/mAP50-95(B)"],
                "status": "inference_sensitivity_only_not_paper_metrics",
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sensitivity_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
