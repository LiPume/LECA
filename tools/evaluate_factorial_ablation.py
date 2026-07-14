#!/usr/bin/env python3
"""在固定 Hard Test 上统一评估八组独立重训练消融。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "runs_repro"
    checkpoints = {
        "eca": base / "mechanism_smoke/eca_seed42/weights/best.pt",
        "var": base / f"factorial_ablation/var_seed{args.seed}_full/weights/best.pt",
        "rec": base / f"factorial_ablation/rec_seed{args.seed}_full/weights/best.pt",
        "bri": base / f"factorial_ablation/bri_seed{args.seed}_full/weights/best.pt",
        "var_rec": base / f"factorial_ablation/var_rec_seed{args.seed}_full/weights/best.pt",
        "var_bri": base / f"factorial_ablation/var_bri_seed{args.seed}_full/weights/best.pt",
        "rec_bri": base / f"factorial_ablation/rec_bri_seed{args.seed}_full/weights/best.pt",
        "full": base / "mechanism_smoke/leca_seed42/weights/best.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少消融权重:\n" + "\n".join(missing))
    output = base / "factorial_hard_test"
    rows = []
    for name, checkpoint in checkpoints.items():
        metrics = YOLO(str(checkpoint)).val(
            data=str(root / "ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml"),
            split="test", imgsz=640, batch=16, device=args.device, conf=.001, iou=.7, plots=True,
            project=str(output), name=f"{name}_seed{args.seed}", exist_ok=False,
        )
        values = metrics.results_dict
        rows.append({"combination": name, "seed": args.seed,
                     "precision": values["metrics/precision(B)"], "recall": values["metrics/recall(B)"],
                     "map50": values["metrics/mAP50(B)"], "map50_95": values["metrics/mAP50-95(B)"],
                     "checkpoint": str(checkpoint.relative_to(root)), "status": "controlled_retrained_ablation_observation"})
    summary = output / f"factorial_ablation_seed{args.seed}.csv"
    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
