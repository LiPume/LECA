#!/usr/bin/env python3
"""在固定 Hard Test 上评估 LECA 插入位置消融并生成小型汇总 CSV。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

from ultralytics import YOLO


PLACEMENT_META = {
    "none0": ("无 LECA", "", 0),
    "backbone4": ("仅 Backbone", "2|4|6|8", 4),
    "fusion4": ("仅多尺度融合路径", "13|16|19|22", 4),
    "scales3": ("仅 P3/P4/P5 输出", "16|19|22", 3),
    "full8": ("Backbone 与融合路径全覆盖", "2|4|6|8|13|16|19|22", 8),
}
UNFUSED_PARAMS = {
    "none0": 2_624_080,
    "backbone4": 2_624_110,
    "fusion4": 2_624_110,
    "scales3": 2_624_102,
    "full8": 2_624_140,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    root = ROOT
    base = root / "runs_repro"
    checkpoints = {
        "none0": base / "mechanism_smoke/baseline_seed42/weights/best.pt",
        "backbone4": base / f"placement_ablation/backbone4_seed{args.seed}_full/weights/best.pt",
        "fusion4": base / f"placement_ablation/fusion4_seed{args.seed}_full/weights/best.pt",
        "scales3": base / f"placement_ablation/scales3_seed{args.seed}_full/weights/best.pt",
        "full8": base / "mechanism_smoke/leca_seed42/weights/best.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少位置消融权重:\n" + "\n".join(missing))

    output = base / "placement_hard_test"
    rows = []
    for name, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(root / "ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml"),
            split="test",
            imgsz=640,
            batch=args.batch,
            device=args.device,
            conf=.001,
            iou=.7,
            plots=True,
            project=str(output),
            name=f"{name}_seed{args.seed}",
            exist_ok=False,
        )
        values = metrics.results_dict
        description, indices, count = PLACEMENT_META[name]
        rows.append({
            "placement": name,
            "description": description,
            "leca_indices": indices,
            "leca_count": count,
            "seed": args.seed,
            "precision": values["metrics/precision(B)"],
            "recall": values["metrics/recall(B)"],
            "map50": values["metrics/mAP50(B)"],
            "map50_95": values["metrics/mAP50-95(B)"],
            # 使用构建时未融合模型参数量，与主结果表统计口径一致；加载 best.pt 后模型可能已 fuse。
            "params": UNFUSED_PARAMS[name],
            "checkpoint": str(checkpoint.relative_to(root)),
            "status": "controlled_retrained_placement_ablation_observation",
        })

    local = output / f"placement_ablation_seed{args.seed}.csv"
    tracked = root / "reports/placement_ablation_results.csv"
    for path in (local, tracked):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
