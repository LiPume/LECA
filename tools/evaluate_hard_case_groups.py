#!/usr/bin/env python3
"""按 Hard Test 场景初标评估八组消融；FP/FN 固定 conf=.25、IoU=.5。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes): return np.empty(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def counts(predictions: np.ndarray, targets: np.ndarray) -> tuple[int, int, int]:
    used = set(); tp = 0
    for box in predictions:
        values = iou(box, targets)
        if len(values):
            index = int(values.argmax())
            if values[index] >= .5 and index not in used: used.add(index); tp += 1
    return tp, len(predictions) - tp, len(targets) - tp


def targets(label: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if label.exists():
        for line in label.read_text().splitlines():
            values = line.split()
            if len(values) != 5: continue
            _, x, y, w, h = map(float, values)
            boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="0"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; base = root / "runs_repro"
    checkpoints = {
        "eca": base / "mechanism_smoke/eca_seed42/weights/best.pt",
        **{name: base / f"factorial_ablation/{name}_seed{args.seed}_full/weights/best.pt" for name in ("var", "rec", "bri", "var_rec", "var_bri", "rec_bri")},
        "full": base / "mechanism_smoke/leca_seed42/weights/best.pt",
    }
    image_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    label_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/labels/test"
    metadata = list(csv.DictReader((root / "metadata/hard_case_index.csv").open()))
    groups = defaultdict(list)
    for row in metadata: groups[row["case_type"]].append(image_dir / row["image_id"])
    generated = root / "artifacts/hard_case_subsets"; generated.mkdir(parents=True, exist_ok=True)
    yamls = {}
    for category, images in groups.items():
        listing = generated / f"{category}.txt"; listing.write_text("\n".join(map(str, images)) + "\n")
        config = generated / f"{category}.yaml"
        config.write_text(yaml.safe_dump({"path": "/", "train": str(listing), "val": str(listing), "test": str(listing), "names": {0: "screw"}}, sort_keys=False, allow_unicode=True))
        yamls[category] = config
    output = base / "hard_case_groups"; output.mkdir(parents=True, exist_ok=True); rows = []
    for model_name, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint)); error_counts = defaultdict(lambda: [0, 0, 0])
        all_images = [str(image_dir / row["image_id"]) for row in metadata]
        categories = {row["image_id"]: row["case_type"] for row in metadata}
        for result in model.predict(all_images, imgsz=640, conf=.25, device=args.device, verbose=False, stream=True):
            image_path = Path(result.path); image = cv2.imread(str(image_path)); gt = targets(label_dir / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
            pred = result.boxes.xyxy.detach().cpu().numpy() if len(result.boxes) else np.empty((0, 4), dtype=np.float32)
            count = counts(pred, gt); category = categories[image_path.name]
            for index, value in enumerate(count): error_counts[category][index] += value
        for category, config in yamls.items():
            metrics = model.val(data=str(config), split="test", imgsz=640, batch=16, device=args.device, conf=.001, iou=.7,
                                plots=False, project=str(output), name=f"{model_name}_{category}_seed{args.seed}", exist_ok=False)
            values = metrics.results_dict; tp, fp, fn = error_counts[category]
            rows.append({"model": model_name, "seed": args.seed, "case_type": category, "images": len(groups[category]),
                         "precision": values["metrics/precision(B)"], "recall": values["metrics/recall(B)"],
                         "map50": values["metrics/mAP50(B)"], "map50_95": values["metrics/mAP50-95(B)"],
                         "tp_conf025_iou05": tp, "fp_conf025_iou05": fp, "fn_conf025_iou05": fn,
                         "annotation_status": "AI_single_reviewer_raw_image_initial_labels"})
    local = output / f"hard_case_results_seed{args.seed}.csv"; tracked = root / "reports/hard_case_results.csv"
    for path in (local, tracked):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(f"写入 {len(rows)} 行: {tracked}")


if __name__ == "__main__":
    main()
