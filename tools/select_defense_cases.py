#!/usr/bin/env python3
"""从 Hard Test 自动挑选答辩案例；所有图片与逐图结果只保存在本地 artifacts。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, box[2] - box[0]) * np.maximum(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    used: set[int] = set(); tp = 0
    for box in pred:
        candidates = iou(box, gt)
        if len(candidates):
            index = int(candidates.argmax())
            if candidates[index] >= 0.5 and index not in used:
                used.add(index); tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def labels(label_path: Path, width: int, height: int) -> np.ndarray:
    output = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            _, x, y, w, h = map(float, values)
            output.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return np.asarray(output, dtype=np.float32).reshape(-1, 4)


def draw(image: np.ndarray, gt: np.ndarray, prediction: np.ndarray, title: str, result: tuple[int, int, int]) -> np.ndarray:
    panel = image.copy()
    for x1, y1, x2, y2 in gt.astype(int):
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 220, 0), 3)
    for x1, y1, x2, y2 in prediction.astype(int):
        cv2.rectangle(panel, (x1, y1), (x2, y2), (20, 20, 230), 2)
    tp, fp, fn = result
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(panel, f"{title}: TP={tp} FP={fp} FN={fn}", (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    image_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    label_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/labels/test"
    output = root / "artifacts/visualizations/defense_selected"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "Baseline": root / "runs_repro/mechanism_smoke/baseline_seed42/weights/best.pt",
        "ECA": root / "runs_repro/mechanism_smoke/eca_seed42/weights/best.pt",
        "LECA": root / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt",
    }
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint)); entries = {}
        for result in model.predict([str(p) for p in images], imgsz=640, conf=.25, device=0, verbose=False, stream=True):
            entries[Path(result.path).name] = result.boxes.xyxy.detach().cpu().numpy() if result.boxes else np.empty((0, 4), dtype=np.float32)
        predictions[name] = entries
    rows = []
    for image_path in images:
        image = cv2.imread(str(image_path)); gt = labels(label_dir / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
        row: dict[str, object] = {"image_id": image_path.name, "gt": len(gt)}
        for name in checkpoints:
            tp, fp, fn = counts(predictions[name][image_path.name], gt)
            row.update({f"{name}_tp": tp, f"{name}_fp": fp, f"{name}_fn": fn, f"{name}_score": tp-fp-fn})
        rows.append(row)
    def choose(key, reverse=True, excluded=set()):
        candidates = [r for r in rows if r["image_id"] not in excluded]
        return sorted(candidates, key=key, reverse=reverse)[0]
    gain = choose(lambda r: r["LECA_score"] - max(r["Baseline_score"], r["ECA_score"]))
    excluded = {gain["image_id"]}
    fp_reduction = choose(lambda r: max(r["Baseline_fp"], r["ECA_fp"]) - r["LECA_fp"], excluded=excluded)
    excluded.add(fp_reduction["image_id"])
    limitation = choose(lambda r: r["LECA_fn"] + r["LECA_fp"] - min(r["Baseline_fn"] + r["Baseline_fp"], r["ECA_fn"] + r["ECA_fp"]), excluded=excluded)
    selected = [("检出增益案例", gain), ("误检减少案例", fp_reduction), ("仍有局限案例", limitation)]
    with (output / "all_case_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "selected_images.txt").open("w") as file:
        for category, row in selected:
            file.write(f"{row['image_id']}\n")
            image_path = image_dir / str(row["image_id"]); image = cv2.imread(str(image_path))
            gt = labels(label_dir / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
            panels = [draw(image, gt, predictions[name][image_path.name], name, (int(row[f"{name}_tp"]), int(row[f"{name}_fp"]), int(row[f"{name}_fn"]))) for name in checkpoints]
            combined = np.concatenate(panels, axis=1)
            cv2.putText(combined, category, (12, combined.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(output / f"{category}_{image_path.stem}.jpg"), combined)
    with (output / "selected_case_notes.csv").open("w", newline="") as file:
        fields = ["category", *rows[0].keys()]; writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for category, row in selected: writer.writerow({"category": category, **row})


if __name__ == "__main__":
    main()
