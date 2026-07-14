#!/usr/bin/env python3
"""生成只含 Hard Test 原图和文件名的本地初标联系表；不显示模型预测。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def tile(image: np.ndarray, name: str, width: int = 360, height: int = 270) -> np.ndarray:
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    scale = min(width / image.shape[1], (height - 34) / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    y = 34 + (height - 34 - resized.shape[0]) // 2; x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    image_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    output = root / "artifacts/hard_case_annotation"
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    for start in range(0, len(images), 12):
        group = images[start:start + 12]
        panels = [tile(cv2.imread(str(path)), path.name) for path in group]
        while len(panels) < 12:
            panels.append(np.full_like(panels[0], 28))
        sheet = np.concatenate([np.concatenate(panels[row:row + 4], axis=1) for row in range(0, 12, 4)], axis=0)
        sheet_name = f"sheet_{start // 12 + 1:02d}.jpg"
        cv2.imwrite(str(output / sheet_name), sheet)
        rows.extend({"image_id": path.name, "sheet": sheet_name, "case_type": "", "notes": ""} for path in group)
    with (output / "annotation_working.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"生成 {len(images)} 张图、{(len(images) + 11) // 12} 张联系表: {output}")


if __name__ == "__main__":
    main()
