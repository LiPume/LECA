#!/usr/bin/env python3
"""写入基于原图联系表的单人视觉初标；不读取模型预测，不修改图像或检测框。"""

from __future__ import annotations

import csv
from pathlib import Path


GROUPS = [
    (1, 13, "structural_clutter", "S01", "广角底盘结构，孔洞和支架密集"),
    (21, 43, "reflection", "S02", "金属区域存在明显局部高亮，同时伴随孔洞干扰"),
    (44, 69, "mixed", "S03", "孔洞、支架和大型圆形构件共同出现"),
    (70, 100, "hole", "S04", "连续竖幅帧，孔洞是主要相似结构干扰"),
    (114, 194, "hole", "S05", "连续竖幅底盘帧，多处圆孔和黑色圆形结构"),
    (195, 220, "normal", "S06", "螺栓近景较清楚，背景干扰相对较少"),
    (222, 232, "circular_plate", "S07", "螺栓邻近大型圆形金属构件与孔洞"),
]


def label(frame: int) -> tuple[str, str, str]:
    for start, end, case_type, group, note in GROUPS:
        if start <= frame <= end:
            return case_type, group, note
    return "mixed", "SXX", "未落入已审阅连续场景范围，保守标为 mixed"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    image_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    output = root / "metadata/hard_case_index.csv"
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    for image in images:
        frame = int(image.stem.split("_")[0])
        case_type, group, note = label(frame)
        rows.append({"image_id": image.name, "case_type": case_type,
                     "notes": f"{group}；AI单人仅看原图初标，未查看该图预测；{note}；答辩前建议人工复核"})
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("image_id", "case_type", "notes"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    counts = {}
    for row in rows: counts[row["case_type"]] = counts.get(row["case_type"], 0) + 1
    print(f"写入 {len(rows)} 张初标: {output}")
    print(counts)


if __name__ == "__main__":
    main()
