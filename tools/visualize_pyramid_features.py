#!/usr/bin/env python3
"""导出已训练 LECA 的 P3/P4/P5 全通道能量图；图片仅保存到 ignored artifacts。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

from ultralytics import YOLO


LAYERS = {
    "P3": ("model.16", 8),
    "P4": ("model.19", 16),
    "P5": ("model.22", 32),
}
DEFAULT_IMAGES = ("0098_c7fc27a4.jpg", "0061_74dea2c9.jpg", "0010_47f223ba.jpg")


def configure_font() -> None:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, cx, cy, bw, bh = map(float, fields[:5])
        x1 = max(0, round((cx - bw / 2) * width))
        y1 = max(0, round((cy - bh / 2) * height))
        x2 = min(width, round((cx + bw / 2) * width))
        y2 = min(height, round((cy + bh / 2) * height))
        boxes.append((x1, y1, x2, y2))
    return boxes


def capture_features(model: YOLO, image_path: Path, device: str) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    modules = dict(model.model.named_modules())
    handles = []
    for level, (layer, _) in LAYERS.items():
        def hook(_module, _inputs, output, level=level):
            # predictor 首次调用可能先 warm-up；保留最后一次真实图像前向结果。
            captured[level] = output.detach().float().cpu()
        handles.append(modules[layer].register_forward_hook(hook))
    model.predict(str(image_path), imgsz=640, rect=False, device=device, verbose=False, save=False)
    for handle in handles:
        handle.remove()
    return captured


def restore_from_letterbox(native: np.ndarray, raw_hw: tuple[int, int], input_size: int = 640) -> np.ndarray:
    """将 square-letterbox 特征响应去 padding 后映射回原图。"""
    raw_h, raw_w = raw_hw
    ratio = min(input_size / raw_h, input_size / raw_w)
    new_w, new_h = round(raw_w * ratio), round(raw_h * ratio)
    pad_w, pad_h = (input_size - new_w) / 2, (input_size - new_h) / 2
    left, top = round(pad_w - 0.1), round(pad_h - 0.1)
    dense = cv2.resize(native, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    valid = dense[top:top + new_h, left:left + new_w]
    return cv2.resize(valid, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)


def normalize_map(values: np.ndarray) -> np.ndarray:
    low, high = np.quantile(values, (0.02, 0.98))
    return np.clip((values - low) / (high - low + 1e-8), 0, 1)


def draw_boxes(axis, boxes: list[tuple[int, int, int, int]], color: str = "lime") -> None:
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0))
        axis.text(x1, max(0, y1 - 5), f"GT{index}", color=color, fontsize=9, weight="bold",
                  bbox={"facecolor": "black", "alpha": .55, "pad": 1, "edgecolor": "none"})


def focus_ratio(values: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float:
    if not boxes:
        return float("nan")
    mask = np.zeros(values.shape, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    inside = float(values[mask].mean()) if mask.any() else 0.0
    outside = float(values[~mask].mean()) if (~mask).any() else 0.0
    return inside / (outside + 1e-8)


def overlay(raw_rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    colored = plt.colormaps["turbo"](heat)[..., :3]
    raw = raw_rgb.astype(np.float32) / 255.0
    return np.clip(raw * 0.48 + colored * 0.52, 0, 1)


def expanded_crop(box: tuple[int, int, int, int], width: int, height: int, scale: float = 6.0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    crop_w = max(180, (x2 - x1) * scale)
    crop_h = max(180, (y2 - y1) * scale)
    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    left = int(np.clip(cx - crop_w / 2, 0, width - crop_w))
    top = int(np.clip(cy - crop_h / 2, 0, height - crop_h))
    return left, top, int(left + crop_w), int(top + crop_h)


def render_image(model: YOLO, image_path: Path, label_path: Path, output_dir: Path, device: str) -> list[dict[str, object]]:
    raw_bgr = cv2.imread(str(image_path))
    if raw_bgr is None:
        raise FileNotFoundError(image_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    height, width = raw_rgb.shape[:2]
    boxes = read_boxes(label_path, width, height)
    captured = capture_features(model, image_path, device)

    restored: dict[str, np.ndarray] = {}
    normalized: dict[str, np.ndarray] = {}
    ratios: dict[str, float] = {}
    native_shapes: dict[str, tuple[int, int, int]] = {}
    for level, tensor in captured.items():
        # 对全部通道采用同一预注册规则：RMS=sqrt(mean_c(X^2))；不挑选通道。
        energy = torch.sqrt(torch.mean(tensor[0] ** 2, dim=0) + 1e-12).numpy()
        mapped = restore_from_letterbox(energy, (height, width))
        restored[level] = mapped
        normalized[level] = normalize_map(mapped)
        ratios[level] = focus_ratio(mapped, boxes)
        native_shapes[level] = (tensor.shape[1], tensor.shape[2], tensor.shape[3])

    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2), constrained_layout=True)
    axes[0].imshow(raw_rgb)
    axes[0].set_title("原图与真实标注框")
    draw_boxes(axes[0], boxes)
    for axis, level in zip(axes[1:], ("P3", "P4", "P5")):
        axis.imshow(overlay(raw_rgb, normalized[level]))
        draw_boxes(axis, boxes)
        channels, grid_h, grid_w = native_shapes[level]
        axis.set_title(f"{level} 全通道RMS响应\n{channels}×{grid_h}×{grid_w} | GT聚焦比={ratios[level]:.2f}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"LECA 最终多尺度特征：{image_path.name}\n同一聚合规则、同一色表；每层仅作层内 P02–P98 归一化",
                 fontsize=16, weight="bold")
    overview = output_dir / f"{image_path.stem}_p3_p4_p5_overview.png"
    fig.savefig(overview, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if boxes:
        fig, axes = plt.subplots(len(boxes), 4, figsize=(16, 4.6 * len(boxes)), squeeze=False, constrained_layout=True)
        for row, box in enumerate(boxes):
            left, top, right, bottom = expanded_crop(box, width, height)
            axes[row, 0].imshow(raw_rgb[top:bottom, left:right])
            axes[row, 0].set_title(f"GT{row + 1} 原图局部")
            for col, level in enumerate(("P3", "P4", "P5"), start=1):
                cropped = overlay(raw_rgb, normalized[level])[top:bottom, left:right]
                axes[row, col].imshow(cropped)
                channels, grid_h, grid_w = native_shapes[level]
                axes[row, col].set_title(f"{level} 响应局部（{channels}×{grid_h}×{grid_w}）")
            for axis in axes[row]:
                local_x1, local_y1, local_x2, local_y2 = box[0] - left, box[1] - top, box[2] - left, box[3] - top
                axis.add_patch(plt.Rectangle((local_x1, local_y1), local_x2 - local_x1, local_y2 - local_y1,
                                             fill=False, edgecolor="lime", linewidth=2.0))
                axis.axis("off")
        fig.suptitle(f"真实螺栓区域的 P3/P4/P5 对齐局部：{image_path.name}", fontsize=16, weight="bold")
        crops = output_dir / f"{image_path.stem}_p3_p4_p5_gt_crops.png"
        fig.savefig(crops, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"{image_path.name}: " + ", ".join(f"{level}_focus={ratios[level]:.3f}" for level in LAYERS))
    return [
        {
            "image_id": image_path.name,
            "level": level,
            "layer": LAYERS[level][0],
            "stride": LAYERS[level][1],
            "channels": native_shapes[level][0],
            "grid_h": native_shapes[level][1],
            "grid_w": native_shapes[level][2],
            "gt_focus_ratio": ratios[level],
            "aggregation": "sqrt(mean_channel(feature_squared))",
            "status": "three_preselected_cases_descriptive_not_population_inference",
        }
        for level in LAYERS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--images", nargs="*", default=list(DEFAULT_IMAGES))
    args = parser.parse_args()
    configure_font()
    image_dir = ROOT / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    label_dir = ROOT / "ultralytics-main/dataset/hardData/YOLODataset/labels/test"
    output_dir = ROOT / "artifacts/visualizations/pyramid_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt"))
    rows = []
    for name in args.images:
        image_path = image_dir / name
        rows.extend(render_image(model, image_path, label_dir / f"{image_path.stem}.txt", output_dir, args.device))
    summary = ROOT / "reports/pyramid_feature_summary_seed42.csv"
    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(summary)


if __name__ == "__main__":
    main()
