#!/usr/bin/env python3
"""生成更常见的平滑多尺度响应图和尺度分类分数 Grad-CAM；图片仅落本地。"""

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
from ultralytics.data.augment import LetterBox


LEVELS = {
    "P3": (16, 8),
    "P4": (19, 16),
    "P5": (22, 32),
}


def configure_font() -> None:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def resolve_device(value: str) -> torch.device:
    if value.isdigit():
        return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.quantile(values, (0.02, 0.98))
    return np.clip((values - low) / (high - low + 1e-8), 0, 1)


def restore_map(values: np.ndarray, raw_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    raw_h, raw_w = raw_hw
    ratio = min(640 / raw_h, 640 / raw_w)
    new_w, new_h = round(raw_w * ratio), round(raw_h * ratio)
    pad_w, pad_h = (640 - new_w) / 2, (640 - new_h) / 2
    left, top = round(pad_w - 0.1), round(pad_h - 0.1)
    dense = cv2.resize(values, (640, 640), interpolation=interpolation)
    dense = dense[top:top + new_h, left:left + new_w]
    return cv2.resize(dense, (raw_w, raw_h), interpolation=interpolation)


def valid_grid_masks(grid_h: int, grid_w: int, raw_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """返回落在真实图像而非 square-letterbox padding 内的特征网格中心。"""
    raw_h, raw_w = raw_hw
    ratio = min(640 / raw_h, 640 / raw_w)
    new_w, new_h = round(raw_w * ratio), round(raw_h * ratio)
    left, top = (640 - new_w) / 2, (640 - new_h) / 2
    xs = (np.arange(grid_w) + .5) * 640 / grid_w
    ys = (np.arange(grid_h) + .5) * 640 / grid_h
    return (ys >= top) & (ys < top + new_h), (xs >= left) & (xs < left + new_w)


def cell_to_raw(cell_y: int, cell_x: int, grid_h: int, grid_w: int, raw_hw: tuple[int, int]) -> tuple[float, float]:
    raw_h, raw_w = raw_hw
    ratio = min(640 / raw_h, 640 / raw_w)
    new_w, new_h = round(raw_w * ratio), round(raw_h * ratio)
    left, top = (640 - new_w) / 2, (640 - new_h) / 2
    input_x, input_y = (cell_x + .5) * 640 / grid_w, (cell_y + .5) * 640 / grid_h
    return (input_x - left) / ratio, (input_y - top) / ratio


def blend(raw_rgb: np.ndarray, heat: np.ndarray, cmap: str, alpha: float = 0.42) -> np.ndarray:
    color = plt.colormaps[cmap](heat)[..., :3]
    raw = raw_rgb.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * raw + alpha * color, 0, 1)


def load_input(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    letterboxed = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=image_bgr)
    tensor = torch.from_numpy(letterboxed[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    tensor.requires_grad_(True)
    return tensor


def read_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for line in path.read_text().splitlines() if path.exists() else []:
        fields = line.split()
        if len(fields) < 5:
            continue
        _, cx, cy, bw, bh = map(float, fields[:5])
        boxes.append((
            round((cx - bw / 2) * width), round((cy - bh / 2) * height),
            round((cx + bw / 2) * width), round((cy + bh / 2) * height),
        ))
    return boxes


def draw_boxes(axis, boxes: list[tuple[int, int, int, int]]) -> None:
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        axis.text(x1, max(0, y1 - 4), f"GT{index}", color="lime", fontsize=8, weight="bold",
                  bbox={"facecolor": "black", "alpha": .5, "edgecolor": "none", "pad": 1})


def render(model: YOLO, image_path: Path, label_path: Path, output_dir: Path, device: str) -> list[dict[str, object]]:
    raw_bgr = cv2.imread(str(image_path))
    if raw_bgr is None:
        raise FileNotFoundError(image_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    raw_h, raw_w = raw_rgb.shape[:2]
    boxes = read_boxes(label_path, raw_w, raw_h)
    torch_device = resolve_device(device)
    tensor = load_input(raw_bgr, torch_device)

    torch_model = model.model.to(torch_device).eval()
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for level, (index, _) in LEVELS.items():
        def hook(_module, _inputs, output, level=level):
            output.retain_grad()
            activations[level] = output
        handles.append(torch_model.model[index].register_forward_hook(hook))

    decoded, raw_outputs = torch_model(tensor)
    del decoded
    scores = {level: raw_outputs[i][:, -1].sigmoid().amax() for i, level in enumerate(LEVELS)}
    energy_maps: dict[str, np.ndarray] = {}
    gradcam_maps: dict[str, np.ndarray] = {}
    peak_cells: dict[str, tuple[int, int]] = {}

    for level, activation in activations.items():
        rms = torch.sqrt(torch.mean(activation[0].detach() ** 2, dim=0) + 1e-12).cpu().numpy()
        energy_maps[level] = normalize(restore_map(rms, (raw_h, raw_w), cv2.INTER_CUBIC))

    for level_index, level in enumerate(LEVELS):
        torch_model.zero_grad(set_to_none=True)
        tensor.grad = None
        for activation in activations.values():
            activation.grad = None
        scores[level].backward(retain_graph=level_index < len(LEVELS) - 1)
        activation = activations[level]
        gradient = activation.grad
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0].detach().cpu().numpy()
        gradcam_maps[level] = normalize(restore_map(cam, (raw_h, raw_w), cv2.INTER_CUBIC))
        score_map = raw_outputs[level_index][0, -1].detach()
        flat_index = int(score_map.argmax())
        peak_cells[level] = (flat_index // score_map.shape[1], flat_index % score_map.shape[1])

    for handle in handles:
        handle.remove()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), constrained_layout=True)
    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title("原图与真实标注")
    draw_boxes(axes[0, 0], boxes)
    axes[1, 0].axis("off")
    axes[1, 0].text(
        .03, .93,
        "上排：全部通道RMS响应\n表示该层总体特征能量\n\n"
        "下排：分类分数Grad-CAM\n由该尺度最高螺栓分数反向传播\n\n"
        "所有热图均使用双线性/三次插值平滑\n颜色只表示各图内部的相对强弱",
        va="top", fontsize=14, linespacing=1.55,
    )
    rows = []
    for col, level in enumerate(LEVELS, start=1):
        activation = activations[level]
        channels, grid_h, grid_w = activation.shape[1:]
        axes[0, col].imshow(blend(raw_rgb, energy_maps[level], "magma", .38))
        axes[0, col].set_title(f"{level} 全通道平滑响应\n{channels}×{grid_h}×{grid_w}")
        score_value = float(scores[level])
        if score_value >= .01:
            axes[1, col].imshow(blend(raw_rgb, gradcam_maps[level], "jet", .45))
            peak_y, peak_x = peak_cells[level]
            raw_x, raw_y = cell_to_raw(peak_y, peak_x, grid_h, grid_w, (raw_h, raw_w))
            axes[1, col].scatter([raw_x], [raw_y], marker="x", s=110, linewidths=2.5, color="cyan")
            axes[1, col].set_title(f"{level} 分类Grad-CAM\n最高螺栓分数={score_value:.4f}，×为候选中心")
        else:
            axes[1, col].imshow(raw_rgb, alpha=.28)
            axes[1, col].text(
                raw_w / 2, raw_h / 2,
                f"该尺度最高螺栓分数仅 {score_value:.4f}\n未形成有效分类候选\n不放大归一化噪声",
                ha="center", va="center", fontsize=14, weight="bold",
                bbox={"facecolor": "white", "alpha": .88, "edgecolor": "#555555", "pad": 8},
            )
            axes[1, col].set_title(f"{level} 分类响应")
        for row in (0, 1):
            draw_boxes(axes[row, col], boxes)
        peak_y, peak_x = peak_cells[level]
        rows.append({
            "image_id": image_path.name,
            "level": level,
            "layer": LEVELS[level][0],
            "stride": LEVELS[level][1],
            "channels": channels,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "top_scale_class_score": float(scores[level].detach().cpu()),
            "top_cell_y": peak_y,
            "top_cell_x": peak_x,
            "cam_target": "maximum_raw_class_probability_within_each_scale",
            "status": "descriptive_gradcam_not_causal_proof",
        })
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(f"真实模型多尺度响应与分类 Grad-CAM：{image_path.name}", fontsize=18, weight="bold")
    output = output_dir / f"{image_path.stem}_pyramid_smooth_gradcam.png"
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)

    # 原始通道值面板：固定按空间标准差选择结构变化最大的6个通道，不按视觉效果人工挑选。
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), constrained_layout=True)
    for row, level in enumerate(LEVELS):
        values = activations[level][0].detach().cpu()
        y_mask, x_mask = valid_grid_masks(values.shape[1], values.shape[2], (raw_h, raw_w))
        valid_values = values[:, torch.from_numpy(y_mask), :][:, :, torch.from_numpy(x_mask)]
        indices = torch.topk(valid_values.flatten(1).std(dim=1), k=6).indices.tolist()
        for col, channel in enumerate(indices):
            channel_map = valid_values[channel].numpy()
            limit = float(np.quantile(np.abs(channel_map), .98)) + 1e-8
            axes[row, col].imshow(channel_map, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="bilinear")
            axes[row, col].set_title(f"{level} 原始ch={channel}\n{channel_map.shape[0]}×{channel_map.shape[1]}")
            axes[row, col].axis("off")
    fig.suptitle("原始特征通道（按空间标准差固定选取Top-6；红正蓝负）", fontsize=17, weight="bold")
    raw_output = output_dir / f"{image_path.stem}_pyramid_raw_channels.png"
    fig.savefig(raw_output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(image_path.name, {level: round(float(scores[level]), 6) for level in LEVELS})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--images", nargs="*", default=["0098_c7fc27a4.jpg", "0061_74dea2c9.jpg", "0010_47f223ba.jpg"])
    args = parser.parse_args()
    configure_font()
    image_dir = ROOT / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    label_dir = ROOT / "ultralytics-main/dataset/hardData/YOLODataset/labels/test"
    output_dir = ROOT / "artifacts/visualizations/pyramid_explainable"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt"))
    rows = []
    for name in args.images:
        image_path = image_dir / name
        rows.extend(render(model, image_path, label_dir / f"{image_path.stem}.txt", output_dir, args.device))
    summary = ROOT / "reports/pyramid_gradcam_summary_seed42.csv"
    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(summary)


if __name__ == "__main__":
    main()
