#!/usr/bin/env python3
"""用 Hard Test 检验 LECA 统计量的语义边界；不修改模型、权重或数据。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


LECA_LAYERS = ["model.2.eca", "model.4.eca", "model.6.eca", "model.8.eca",
               "model.13.eca", "model.16.eca", "model.19.eca", "model.22.eca"]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool); n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = average_ranks(scores)[labels].sum()
    return float((rank_sum - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def bootstrap_mean(values: list[float], seed: int = 42, repeats: int = 2000) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    estimates = np.array([rng.choice(array, len(array), replace=True).mean() for _ in range(repeats)])
    return float(array.mean()), float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975)), len(array)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int = 42, repeats: int = 2000) -> tuple[float, float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y); x = x[valid]; y = y[valid]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), len(x)
    rng = np.random.default_rng(seed); estimates = []
    for _ in range(repeats):
        indices = rng.integers(0, len(x), len(x)); value = spearman(x[indices], y[indices])
        if np.isfinite(value): estimates.append(value)
    return spearman(x, y), float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975)), len(x)


def summary_row(experiment: str, layer: str, metric: str, stats: tuple[float, float, float, int], interpretation: str) -> dict:
    value, low, high, count = stats
    return {"experiment": experiment, "layer": layer, "metric": metric, "value": value,
            "ci_low": low, "ci_high": high, "n_images": count, "interpretation": interpretation}


def load_gt_mask(label_path: Path, raw_hw: tuple[int, int], input_hw: tuple[int, int], feature_hw: tuple[int, int]) -> np.ndarray:
    raw_h, raw_w = raw_hw; input_h, input_w = input_hw
    raw = np.zeros((raw_h, raw_w), dtype=np.uint8)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            _, cx, cy, width, height = map(float, values)
            x1 = max(0, round((cx - width / 2) * raw_w)); x2 = min(raw_w, round((cx + width / 2) * raw_w))
            y1 = max(0, round((cy - height / 2) * raw_h)); y2 = min(raw_h, round((cy + height / 2) * raw_h))
            raw[y1:y2, x1:x2] = 1
    ratio = min(input_h / raw_h, input_w / raw_w)
    new_w, new_h = round(raw_w * ratio), round(raw_h * ratio)
    resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    padded = np.zeros((input_h, input_w), dtype=np.uint8)
    left = round((input_w - new_w) / 2 - 0.1); top = round((input_h - new_h) / 2 - 0.1)
    padded[top:top + new_h, left:left + new_w] = resized
    return cv2.resize(padded, (feature_hw[1], feature_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


class Capture:
    def __init__(self, yolo: YOLO):
        self.features: dict[str, torch.Tensor] = {}; self.input_hw = (0, 0); self.handles = []
        modules = dict(yolo.model.named_modules())
        self.handles.append(yolo.model.model[0].register_forward_pre_hook(self._input_hook))
        for layer in LECA_LAYERS:
            self.handles.append(modules[layer].register_forward_hook(self._feature_hook(layer)))

    def _input_hook(self, _module, inputs):
        self.input_hw = tuple(inputs[0].shape[-2:])

    def _feature_hook(self, layer):
        def hook(_module, inputs, _output):
            self.features[layer] = inputs[0].detach().float().cpu()
        return hook

    def clear(self):
        self.features.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def match_mean(image: np.ndarray, multiplier: np.ndarray, target: float) -> np.ndarray:
    result = image.astype(np.float32) * multiplier[..., None]
    for _ in range(5):
        current = float(cv2.cvtColor(np.uint8(np.clip(result, 0, 255)), cv2.COLOR_BGR2GRAY).mean())
        result *= target / max(current, 1e-6)
    return np.uint8(np.clip(result, 0, 255))


def variants(image: np.ndarray) -> dict[str, np.ndarray]:
    gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(((xx - (w - 1) / 2) / max(w / 2, 1)) ** 2 + ((yy - (h - 1) / 2) / max(h / 2, 1)) ** 2)
    radial = np.clip(.55 + .9 * radius, .55, 1.45)
    linear = np.tile(np.linspace(.55, 1.45, w, dtype=np.float32), (h, 1))
    spot = 1 + 1.8 * np.exp(-(((xx - .72 * w) / max(.08 * w, 1)) ** 2 + ((yy - .30 * h) / max(.08 * h, 1)) ** 2) / 2)
    return {
        "original": image,
        "global_dark_0.5": np.uint8(np.clip(image.astype(np.float32) * .5, 0, 255)),
        "global_bright_1.35": np.uint8(np.clip(image.astype(np.float32) * 1.35, 0, 255)),
        "radial_equal_mean": match_mean(image, radial, gray_mean),
        "linear_equal_mean": match_mean(image, linear, gray_mean),
        "local_highlight_equal_mean": match_mean(image, spot, gray_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--brightness-images", type=int, default=24)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    label_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/labels/test"
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    output = root / "artifacts/mechanism_diagnostics"; output.mkdir(parents=True, exist_ok=True)
    report = root / "reports/mechanism_diagnostics_seed42.csv"
    model = YOLO(str(root / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt"))
    capture = Capture(model)

    # 诊断 1：方差/均值能否直接判定目标聚焦通道。按图像、按层统计，避免把通道当独立样本。
    channel_rows = []
    image_layer_metrics = []
    for image_path in images:
        image = cv2.imread(str(image_path)); capture.clear()
        model.predict(image, imgsz=640, device=args.device, conf=.25, verbose=False, save=False)
        for layer in LECA_LAYERS:
            x = capture.features[layer][0]
            mask = load_gt_mask(label_dir / f"{image_path.stem}.txt", image.shape[:2], capture.input_hw, tuple(x.shape[-2:]))
            if not mask.any() or mask.all():
                continue
            energy = x.abs().numpy(); inside = energy[:, mask].mean(1); outside = energy[:, ~mask].mean(1)
            focus = np.log((inside + 1e-6) / (outside + 1e-6))
            mu = x.mean((1, 2)).numpy(); variance = ((x * x).mean((1, 2)) - x.mean((1, 2)) ** 2).numpy()
            target = focus > math.log(2); background = focus < 0
            high_var = variance >= np.quantile(variance, .8); low_mu = mu <= np.quantile(mu, .2)
            image_layer_metrics.append({
                "image_id": image_path.name, "layer": layer,
                "spearman_variance_vs_target_focus": spearman(variance, focus),
                "auc_variance_predict_target_focused": auc(variance, target),
                "high_variance_target_focused_fraction": float(target[high_var].mean()),
                "high_variance_background_focused_fraction": float(background[high_var].mean()),
                "spearman_negative_mu_vs_target_focus": spearman(-mu, focus),
                "auc_negative_mu_predict_target_focused": auc(-mu, target),
                "low_mu_target_focused_fraction": float(target[low_mu].mean()),
                "all_target_focused_fraction": float(target.mean()),
            })
            for channel in range(len(mu)):
                channel_rows.append({"image_id": image_path.name, "layer": layer, "channel": channel, "mu": float(mu[channel]),
                                     "variance": float(variance[channel]), "focus_log_ratio": float(focus[channel]),
                                     "target_focused": int(target[channel]), "background_focused": int(background[channel]),
                                     "high_variance": int(high_var[channel]), "low_mu": int(low_mu[channel])})
    with (output / "channel_semantics_raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(channel_rows[0])); writer.writeheader(); writer.writerows(channel_rows)

    with (output / "channel_semantics_by_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(image_layer_metrics[0])); writer.writeheader(); writer.writerows(image_layer_metrics)
    interpretations = {
        "spearman_variance_vs_target_focus": "方差与目标聚焦程度的图像级单调关联",
        "auc_variance_predict_target_focused": "方差单独识别目标聚焦通道的图像级 AUC；0.5近随机",
        "high_variance_target_focused_fraction": "每张图高方差通道中的目标聚焦比例",
        "high_variance_background_focused_fraction": "每张图高方差通道中的背景更强比例",
        "spearman_negative_mu_vs_target_focus": "低均值与目标聚焦程度的图像级单调关联",
        "auc_negative_mu_predict_target_focused": "低均值单独识别目标聚焦通道的图像级 AUC；0.5近随机",
        "low_mu_target_focused_fraction": "每张图低均值通道中的目标聚焦比例",
        "all_target_focused_fraction": "每张图全部通道的目标聚焦基准比例",
    }
    summaries = []
    for layer in LECA_LAYERS:
        layer_rows = [row for row in image_layer_metrics if row["layer"] == layer]
        for metric, interpretation in interpretations.items():
            summaries.append(summary_row("channel_semantics", layer, metric,
                                         bootstrap_mean([float(row[metric]) for row in layer_rows]), interpretation))

    # 诊断 2：corr 与真实灰度及受控光度变化的关系。
    indices = np.linspace(0, len(images) - 1, min(args.brightness_images, len(images)), dtype=int)
    brightness_rows = []
    modules = dict(model.model.named_modules())
    for index in indices:
        image_path = images[index]; raw = cv2.imread(str(image_path))
        for variant, image in variants(raw).items():
            capture.clear(); model.predict(image, imgsz=640, device=args.device, conf=.25, verbose=False, save=False)
            luminance = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255)
            for layer_name in LECA_LAYERS:
                x = capture.features[layer_name]
                corr = float(torch.sigmoid(-x.mean(1, keepdim=True)).mean())
                module = modules[layer_name]
                mu = x.mean((2, 3), keepdim=True)
                variance = (x * x).mean((2, 3), keepdim=True) - mu * mu
                beta = float(module.beta.detach().cpu())
                w_sup = 1 / (1 + beta * torch.nn.functional.softplus(variance))
                gamma = float(module.gamma.detach().cpu())
                brightness_rows.append({"image_id": image_path.name, "variant": variant, "layer": layer_name,
                                        "gray_mean": luminance, "corr": corr, "gamma": gamma,
                                        "w_bri": 1 + gamma * corr, "mean_variance": float(variance.mean()),
                                        "beta": beta, "mean_w_sup": float(w_sup.mean())})
    capture.close()
    with (output / "brightness_semantics_raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(brightness_rows[0])); writer.writeheader(); writer.writerows(brightness_rows)

    for layer_name in LECA_LAYERS:
        rows = [r for r in brightness_rows if r["layer"] == layer_name]
        natural = [r for r in rows if r["variant"] == "original"]
        summaries.append(summary_row("brightness", layer_name, "natural_spearman_gray_vs_corr",
                                     bootstrap_spearman(np.array([r["gray_mean"] for r in natural]), np.array([r["corr"] for r in natural])),
                                     "自然样本中像素灰度与深层 corr 的关联"))
        by_image = {}
        for row in rows: by_image.setdefault(row["image_id"], {})[row["variant"]] = row
        dark_direction = []; bright_direction = []; dark_wbri_up = []; bright_wbri_down = []
        dark_effect = []; radial_effect = []; linear_effect = []; highlight_effect = []; mean_errors = []
        highlight_variance_up = []; highlight_wsup_down = []; highlight_variance_change = []; highlight_wsup_change = []
        for variants_by_name in by_image.values():
            base = variants_by_name["original"]; dark = variants_by_name["global_dark_0.5"]; bright = variants_by_name["global_bright_1.35"]
            radial = variants_by_name["radial_equal_mean"]; linear = variants_by_name["linear_equal_mean"]
            highlight = variants_by_name["local_highlight_equal_mean"]
            dark_direction.append(dark["corr"] > base["corr"]); bright_direction.append(bright["corr"] < base["corr"])
            dark_wbri_up.append(dark["w_bri"] > base["w_bri"]); bright_wbri_down.append(bright["w_bri"] < base["w_bri"])
            dark_effect.append(abs(dark["corr"] - base["corr"])); radial_effect.append(abs(radial["corr"] - base["corr"])); linear_effect.append(abs(linear["corr"] - base["corr"])); highlight_effect.append(abs(highlight["corr"] - base["corr"]))
            highlight_variance_up.append(highlight["mean_variance"] > base["mean_variance"])
            highlight_wsup_down.append(highlight["mean_w_sup"] < base["mean_w_sup"])
            highlight_variance_change.append(highlight["mean_variance"] - base["mean_variance"])
            highlight_wsup_change.append(highlight["mean_w_sup"] - base["mean_w_sup"])
            mean_errors.extend([abs(radial["gray_mean"] - base["gray_mean"]), abs(linear["gray_mean"] - base["gray_mean"]), abs(highlight["gray_mean"] - base["gray_mean"])])
        values = {
            "dark_expected_direction_fraction": dark_direction,
            "bright_expected_direction_fraction": bright_direction,
            "dark_w_bri_increase_fraction": dark_wbri_up,
            "bright_w_bri_decrease_fraction": bright_wbri_down,
            "mean_abs_corr_change_global_dark": dark_effect,
            "mean_abs_corr_change_radial_equal_mean": radial_effect,
            "mean_abs_corr_change_linear_equal_mean": linear_effect,
            "mean_abs_corr_change_local_highlight_equal_mean": highlight_effect,
            "local_highlight_variance_increase_fraction": highlight_variance_up,
            "local_highlight_w_sup_decrease_fraction": highlight_wsup_down,
            "mean_local_highlight_variance_change": highlight_variance_change,
            "mean_local_highlight_w_sup_change": highlight_wsup_change,
            "equal_mean_gray_error": mean_errors,
        }
        for metric, values_by_image in values.items():
            summaries.append(summary_row("brightness", layer_name, metric, bootstrap_mean(values_by_image), "受控光度诊断"))
        scalars = {"learned_alpha": float(modules[layer_name].alpha.detach().cpu()),
                   "learned_beta": float(modules[layer_name].beta.detach().cpu()),
                   "learned_gamma": float(rows[0]["gamma"])}
        for metric, scalar in scalars.items():
            summaries.append(summary_row("learned_scalar", layer_name, metric, (scalar, scalar, scalar, len(by_image)), "训练后标量；符号决定该层校准方向"))
    with report.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(f"写入汇总: {report}")
    print(f"逐样本原始统计（本地忽略）: {output}")


if __name__ == "__main__":
    main()
