#!/usr/bin/env python3
"""Target-driven Grad-CAM comparison for baseline vs LECA YOLO checkpoints."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = ROOT / "dataset/hardData/YOLODataset/images/train/0001_b1214a5d.jpg"
DEFAULT_BASELINE = ROOT / "runs/detect/car_bolt_baseline/weights/best.pt"
DEFAULT_LECA = ROOT / "runs/detect/car_bolt_LECA7/weights/best.pt"
DEFAULT_OUT = ROOT / "draw/results"
DEFAULT_LAYERS = [13, 16, 19]


@dataclass
class Target:
    name: str
    raw_index: int
    score: float
    box_xyxy: np.ndarray
    iou_gt: float


def patch_c2f_forward_for_old_baseline():
    """Run old checkpoints that do not have the newly added `eca` attribute."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.eca(out) if hasattr(self, "eca") else out

    C2f.forward = forward


def letterbox(im: np.ndarray, new_shape: int = 640, color: tuple[int, int, int] = (114, 114, 114)):
    shape = im.shape[:2]
    ratio = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top, new_unpad[0], new_unpad[1])


def to_tensor(image_bgr: np.ndarray, imgsz: int, device: torch.device):
    lb, ratio, crop = letterbox(image_bgr, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x.requires_grad_(True)
    return x, lb, ratio, crop


def load_gt_box(image_path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not label_path.exists():
        return None
    h, w = image_shape
    rows = [line.strip().split() for line in label_path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    _, xc, yc, bw, bh = map(float, rows[0])
    return np.array([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], dtype=np.float32)


def original_to_letterbox_box(box: np.ndarray, ratio: float, crop: tuple[int, int, int, int]) -> np.ndarray:
    left, top, _, _ = crop
    out = box.copy().astype(np.float32)
    out[[0, 2]] = out[[0, 2]] * ratio + left
    out[[1, 3]] = out[[1, 3]] * ratio + top
    return out


def letterbox_to_original_box(box: np.ndarray, ratio: float, crop: tuple[int, int, int, int], shape: tuple[int, int]):
    left, top, _, _ = crop
    out = box.copy().astype(np.float32)
    out[[0, 2]] = (out[[0, 2]] - left) / ratio
    out[[1, 3]] = (out[[1, 3]] - top) / ratio
    out[[0, 2]] = np.clip(out[[0, 2]], 0, shape[1] - 1)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, shape[0] - 1)
    return out


def xywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou_np(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    area2 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return inter / (area1 + area2 - inter + 1e-7)


def normalize(x: np.ndarray, high: float = 99.5) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, [0, high])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def deletterbox_heatmap(heat: np.ndarray, crop: tuple[int, int, int, int], original_hw: tuple[int, int]) -> np.ndarray:
    left, top, resized_w, resized_h = crop
    heat_640 = cv2.resize(heat, (640, 640), interpolation=cv2.INTER_LINEAR)
    heat_crop = heat_640[top : top + resized_h, left : left + resized_w]
    return cv2.resize(heat_crop, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_LINEAR)


def overlay_sparse(image: np.ndarray, heat: np.ndarray, threshold: float = 0.45, alpha: float = 0.58):
    heat = normalize(heat)
    color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    mask = (heat >= threshold).astype(np.float32)[..., None]
    blended = cv2.addWeighted(image, 1 - alpha, color, alpha, 0)
    return (image * (1 - mask) + blended * mask).astype(np.uint8)


def draw_boxes(image: np.ndarray, gt: np.ndarray | None, preds, names: list[str]):
    out = image.copy()
    if gt is not None:
        cv2.rectangle(out, tuple(gt[:2].astype(int)), tuple(gt[2:].astype(int)), (0, 255, 0), 3)
        cv2.putText(out, "GT bolt", tuple(gt[:2].astype(int) + np.array([0, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for box, conf, name in zip(preds.boxes.xyxy.cpu().numpy(), preds.boxes.conf.cpu().numpy(), names):
        color = (0, 0, 255) if name == "FP" else (255, 128, 0)
        cv2.rectangle(out, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), color, 3)
        cv2.putText(out, f"{name} {conf:.2f}", tuple(box[:2].astype(int) + np.array([0, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def save_panel(path: Path, images: list[np.ndarray], titles: list[str], cols: int = 2):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.1 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_crop(path: Path, image: np.ndarray, box: np.ndarray, pad: int = 180):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    cv2.imwrite(str(path), image[y1:y2, x1:x2])


def register_layer_hooks(model, layers: list[int]):
    activations: dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        layer = model.model[layer_idx]

        def forward_hook(_module, _inp, out, idx=layer_idx):
            activations[idx] = out
            out.retain_grad()

        handles.append(layer.register_forward_hook(forward_hook))
    return activations, handles


def pick_targets(pred: torch.Tensor, gt_lb: np.ndarray, ratio: float, crop: tuple[int, int, int, int], original_hw: tuple[int, int], conf: float):
    scores = pred[0, 4, :].detach()
    boxes_lb = xywh_to_xyxy(pred[0, :4, :].detach().T).cpu().numpy()
    ious = box_iou_np(boxes_lb, gt_lb)
    valid = scores.cpu().numpy() >= conf

    tp_candidates = np.where((ious > 0.2) & valid)[0]
    if len(tp_candidates) == 0:
        tp_candidates = np.argsort(-(ious * scores.cpu().numpy()))[:1]
    tp_idx = int(tp_candidates[np.argmax(ious[tp_candidates] * scores.cpu().numpy()[tp_candidates])])

    fp_candidates = np.where((ious < 0.05) & valid)[0]
    fp = None
    if len(fp_candidates):
        fp_idx = int(fp_candidates[np.argmax(scores.cpu().numpy()[fp_candidates])])
        fp = Target(
            "false_positive",
            fp_idx,
            float(scores[fp_idx]),
            letterbox_to_original_box(boxes_lb[fp_idx], ratio, crop, original_hw),
            float(ious[fp_idx]),
        )

    tp = Target(
        "true_bolt",
        tp_idx,
        float(scores[tp_idx]),
        letterbox_to_original_box(boxes_lb[tp_idx], ratio, crop, original_hw),
        float(ious[tp_idx]),
    )
    return tp, fp


def gradcam_for_target(weight: Path, image: np.ndarray, image_path: Path, layers: list[int], imgsz: int, device: torch.device, conf: float):
    yolo = YOLO(str(weight))
    model = yolo.model.to(device).eval()
    x, _, ratio, crop = to_tensor(image, imgsz, device)
    gt = load_gt_box(image_path, image.shape[:2])
    gt_lb = original_to_letterbox_box(gt, ratio, crop) if gt is not None else None
    if gt_lb is None:
        raise FileNotFoundError("No label file was found for GT-guided target selection.")

    with torch.no_grad():
        pred_for_selection = model(x.detach())[0]
    tp, fp = pick_targets(pred_for_selection, gt_lb, ratio, crop, image.shape[:2], conf)

    target_cams: dict[str, dict[int, np.ndarray]] = {}
    target_info = {"true_bolt": tp}
    if fp is not None:
        target_info["false_positive"] = fp

    for target_name, target in target_info.items():
        x_target, _, _, _ = to_tensor(image, imgsz, device)
        activations, handles = register_layer_hooks(model, layers)
        model.zero_grad(set_to_none=True)
        pred = model(x_target)[0]
        score = pred[0, 4, target.raw_index]
        score.backward()
        target_cams[target_name] = {}
        for idx, act in activations.items():
            grad = act.grad
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1, keepdim=True))[0, 0]
            cam = normalize(cam.detach().cpu().numpy())
            target_cams[target_name][idx] = deletterbox_heatmap(cam, crop, image.shape[:2])
        for h in handles:
            h.remove()

    result = yolo.predict(source=image, imgsz=imgsz, conf=conf, verbose=False, device=str(device))[0]
    return target_cams, target_info, result


def attention_mass(heat: np.ndarray, box: np.ndarray) -> tuple[float, float]:
    mask = np.zeros_like(heat, dtype=bool)
    x1, y1, x2, y2 = box.astype(int)
    mask[max(0, y1) : min(mask.shape[0], y2), max(0, x1) : min(mask.shape[1], x2)] = True
    total = float(heat.sum() + 1e-7)
    inside = float(heat[mask].sum() / total)
    peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
    peak_inside = bool(mask[peak])
    return inside, float(peak_inside)


def scalar_value(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def save_leca_scalars(model_path: Path, out_csv: Path):
    model = YOLO(str(model_path)).model
    rows = []
    for i, layer in enumerate(model.model):
        if hasattr(layer, "eca") and all(hasattr(layer.eca, k) for k in ("alpha", "beta", "gamma")):
            rows.append(
                {
                    "layer": i,
                    "alpha_weak_recovery": scalar_value(layer.eca.alpha),
                    "beta_variance_suppression": scalar_value(layer.eca.beta),
                    "gamma_brightness_correction": scalar_value(layer.eca.gamma),
                }
            )
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "alpha_weak_recovery", "beta_variance_suppression", "gamma_brightness_correction"])
        writer.writeheader()
        writer.writerows(rows)


def classify_predictions(result, gt: np.ndarray | None):
    if gt is None or len(result.boxes) == 0:
        return ["Pred"] * len(result.boxes)
    boxes = result.boxes.xyxy.cpu().numpy()
    ious = box_iou_np(boxes, gt)
    names = []
    used_tp = False
    for iou in ious:
        if iou > 0.2 and not used_tp:
            names.append("TP")
            used_tp = True
        else:
            names.append("FP")
    return names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--leca", type=Path, default=DEFAULT_LECA)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layers", type=str, default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlay-threshold", type=float, default=0.62)
    return parser.parse_args()


def main():
    patch_c2f_forward_for_old_baseline()
    args = parse_args()
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    device = torch.device(args.device)
    tag = f"{args.image.stem}_conf{int(args.conf * 100):03d}"
    out_dir = args.out_root / tag
    base_dir = out_dir / "baseline"
    leca_dir = out_dir / "LECA7"
    compare_dir = out_dir / "compare"
    for d in (base_dir, leca_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    gt = load_gt_box(args.image, image.shape[:2])

    base_cams, base_targets, base_result = gradcam_for_target(args.baseline, image, args.image, layers, args.imgsz, device, args.conf)
    leca_cams, leca_targets, leca_result = gradcam_for_target(args.leca, image, args.image, layers, args.imgsz, device, args.conf)

    base_pred = draw_boxes(image, gt, base_result, classify_predictions(base_result, gt))
    leca_pred = draw_boxes(image, gt, leca_result, classify_predictions(leca_result, gt))
    cv2.imwrite(str(base_dir / "predictions_conf050.jpg"), base_pred)
    cv2.imwrite(str(leca_dir / "predictions_conf050.jpg"), leca_pred)
    save_panel(compare_dir / "01_predictions_baseline_vs_LECA7_conf050.jpg", [base_pred, leca_pred], ["Baseline: TP + FP", "LECA7: TP only"], cols=2)

    metric_rows = []
    comparison_images, comparison_titles = [], []
    for layer in layers:
        for model_name, model_dir, cams, targets in (
            ("baseline", base_dir, base_cams, base_targets),
            ("LECA7", leca_dir, leca_cams, leca_targets),
        ):
            heat = cams["true_bolt"][layer]
            overlay = overlay_sparse(image, heat, threshold=args.overlay_threshold)
            boxed = draw_boxes(overlay, gt, base_result if model_name == "baseline" else leca_result, classify_predictions(base_result if model_name == "baseline" else leca_result, gt))
            cv2.imwrite(str(model_dir / f"gradcam_true_bolt_layer{layer:02d}.jpg"), boxed)
            save_crop(model_dir / f"crop_true_bolt_layer{layer:02d}.jpg", boxed, targets["true_bolt"].box_xyxy)
            inside, peak_inside = attention_mass(heat, gt)
            metric_rows.append({"model": model_name, "target": "true_bolt", "layer": layer, "score": targets["true_bolt"].score, "iou_gt": targets["true_bolt"].iou_gt, "attention_mass_in_gt": inside, "peak_inside_gt": peak_inside})
            comparison_images.append(boxed)
            comparison_titles.append(f"{model_name} layer {layer} true bolt")

        if "false_positive" in base_cams:
            heat = base_cams["false_positive"][layer]
            overlay = overlay_sparse(image, heat, threshold=args.overlay_threshold)
            boxed = draw_boxes(overlay, gt, base_result, classify_predictions(base_result, gt))
            cv2.imwrite(str(base_dir / f"gradcam_false_positive_layer{layer:02d}.jpg"), boxed)
            save_crop(base_dir / f"crop_false_positive_layer{layer:02d}.jpg", boxed, base_targets["false_positive"].box_xyxy)
            inside, peak_inside = attention_mass(heat, gt)
            metric_rows.append({"model": "baseline", "target": "false_positive", "layer": layer, "score": base_targets["false_positive"].score, "iou_gt": base_targets["false_positive"].iou_gt, "attention_mass_in_gt": inside, "peak_inside_gt": peak_inside})

    save_panel(compare_dir / "02_gradcam_true_bolt_layers_baseline_vs_LECA7.jpg", comparison_images, comparison_titles, cols=2)

    if "false_positive" in base_cams:
        fp_images, fp_titles = [], []
        for layer in layers:
            heat = base_cams["false_positive"][layer]
            fp_images.append(draw_boxes(overlay_sparse(image, heat, threshold=args.overlay_threshold), gt, base_result, classify_predictions(base_result, gt)))
            fp_titles.append(f"baseline FP source layer {layer}")
        save_panel(compare_dir / "03_baseline_false_positive_gradcam_layers.jpg", fp_images, fp_titles, cols=2)

    if 16 in layers:
        save_panel(
            compare_dir / "00_key_layer16_summary.jpg",
            [
                base_pred,
                leca_pred,
                cv2.imread(str(base_dir / "gradcam_true_bolt_layer16.jpg")),
                cv2.imread(str(leca_dir / "gradcam_true_bolt_layer16.jpg")),
            ],
            [
                "Baseline detection: one TP + one FP",
                "LECA7 detection: FP suppressed",
                "Baseline Grad-CAM layer 16",
                "LECA7 Grad-CAM layer 16",
            ],
            cols=2,
        )

    with (compare_dir / "gradcam_attention_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "target", "layer", "score", "iou_gt", "attention_mass_in_gt", "peak_inside_gt"])
        writer.writeheader()
        writer.writerows(metric_rows)

    save_leca_scalars(args.leca, leca_dir / "learned_LECA_scalars.csv")
    print(f"Saved organized Grad-CAM results to: {out_dir}")
    print(f"Main comparison: {compare_dir / '02_gradcam_true_bolt_layers_baseline_vs_LECA7.jpg'}")
    print(f"Baseline FP source: {compare_dir / '03_baseline_false_positive_gradcam_layers.jpg'}")
    print(f"Metrics: {compare_dir / 'gradcam_attention_metrics.csv'}")


if __name__ == "__main__":
    main()
