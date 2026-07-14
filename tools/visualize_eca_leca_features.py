#!/usr/bin/env python3
"""Create local-only ECA/LECA feature-map comparison panels for hard-test images."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO


def attention_values(module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    x = x.float()
    w_eca = module.sigmoid(module.conv(module.avg_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
    if module.__class__.__name__ != "LECA":
        return {"w_eca": w_eca, "w_final": w_eca}
    mu = x.mean((2, 3), keepdim=True)
    variance = (x * x).mean((2, 3), keepdim=True) - mu * mu
    corr = torch.sigmoid(-x.mean(1, keepdim=True)).mean((2, 3), keepdim=True)
    w_sup = 1 / (1 + module.beta.float() * torch.nn.functional.softplus(variance))
    w_rec = 1 + module.alpha.float() * torch.sigmoid(-mu)
    w_bri = 1 + module.gamma.float() * corr
    return {"mu": mu, "variance": variance, "corr": corr, "w_eca": w_eca, "w_sup": w_sup, "w_rec": w_rec, "w_bri": w_bri, "w_final": w_eca * w_sup * w_rec * w_bri}


def capture(yolo: YOLO, image: Path, layer: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured = {}
    module = dict(yolo.model.named_modules())[layer]
    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()
    handle = module.register_forward_hook(hook)
    yolo.predict(str(image), imgsz=640, device=0, verbose=False, save=False)
    handle.remove()
    return captured["x"], attention_values(module, captured["x"])


def overlay(image: np.ndarray, feature: torch.Tensor) -> np.ndarray:
    fmap = feature.detach().float().cpu().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    heat = cv2.applyColorMap(np.uint8(fmap * 255), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(image, 0.48, heat, 0.52, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Hard Test 的 ECA/LECA 特征图对比，仅保存本地。")
    parser.add_argument("--image-list", type=Path, help="每行一个 Hard Test 图像文件名；省略时取前 5 张。")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image_dir = root / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    if args.image_list:
        images = [image_dir / name.strip() for name in args.image_list.read_text().splitlines() if name.strip()]
    else:
        images = sorted(image_dir.glob("*"))[:5]
    output = root / "artifacts/visualizations/hard_feature_maps"
    output.mkdir(parents=True, exist_ok=True)
    eca = YOLO(str(root / "runs_repro/mechanism_smoke/eca_seed42/weights/best.pt"))
    leca = YOLO(str(root / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt"))
    layer = "model.16.eca"  # P3-resolution matched attention site
    rows = []
    for image_path in images:
        raw = cv2.imread(str(image_path))
        eca_x, eca_v = capture(eca, image_path, layer)
        leca_x, leca_v = capture(leca, image_path, layer)
        difference = (leca_v["w_final"][0, :, 0, 0] - eca_v["w_final"][0, :, 0, 0]).abs()
        for channel in torch.topk(difference, k=3).indices.tolist():
            eca_panel = overlay(raw, eca_x[0, channel])
            leca_panel = overlay(raw, leca_x[0, channel])
            panel = np.concatenate([raw, eca_panel, leca_panel], axis=1)
            values = {key: float(value[0, channel, 0, 0]) if value.shape[1] > 1 else float(value[0, 0, 0, 0]) for key, value in leca_v.items()}
            text = f"ch={channel} | ECA={float(eca_v['w_final'][0, channel, 0, 0]):.4f} | LECA={values['w_final']:.4f} | sup={values['w_sup']:.4f} rec={values['w_rec']:.4f} bri={values['w_bri']:.4f}"
            cv2.putText(panel, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            name = f"{image_path.stem}_layer16_ch{channel}.jpg"
            cv2.imwrite(str(output / name), panel)
            rows.append({"image_id": image_path.name, "layer": layer, "channel": channel, "file": name, **values, "eca_weight": float(eca_v["w_final"][0, channel, 0, 0])})
    with (output / "feature_map_index.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
