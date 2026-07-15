#!/usr/bin/env python3
"""汇总收敛 LECA 在全部 Hard Test 上的逐层动态因子；只保存聚合统计。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

from ultralytics import YOLO


LAYERS = ["model.2.eca", "model.4.eca", "model.6.eca", "model.8.eca",
          "model.13.eca", "model.16.eca", "model.19.eca", "model.22.eca"]


def describe(parts: list[np.ndarray]) -> dict[str, float | int]:
    values = np.concatenate(parts).astype(np.float64, copy=False)
    values = values[np.isfinite(values)]
    quantiles = np.quantile(values, (.05, .50, .95))
    return {
        "count": len(values), "mean": float(values.mean()), "std": float(values.std()),
        "p05": float(quantiles[0]), "p50": float(quantiles[1]), "p95": float(quantiles[2]),
        "min": float(values.min()), "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    checkpoint = ROOT / "runs_repro/mechanism_smoke/leca_seed42/weights/best.pt"
    image_dir = ROOT / "ultralytics-main/dataset/hardData/YOLODataset/images/test"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    model = YOLO(str(checkpoint))
    # 先完成 predictor 初始化和模型 warm-up，避免 dummy forward 污染 146 张真实样本统计。
    model.predict(str(images[0]), imgsz=640, batch=1, device=args.device, conf=.25, save=False, verbose=False)
    modules = dict(model.model.named_modules())
    values = {layer: {name: [] for name in ("corr", "w_eca", "w_sup", "w_rec", "w_bri", "w_stat", "w_final")} for layer in LAYERS}
    handles = []

    def make_hook(layer: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            with torch.no_grad():
                x = inputs[0].detach().float()
                mean = x.mean(dim=(2, 3), keepdim=True)
                variance = (x * x).mean(dim=(2, 3), keepdim=True) - mean * mean
                corr = torch.sigmoid(-x.mean(dim=1, keepdim=True)).mean(dim=(2, 3), keepdim=True)
                pooled = module.avg_pool(x).squeeze(-1).transpose(-1, -2)
                w_eca = module.sigmoid(module.conv(pooled).transpose(-1, -2).unsqueeze(-1))
                w_sup = 1 / (1 + module.beta.detach().float() * torch.nn.functional.softplus(variance))
                w_rec = 1 + module.alpha.detach().float() * torch.sigmoid(-mean)
                w_bri = 1 + module.gamma.detach().float() * corr
                computed = {
                    "corr": corr, "w_eca": w_eca, "w_sup": w_sup, "w_rec": w_rec,
                    "w_bri": w_bri, "w_stat": w_sup * w_rec * w_bri,
                    "w_final": w_eca * w_sup * w_rec * w_bri,
                }
                for name, tensor in computed.items():
                    values[layer][name].append(tensor.float().cpu().numpy().reshape(-1))
        return hook

    for layer in LAYERS:
        handles.append(modules[layer].register_forward_hook(make_hook(layer)))
    model.predict([str(path) for path in images], imgsz=640, batch=args.batch, device=args.device,
                  conf=.25, save=False, verbose=False)
    for handle in handles:
        handle.remove()

    rows = []
    for layer in LAYERS:
        module = modules[layer]
        for factor, parts in values[layer].items():
            row = {
                "layer": layer,
                "stage": "Backbone" if int(layer.split(".")[1]) <= 8 else "Fusion/Detect",
                "factor": factor,
                "alpha": float(module.alpha.detach().cpu()),
                "beta": float(module.beta.detach().cpu()),
                "gamma": float(module.gamma.detach().cpu()),
                "n_images": len(images),
                **describe(parts),
                "status": "Hard_Test_aggregate_from_seed42_best_checkpoint",
            }
            rows.append(row)
    output = ROOT / "reports/leca_layer_factor_summary_seed42.csv"
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"写入 {len(rows)} 行聚合统计: {output}")


if __name__ == "__main__":
    main()
