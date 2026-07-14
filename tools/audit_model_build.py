#!/usr/bin/env python3
"""Read-only architecture and LECA numerical audit for the local source tree."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import LECA


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ultralytics-main"
CONFIGS = {
    "baseline": SOURCE / "ultralytics/cfg/models/11/yolo11.yaml",
    "ema": SOURCE / "ultralytics/cfg/models/11/yolo11EMA.yaml",
    "se": SOURCE / "ultralytics/cfg/models/11/yolo11SE.yaml",
    "cbam": SOURCE / "ultralytics/cfg/models/11/yolo11CBAM.yaml",
    "nam": SOURCE / "ultralytics/cfg/models/11/yolo11NAM.yaml",
}


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def audit_leca() -> dict:
    result: dict[str, object] = {"direct_modules": []}
    for channels in (64, 128, 256, 512, 1024):
        module = LECA(channels).train()
        x = torch.randn(2, channels, 8, 8, requires_grad=True)
        y = module(x)
        y.square().mean().backward()
        result["direct_modules"].append(
            {
                "channels": channels,
                "kernel_size": module.conv.kernel_size[0],
                "parameter_count": count_parameters(module),
                "scalar_parameter_shapes": {
                    "alpha": list(module.alpha.shape),
                    "beta": list(module.beta.shape),
                    "gamma": list(module.gamma.shape),
                },
                "finite_output": bool(torch.isfinite(y).all()),
                "finite_input_gradient": bool(torch.isfinite(x.grad).all()),
                "scalar_gradients": {
                    name: float(getattr(module, name).grad.detach())
                    for name in ("alpha", "beta", "gamma")
                },
            }
        )
    return result


def audit_configs() -> dict:
    result: dict[str, object] = {"models": {}}
    for name, config in CONFIGS.items():
        model = YOLO(str(config)).model
        attention = [
            {"name": module_name, "class": module.__class__.__name__, "parameters": count_parameters(module)}
            for module_name, module in model.named_modules()
            if module.__class__.__name__ in {"LECA", "ECA", "EMA", "SE", "CBAM", "NAM"}
        ]
        result["models"][name] = {
            "config": str(config.relative_to(ROOT)),
            "parameter_count": count_parameters(model),
            "attention_modules": attention,
            "leca_count": sum(item["class"] == "LECA" for item in attention),
        }
    result["published_yolo11_leca_yaml_present"] = bool(
        list((SOURCE / "ultralytics/cfg/models/11").glob("*LECA*.yaml"))
    )
    return result


def main() -> None:
    result = {"leca": audit_leca(), "model_configs": audit_configs()}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
