#!/usr/bin/env python3
"""受控训练 LECA 插入位置消融；生成的 YAML 与训练输出均位于 ignored runs_repro。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))

from ultralytics import YOLO


PLACEMENTS = {
    "none0": set(),
    "backbone4": {2, 4, 6, 8},
    "fusion4": {13, 16, 19, 22},
    "scales3": {16, 19, 22},
    "full8": {2, 4, 6, 8, 13, 16, 19, 22},
}


def generated_config(root: Path, placement: str) -> Path:
    source = root / "configs/yolo11_audit_leca.yaml"
    config = yaml.safe_load(source.read_text())
    selected = PLACEMENTS[placement]
    for index, layer in enumerate(config["backbone"] + config["head"]):
        if layer[2] == "C3k2LECA":
            layer[2] = "C3k2LECA" if index in selected else "C3k2Baseline"
    output = root / "runs_repro/placement_ablation/generated_configs" / f"{placement}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 自动生成的位置消融配置；不要手工修改。\n"
        f"# placement={placement}; LECA_indices={sorted(selected)}\n"
    )
    output.write_text(header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", choices=PLACEMENTS, required=True)
    parser.add_argument("--mode", choices=("build", "smoke", "full"), default="smoke")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = ROOT
    config = generated_config(root, args.placement)
    model = YOLO(str(config))
    leca_modules = [module for module in model.model.modules() if module.__class__.__name__ == "LECA"]
    expected = len(PLACEMENTS[args.placement])
    if len(leca_modules) != expected:
        raise RuntimeError(f"位置消融构建错误: placement={args.placement}, expected={expected}, actual={len(leca_modules)}")
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    print(
        f"构建通过: placement={args.placement}, indices={sorted(PLACEMENTS[args.placement])}, "
        f"LECA={len(leca_modules)}, parameters={parameters}"
    )
    if args.mode == "build":
        return

    epochs = 1 if args.mode == "smoke" else 200
    model.train(
        data=str(root / "ultralytics-main/ultralytics/cfg/models/datasets/screw.yaml"),
        pretrained=str(root / "ultralytics-main/yolo11n.pt"),
        epochs=epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=20,
        amp=True,
        cache=False,
        plots=True,
        val=True,
        project=str(root / "runs_repro/placement_ablation"),
        name=f"{args.placement}_seed{args.seed}_{args.mode}",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
