#!/usr/bin/env python3
"""独立重训练 LECA 六个缺失分支组合；ECA 与 Full 使用已完成的同协议 seed=42 运行。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


MODULES = {
    "var": "C3k2LECAVar",
    "rec": "C3k2LECARec",
    "bri": "C3k2LECABri",
    "var_rec": "C3k2LECAVarRec",
    "var_bri": "C3k2LECAVarBri",
    "rec_bri": "C3k2LECARecBri",
}
EXPECTED = {
    "var": {"var"}, "rec": {"rec"}, "bri": {"bri"},
    "var_rec": {"var", "rec"}, "var_bri": {"var", "bri"}, "rec_bri": {"rec", "bri"},
}


def generated_config(root: Path, combination: str) -> Path:
    source = root / "configs/yolo11_audit_leca.yaml"
    output = root / "runs_repro/factorial_ablation/generated_configs" / f"{combination}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text().replace("C3k2LECA", MODULES[combination])
    output.write_text("# 自动生成的受控重训练消融配置；生成逻辑受 Git 跟踪。\n" + text)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combination", choices=MODULES, required=True)
    parser.add_argument("--mode", choices=("build", "smoke", "full"), default="smoke")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(generated_config(root, args.combination)))
    modules = [module for module in model.model.modules() if module.__class__.__name__ == "FactorialLECA"]
    if len(modules) != 8 or any(set(module.branches) != EXPECTED[args.combination] for module in modules):
        raise RuntimeError(f"消融模块构建错误: count={len(modules)}, combination={args.combination}")
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    print(f"构建通过: {args.combination}, FactorialLECA=8, parameters={parameters}")
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
        project=str(root / "runs_repro/factorial_ablation"),
        name=f"{args.combination}_seed{args.seed}_{args.mode}",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
