#!/usr/bin/env python3
"""One-epoch controlled mechanism smoke test with non-invasive LECA summaries.

This is an audit utility, not a paper-reproduction launcher. It never saves image
pixels or feature tensors outside the normal ignored Ultralytics run directory.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch

from ultralytics import YOLO


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


class Aggregate:
    """Streaming moments plus a bounded local reservoir for approximate quantiles."""

    def __init__(self, limit: int = 4096):
        self.limit, self.count, self.total, self.total_sq = limit, 0, 0.0, 0.0
        self.minimum, self.maximum, self.nan_count, self.inf_count = float("inf"), float("-inf"), 0, 0
        self.samples: list[torch.Tensor] = []

    def add(self, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1)
        self.nan_count += int(torch.isnan(flat).sum())
        self.inf_count += int(torch.isinf(flat).sum())
        finite = flat[torch.isfinite(flat)]
        if not finite.numel():
            return
        self.count += finite.numel()
        self.total += float(finite.sum())
        self.total_sq += float((finite * finite).sum())
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        remaining = self.limit - sum(sample.numel() for sample in self.samples)
        if remaining > 0:
            stride = max(1, finite.numel() // remaining)
            self.samples.append(finite[::stride][:remaining].cpu())

    def row(self, run_id: str, epoch: int, layer: str, statistic: str) -> dict[str, object]:
        sample = torch.cat(self.samples) if self.samples else torch.empty(0)
        mean = self.total / self.count if self.count else float("nan")
        variance = max(self.total_sq / self.count - mean * mean, 0.0) if self.count else float("nan")
        q = torch.quantile(sample, torch.tensor(QUANTILES)).tolist() if sample.numel() else [float("nan")] * 5
        return {
            "run_id": run_id,
            "epoch": epoch,
            "layer": layer,
            "statistic": statistic,
            "mean": mean,
            "std": variance**0.5,
            "p05": q[0],
            "p25": q[1],
            "p50": q[2],
            "p75": q[3],
            "p95": q[4],
            "min": self.minimum if self.count else float("nan"),
            "max": self.maximum if self.count else float("nan"),
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "status": "smoke_only_not_for_paper_metrics",
        }


class LECACollector:
    def __init__(self):
        self.active, self.epoch, self.records, self.handles = False, 0, defaultdict(Aggregate), []

    @staticmethod
    def _is_leca(module: torch.nn.Module) -> bool:
        return module.__class__.__name__ == "LECA" and all(hasattr(module, key) for key in ("alpha", "beta", "gamma"))

    def _hook(self, layer: str):
        def record(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if not self.active:
                return
            with torch.no_grad():
                x = inputs[0].detach().float()
                mean = x.mean(dim=(2, 3), keepdim=True)
                variance = (x * x).mean(dim=(2, 3), keepdim=True) - mean * mean
                noise = torch.nn.functional.softplus(variance)
                eca = module.sigmoid(module.conv(module.avg_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
                sup = 1.0 / (1.0 + module.beta.detach().float() * noise)
                rec = 1.0 + module.alpha.detach().float() * torch.sigmoid(-mean)
                corr = torch.sigmoid(-x.mean(dim=1, keepdim=True)).mean(dim=(2, 3), keepdim=True)
                bri = 1.0 + module.gamma.detach().float() * corr
                values = {"mu": mean, "variance": variance, "corr": corr, "w_eca": eca, "w_sup": sup, "w_rec": rec, "w_bri": bri, "w_final": eca * sup * rec * bri}
                for name, value in values.items():
                    self.records[(layer, name)].add(value)
        return record

    def attach(self, trainer) -> None:
        """Attach after Ultralytics has constructed/trainer-copied the actual model."""
        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        self.handles = [module.register_forward_hook(self._hook(name)) for name, module in model.named_modules() if self._is_leca(module)]
        if not self.handles and "leca" in str(trainer.args.model).lower():
            raise RuntimeError("LECA statistics requested but no trainable LECA module was found in trainer.model")

    def start(self, trainer) -> None:
        self.active, self.epoch, self.records = True, trainer.epoch + 1, defaultdict(Aggregate)

    def flush(self, trainer) -> None:
        self.active = False
        save_dir = Path(trainer.save_dir)
        rows = [aggregate.row(save_dir.name, self.epoch, layer, statistic) for (layer, statistic), aggregate in self.records.items()]
        fields = list(rows[0]) if rows else []
        if rows:
            path = save_dir / "leca_statistics.csv"
            write_header = not path.exists()
            with path.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
        scalar_path = save_dir / "leca_scalar_evolution.csv"
        write_header = not scalar_path.exists()
        with scalar_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["run_id", "epoch", "layer", "alpha", "beta", "gamma", "status"])
            if write_header:
                writer.writeheader()
            for name, module in trainer.model.named_modules():
                if self._is_leca(module):
                    writer.writerow({"run_id": save_dir.name, "epoch": self.epoch, "layer": name, "alpha": float(module.alpha.detach()), "beta": float(module.beta.detach()), "gamma": float(module.gamma.detach()), "status": "smoke_only_not_for_paper_metrics"})

    def close(self, _trainer) -> None:
        for handle in self.handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "eca", "leca"), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", help="Ignored local run directory name; defaults to the model name.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / "configs" / f"yolo11_audit_{args.model}.yaml"))
    collector = LECACollector()
    model.add_callback("on_pretrain_routine_end", collector.attach)
    model.add_callback("on_train_epoch_start", collector.start)
    model.add_callback("on_train_epoch_end", collector.flush)
    model.add_callback("on_train_end", collector.close)
    model.train(
        data=str(root / "ultralytics-main/ultralytics/cfg/models/datasets/screw.yaml"),
        pretrained=str(root / "ultralytics-main/yolo11n.pt"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=42,
        deterministic=True,
        patience=args.patience,
        amp=True,
        cache=False,
        plots=True,
        val=True,
        project=str(root / "runs_repro/mechanism_smoke"),
        name=args.name or args.model,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
