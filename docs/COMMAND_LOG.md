# Command log

All commands below are run from `/home/lzx/car_bolt_detection` unless stated otherwise. Paths to datasets, runs, images, and weights remain local and are ignored by Git.

## 2026-07-14: phase 0 inspection

```bash
pwd
conda run -n yolo python -V
conda run -n yolo python -c "import torch; ..."
nvidia-smi || true
find . -type f -size +50M -not -path './.git/*'
grep -RInE '(BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|github_pat_|ghp_|AKIA|api[_-]?key|secret[_-]?key|password[ ]*=)' . ... || true
conda env export -n yolo --no-builds | sed '/^prefix:/d' > environment.yml
```

## Controlled commands (not yet authorized to run)

Long training is blocked until the published YOLO11 LECA YAML and its exact attention placement are recovered, and the confirmed train/validation leakage is resolved in a new, documented split. The future smoke and reproduction commands are specified in `docs/REPRODUCTION_PROTOCOL.md`.

## 2026-07-14: read-only implementation and data audit

```bash
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/audit_dataset.py \
  ultralytics-main/ultralytics/cfg/models/datasets/screw.yaml \
  --output artifacts/data_audit/trainDataV3_summary.json
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/audit_dataset.py \
  ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml \
  --output artifacts/data_audit/hard_test_summary.json
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/audit_model_build.py
```

These commands generated local-only aggregate metadata under ignored `artifacts/data_audit/`. The first script found five exact image SHA256 matches spanning train and validation. The model auditor found eight implicit LECA modules in the current `yolo11.yaml`; it also found no YOLO11 LECA YAML.

## 2026-07-14: Git safety actions

```bash
git init
git branch -M main
git remote add origin git@github.com:LiPume/LECA.git
git add .
git commit -m "first commit"
git tag -a paper-original -m "Published LECA implementation before audit"
git push -u origin main
git push origin paper-original
```

The remote was empty and the non-force push succeeded. Media, datasets, weights, training outputs, artifacts, and visualizations were verified ignored before staging.

## 2026-07-14: controlled mechanism-audit branch

```bash
git switch -c exp/mechanism-audit
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_mechanism_smoke.py \
  --model baseline --device 0 --batch 16 --workers 4
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_mechanism_smoke.py \
  --model eca --device 0 --batch 16 --workers 4
conda run -n yolo env PYTHONPATH=ultralytics-main python tools/train_mechanism_smoke.py \
  --model leca --name leca_stats_retry --device 0 --batch 16 --workers 4
```

This branch adds explicit topology-matched Identity/ECA/LECA C3k2 wrappers and three audit YAMLs. It is not a paper reproduction. The first LECA statistics attempt recorded scalars but no feature aggregates because the hook was registered before Ultralytics copied its training model; the corrected hook is registered at `on_pretrain_routine_end`, and the successful retry is retained separately.
