# LECA paper reproduction and audit

This repository preserves the published LECA implementation and adds a conservative audit and reproducibility protocol for bolt detection. The retained implementation is under `ultralytics-main/`.

- Conda environment: `yolo` (see `environment.yml`).
- Training entry points: historical scripts are in `ultralytics-main/train_*.py`; the controlled protocol and commands are in `docs/`.
- Local data paths must be configured in a dataset YAML. Datasets, annotations, images, weights, runs, and visualizations are intentionally not versioned.
- `paper-original` is the immutable tag for the published-code snapshot. Any future LECA-v2 work must use `exp/leca-v2`, never overwrite the paper reproduction on `main`.
- Local outputs belong in ignored `runs_repro/` and `artifacts/` directories. Versioned reports contain only compact, non-image summaries.

See `docs/CODE_AUDIT.md` before starting any long training.
