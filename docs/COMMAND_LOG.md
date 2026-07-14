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
