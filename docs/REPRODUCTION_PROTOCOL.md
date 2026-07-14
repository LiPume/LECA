# Reproduction protocol

## Non-negotiable guardrails

- `paper-original` is immutable. `main` preserves paper reproduction/audit only; future formula work starts with `git switch -c exp/leca-v2`.
- Never change a dataset split, filter test samples, overwrite weights, delete original outputs, force-push, or upload data/images/weights/runs/visualizations.
- Record each command in `docs/COMMAND_LOG.md`; persist compact results only in `reports/`.
- Use **Confirmed**, **Observed**, and **Hypothesis** labels in every report.

## Current gate: STOP

Do not run smoke or full training yet. The current build cannot identify an attention-free Baseline or a paper-matching ECA/LECA architecture, and its canonical train/val split leaks five exact images. A new split is a new benchmark—not a retroactive paper result—unless the original published split can be recovered separately.

## Once the gate is resolved

1. Save recovered source/YAML hashes and the train/val/test manifest; create explicit matched Baseline, ECA, LECA configs without changing `paper-original`.
2. Run `conda run -n yolo env PYTHONPATH=ultralytics-main python tools/audit_model_build.py` and local dataset auditor. Confirm all architectures, Params/FLOPs tool/input, insertion positions, and split hashes.
3. Use an explicit config with seed=42, deterministic=true, explicit optimizer/LR/weight decay, same pretrained checkpoint hash, 640 input, batch size, AMP setting, workers, cache, augmentation and patience for every model.
4. Run one forward/backward batch and finite checks, then exactly one epoch smoke per model. Save to `runs_repro/<model>/42/smoke/`.
5. Check: loss finite, alpha/beta/gamma gradients nonzero when branch enabled, no finite-count alert, checkpoint created, validation operates, and test has not been used for selection.
6. Only then train seed=42; assess paper trend; then the multi-seed and complete-ablation schedule in `EXPERIMENT_TODO.md`.

## Measurement protocol

- Params: `sum(p.numel() for p in model.parameters())` on the exact instantiated graph.
- FLOPs: use one fixed tool/version, FP32 input `[1,3,640,640]`, report tool and units.
- Latency: same GPU, batch=1, resolution=640, precision, warm-up count, synchronized timing and number of timed trials. Report whether preprocessing/NMS/transfer are excluded or included; do not compare mixed protocols.
- Testing: select `best.pt` from validation only; test exactly once per fixed trained checkpoint. Hard-test use is evaluation-only.

## Hook schema (default off)

For each LECA layer and epoch, aggregate `alpha,beta,gamma,mu,var,corr,w_eca,w_sup,w_rec,w_bri,w_final`: mean/std/p05/p25/p50/p75/p95/min/max and NaN/Inf counts. Raw data stays ignored under `artifacts/statistics/`; append only aggregate rows to `reports/scalar_evolution.csv` and `reports/statistics_summary.csv`.
