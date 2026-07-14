# LECA experiment TODO

Status legend: `[x]` completed audit item; `[!]` blocking item; `[ ]` pending. No item permits changing the published implementation on `main`.

## P0 — required before training

- [x] Environment snapshot, local source locations, and ignored-output policy.
- [x] Formula/shape/gradient review of the current LECA body.
- [x] SHA256/dHash data audit; five exact train/val duplicates were found.
- [x] Baseline/comparison construction audit; current base has eight implicit LECA modules and comparison YAMLs are structurally unequal.
- [x] Direct synthetic LECA forward/backward/finite check.
- [!] Recover the exact paper source commit and YOLO11 Baseline/ECA/LECA YAMLs, or formally start a separate controlled benchmark.
- [!] Create a new group-aware split without duplicates; archive—not overwrite—the historical split.
- [ ] Build topology-matched Baseline/ECA/LECA models; record parameters, FLOPs, module inventory, and forward shapes.
- [ ] Write a resolved immutable config (seed=42, optimizer, LR, decay, batch, imgsz, augmentations, pretrained hash, AMP, workers, cache, device).
- [ ] Add default-off aggregate hooks; verify no raw feature tensors/images enter Git.
- [ ] Run one batch forward/backward and finite checks.
- [ ] Run 1 epoch smoke per model to distinct `runs_repro/<model>/42/smoke/`; inspect loss, scalar gradients, checkpoints, validation and one test execution.

## P1 — paper reproduction and ablations

Only begin after P0 is fully green.

1. Train Baseline, ECA, LECA using identical recovered configuration with seed 42.
2. Compare the direction/trend against the paper. If opposite, stop and diagnose before expanding seeds.
3. Run seeds 42, 123, 2026 for Baseline/ECA/LECA. Report P, R, mAP50, mAP50-95, best epoch, Params, FLOPs, latency, FPS, mean±std.
4. Train seed-42 ablations: ECA; ECA+Var; ECA+Rec; ECA+Bri; ECA+Var+Rec; ECA+Var+Bri; ECA+Rec+Bri; full LECA. Retrain selected key configurations for three seeds.
5. Keep one evaluation script, one untouched test split, same weights, early-stopping rule, and fixed protocol.

## P2 — mechanism diagnostics, interview evidence, and stress tests

- H1: find equal-mean/different-variance channels and compare ECA/LECA weights; report only feature-statistic observations.
- H2: compare TP/FP/FN distributions of variance, `w_sup`, final weight, including reflection/hole/rivet/low-light cases; use Spearman, effect sizes, distributions.
- H3: compare `mu`/`w_rec` by illumination/outcome and perform inference-only alpha=0 sensitivity.
- H4: correlate `corr` with image luminance and inspect depth/spatial-illumination cases; do not call it physical brightness without evidence.
- H5: use complete retrained ablations plus inference-only beta/alpha/gamma neutralization; label the latter sensitivity analysis.
- Controlled Stress Tests: photometric-only global/radial/linear brightness, gamma, high spots/overexposure, contrast perturbations; preserve labels and record parameters.
- Add `metadata/hard_case_index.csv` with image id, case type, notes only; report hard-case P/R/mAP/FP/FN.
- Store heatmaps/features/stress images only in `artifacts/visualizations/`; store raw feature samples only in ignored `artifacts/statistics/`.
