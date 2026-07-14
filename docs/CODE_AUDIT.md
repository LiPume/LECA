# LECA conservative code audit

**Audit status: long training blocked.** This report describes the local state captured on 2026-07-14. It does not alter the published implementation. Terms are deliberate: **Confirmed** is directly established by source, checkpoint, or local audit output; **Observed** is an experiment observation; **Hypothesis** remains untested.

## A. Project structure and execution paths

| Item | Location / finding |
| --- | --- |
| Project root | `/home/lzx/car_bolt_detection` |
| Main source tree | `ultralytics-main/` (Ultralytics 8.3.215) |
| Historical training scripts | `ultralytics-main/train_test.py`, `train_abc.py`, `run_ablation_parallel.py`, `train_adaptive_ablation.py`, `train_cascade_ablation.py` |
| Historical inference/evaluation | `detect.py`, `eval_ablation_final.py`, `final_latency_test.py`, `compare_gt_pred*.py` |
| Canonical historical train YAML | `ultralytics-main/ultralytics/cfg/models/datasets/screw.yaml` → `dataset/trainDataV3` |
| Historical hard-test YAML | `dataset/hardData/YOLODataset/hard_test_set.yaml` |
| Base model YAML | `ultralytics/cfg/models/11/yolo11.yaml` |
| Comparison YAMLs | `yolo11EMA.yaml`, `yolo11SE.yaml`, `yolo11CBAM.yaml`, `yolo11NAM.yaml` |
| Active parser registration | `ultralytics/nn/tasks.py` imports `LECA` from `ultralytics.nn.modules`; that export resolves to `block.py`'s class |
| LECA implementations present | `ultralytics/nn/modules/attention.py` and `block.py` both define a same-named `LECA` |

**Confirmed:** no YOLO11 LECA model YAML exists in `ultralytics/cfg/models/11/`. The only LECA YAMLs are ResNet/RT-DETR variants. Historical run `args.yaml` files merely say `model: yolo11n.yaml`; they do not preserve the exact customized YAML text or source commit.

### Actual insertion and version drift

**Confirmed:** `C2f` in `block.py` imports `LECA` from `attention.py` and unconditionally contains `self.eca = LECA(c2)` followed by `out = self.eca(out)`. `C3k2` derives from this block. Thus the current `yolo11.yaml` build has eight attention modules: model indices 2, 4, 6, 8, 13, 16, 19, 22. They occur in backbone (4), neck (3), and the final pre-detect block (1); none is a separately declared LECA YAML layer.

**Confirmed:** the current source-built `yolo11.yaml` (nano scale) has 2,624,140 parameters and eight `block.LECA` modules. This is not a no-attention baseline.

**Confirmed:** historical checkpoint inspection finds `car_bolt_baseline/best.pt` has no LECA modules, whereas historical `car_bolt_LECA*` and ablation checkpoints serialize eight `attention.LECA` modules. The current `block.LECA` and `attention.LECA` are distinct Python classes. Some older LECA checkpoints carry only alpha/beta/Conv state, whereas current source declares gamma as a parameter. This proves source/checkpoint drift; it does not identify the exact missing historical source revision.

## B. Published-formula to code correspondence

The following mapping is for the currently active *formula body* in both same-named source variants. The build path above determines which class a given model uses.

| Expected operation | Current code | Status |
| --- | --- | --- |
| `X: [B,C,H,W]` | `x.size()` | Confirmed |
| `mu=mean(X,H,W): [B,C,1,1]` | `mean = x.mean(dim=(2,3), keepdim=True)` | Confirmed |
| `m=mean(X,C): [B,1,H,W]` | `local_mean = torch.mean(x, dim=1, keepdim=True)` | Confirmed |
| ECA | GAP → reshape `[B,1,C]` → Conv1D → sigmoid | Confirmed |
| `var=mean(X²)-mu²` | `mean2 - mean * mean` | Confirmed |
| `noise=softplus(var)` | `F.softplus(var)` | Confirmed |
| `w_sup=1/(1+beta*noise)` | identical | Confirmed |
| `w_rec=1+alpha*sigmoid(-mu)` | identical (`low`) | Confirmed |
| `corr=mean(sigmoid(-m),H,W)` | identical; `[B,1,1,1]` | Confirmed |
| `w_bri=1+gamma*corr` and channel broadcast | identical PyTorch broadcast | Confirmed |
| Product and output | `x * (w_eca*w_sup*w_rec*w_bri)` | Confirmed |
| Initial scalars | alpha=.02, beta=.04, gamma=.01 in current classes | Confirmed for current source, not all historical checkpoints |

The ECA kernel is adaptive: `t=int(abs((log2(C)+1)/2))`, made odd. It is 3 for C=64 and 5 for C≥128 in the direct audit. The Conv1D has exactly `k` trainable weights; current LECA adds three scalar parameters, so a module has `k+3` parameters.

### Tensor, parameter, and gradient audit

**Confirmed:** direct CPU/GPU-compatible forward/backward checks for C={64,128,256,512,1024} produced finite output and input gradients. alpha, beta, and gamma are scalar `nn.Parameter`s in the current implementation and each received nonzero gradients in that synthetic test. Each module owns independent scalar objects; they are not globally shared. They are returned by `model.parameters()`, so a standard Ultralytics optimizer will include them.

**Confirmed:** there is no `detach()`, `.item()`, or NumPy conversion in the current LECA forward path.

**Confirmed:** variance is the population-moment identity, not an unbiased sample variance. There is no `clamp_min(0)`. Exact arithmetic is non-negative, but cancellation can produce a small negative value in FP32/AMP; this needs stress testing before any claim of numerical robustness.

**Potential bug / risk:** alpha, beta, gamma are unconstrained. If beta crosses below zero, `1 + beta*softplus(var)` can approach zero, become negative, or overflow; no epsilon, parameterization, range check, or NaN guard exists. AMP/FP16 can worsen cancellation in `mean(x*x)-mean(x)^2` and denominator sensitivity.

At initialization, assuming beta≥0: `w_sup` is positive and at most `1/(1+.04*ln(2))≈0.973`; `w_rec∈[1,1.02]`; `w_bri∈[1,1.01]`; and `w_eca∈(0,1)`. Training removes these bounds because scalars are unconstrained. Therefore a final weight slightly above one is possible even initially, and arbitrarily large/non-finite values are possible in principle after training. This is a **Confirmed mathematical property**, not an observed failure.

LECA operates after the C2f/C3k2 internal convolutions (`cv2`) and their Conv/BN/SiLU processing, not on raw pixels. Its inputs can be negative. `corr` is a feature-space global activation statistic, not a direct physical brightness measurement. No causal “reflection/weak-bolt/illumination” claim is supported by code alone.

## C. Model-structure fairness audit

| Current source YAML | Parameters | Explicit comparison modules | Implicit `attention.LECA` in C2f/C3k2 | Finding |
| --- | ---: | --- | ---: | --- |
| `yolo11.yaml` | 2,624,140 | none | 8 | Not baseline |
| `yolo11EMA.yaml` | 2,316,682 | EMA: 2 (48+176 params) | 8 | Not a LECA-free or architecture-matched comparison |
| `yolo11SE.yaml` | 2,319,018 | SE: 2 (512+2,048 params) | 8 | Same issue |
| `yolo11NAM.yaml` | 2,317,226 | NAM: 2 (256+512 params) | 8 | Same issue |
| `yolo11CBAM.yaml` | 2,337,326 | CBAM: 2 (4,258+16,610 params) | 8 | Same issue |

**Confirmed:** the comparison YAMLs change C3k2 repeat specifications from base `2/2/2/2` (before nano depth scaling) to `3/6/3` at early backbone positions, and change several channel declarations. They also insert modules at different indices. Consequently, their parameter counts being *lower* than the 2.624M current base is explained by architecture change, not attention efficiency. The table cannot support a fair attention comparison.

**Confirmed:** the comments inside the YAMLs claim base parameter/FLOPs values, but those comments do not match constructed models after local modifications. No consistent FLOPs report for Baseline/ECA/LECA was found. `final_latency_test.py` benchmarks only one ULECA checkpoint, uses GPU 5, 200 warm-ups and 1,000 model-only calls, and does not include preprocessing/NMS/data transfer. It cannot be used as a cross-method FPS comparison.

**Blocking missing artifact:** the published YOLO11 LECA YAML and the exact source revision/configuration for historical baseline, ECA, and LECA are not recoverable from the current tracked files. Do not label a current run as a paper reproduction until these are recovered or a clearly new controlled benchmark is declared.

## D. Local data audit

The canonical historical `screw.yaml` points to `dataset/trainDataV3`: 122 train images, 37 val images, one class, 159 instances, no empty labels, no corrupt images, no missing labels, and no invalid/out-of-range boxes.

**Confirmed leakage:** SHA256 found eight exact duplicate image groups; five groups cross train/val:

- train `screw_139_8711369d.jpg` = val `screw_146_990d855c.jpg`
- train `screw_141_75d82b48.jpg` = val `screw_134_b9a045fb.jpg`
- train `screw_144_ff5c5ff8.jpg` = val `screw_137_bb9b62c9.jpg`
- train `screw_147_8da4021e.jpg` = val `screw_140_568185d8.jpg`
- train `screw_189_bbfd82c9.jpg` = val `screw_185_2e22ccb2.jpg`

Two further train-only near-duplicate pairs have dHash distance ≤4. This is sufficient to block use of the current validation metrics as unbiased evidence. No data were altered.

The hard-test YAML maps `train`, `val`, and `test` all to the same `test/` directory as a compatibility workaround. It contains 146 unique test images and 155 one-class instances; those three entries are not independent splits. Historical scripts load it only after training, but the audit cannot prove from source/logs alone that it was never used for checkpoint selection or tuning. Treat that as a reproducibility risk until experiment logs establish the chronology.

Near-duplicate metadata are local-only in ignored `artifacts/data_audit/`; the reusable auditor is `tools/audit_dataset.py`.

## E. Historical training configuration

Historical `args.yaml` files consistently record: epochs=200, patience=20, imgsz=640, batch=16, `optimizer=auto`, pretrained=`yolo11n.pt`, AMP=true, deterministic=true, seed=0, cache=false, workers 0 or 8, mosaic=1.0, close_mosaic=10, translate=.1, scale=.5, fliplr=.5, and no mixup/copy-paste. Device varies (0, 3, 5, 7). The requested paper seed=42 is absent.

**Reproducibility risks:** optimizer auto-selection can change by Ultralytics version; dataset YAML contains absolute Linux and stale Windows paths; no immutable requirements lock or code commit was preserved before this audit; model architecture is not recorded in historical run arguments; and historical results use multiple devices/config generations.

## F. Conclusions

### Confirmed implementation facts

1. The current LECA formula body matches the supplied equations and has correct shapes/broadcasting.
2. Scalars are independent registered parameters in current source; the formula has valid synthetic backward gradients.
3. Current `yolo11.yaml` contains eight implicit LECA modules and is therefore not a baseline.
4. Current comparison YAMLs change backbone structure and retain implicit LECA, so their Params/FLOPs are not fair method comparisons.
5. Current data split has five exact train/validation image duplicates.
6. Git snapshot `paper-original` preserves the local code state; local raw data, images, weights, runs, and visualizations are ignored.

### Potential bugs

- Unconstrained beta can make the suppression denominator unstable.
- Variance has no non-negative clamp and no AMP-specific protection.
- Same class name in two source files creates ambiguous registration/serialization behavior.
- Current implicit LECA inside every C2f/C3k2 makes a baseline impossible without controlled recovery of the intended original code.

### Reproducibility and fairness risks

- Missing exact published YAML/source revision; historical checkpoints and source disagree.
- Leakage in canonical train/val split.
- No clean ECA-only YAML found; ablation scripts set `ULECA_*` environment variables, but the active current LECA does not read them.
- Historical hard-test protocol and checkpoint-selection chronology are not fully evidenced.

### Mechanism claims not yet supported

No observation currently establishes that high variance is reflection, low mean is a bolt, or deep feature mean is physical brightness. The specified H1--H5 are valid future hypotheses only.

### Required fixes *before any rerun*

1. Recover and archive the exact published source/YAML, or declare a new benchmark rather than a paper reproduction.
2. Build three explicit, topology-matched YAMLs (Baseline/ECA/LECA) from that recovered source and verify module inventory before training.
3. Create a new, group-aware split that removes exact and near duplicate leakage; never overwrite the historical split.
4. Lock an explicit train/val/test protocol and prohibit hard-test use for early stopping/tuning.
5. Add non-invasive numerical/activation hooks and a checkpointed configuration manifest only after the published build is recovered.

Until items 1--4 are resolved, smoke/full training would not validate the published LECA and is intentionally not run.
