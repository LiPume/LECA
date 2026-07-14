# LECA interview evidence

## 1. Confirmed facts

The current LECA code multiplies ECA channel weights by variance, low-response, and feature-space global-activation factors. Current scalar initial values are .02/.04/.01 and are trainable parameters. The audited source/dataset state contains an architecture mismatch and train/val leakage; those are reported rather than hidden.

## 2. Paper-stage design assumptions

Variance, low response, and global activation were proposed as statistical clues. They are weak priors, not labels for reflection, bolts, or physical illumination.

## 3–7. What experiments support / do not support

No new controlled experiment has been run because P0 gates failed. Therefore no causal mechanism is claimed. H1–H5 in `EXPERIMENT_TODO.md` specify the planned evidence for variance, low-response, and brightness branches.

## 8–10. Typical cases, failures, reflection

Visual examples and TP/FP/FN mechanism cases are pending a valid fixed protocol. The present failure case is methodological: a model called baseline must be verified to have no LECA, and a validation split must not contain training images.

## 11. Future improvement

First recover the paper version and evaluate it honestly. Only after completing that work may an isolated `exp/leca-v2` explore constrained beta, stable variance, or a spatially explicit illumination mechanism.

## 12. Three-minute interview script

“My work studies bolt detection under difficult visual conditions using LECA, an ECA-based attention mechanism augmented with three feature statistics. I now describe the mechanism conservatively: variance, low response, and global activation are statistical clues, not direct detectors of reflection, bolts, or physical brightness. In the code, LECA computes channel ECA weights, then multiplicatively calibrates them with a softplus-variance suppression term, a low-response recovery term, and a global feature-activation condition.

Before claiming a gain, I audited reproducibility. The formula implementation itself matched the intended tensor operations and had finite forward and backward gradients. But the audit also found two important threats: the current model named baseline already contains implicit LECA modules, and the current train/validation split contains exact duplicate images. Therefore I do not use those numbers as proof of LECA’s superiority.

My next evidence plan is controlled: recover the exact published source and YAML, construct topology-matched Baseline/ECA/LECA, use a group-aware split, and run a 1-epoch smoke test before full training. Then I will report three seeds and complete branch ablations. For mechanism validation, I will compare feature statistics across TP, FP and FN cases, use inference-only branch neutralization as sensitivity analysis, and run controlled photometric stress tests. My conclusion will distinguish observed robustness from causal interpretation.”

## 13. Likely questions and answers

**Why not say high variance means reflection?** Because deep features are learned mixtures; the experiment can show association, not semantic identity.

**Why not train immediately?** A contaminated split and an attention-containing baseline would make an apparently good result scientifically uninterpretable.

**What does brightness mean here?** In this implementation it is a global deep-feature activation statistic. It must be correlated with image luminance and spatial perturbations before calling it illumination related.

**How will you prove robustness?** Fixed untouched test data, identical training protocols, three seeds, and documented controlled photometric stress tests—not selected examples.
