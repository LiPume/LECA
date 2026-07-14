# Results summary

## Audit-stage result (not a reproduction result)

| Status | Result |
| --- | --- |
| Confirmed | The current formula body matches the requested LECA equations and has finite synthetic forward/backward gradients. |
| Confirmed | Current `yolo11.yaml` has eight implicit LECA modules, so it is not a baseline. |
| Confirmed | Current comparison YAMLs are structurally mismatched and lower-parameter; no fair Params/FLOPs comparison exists. |
| Confirmed | The historical canonical train/val split has five exact cross-split duplicates. |
| Confirmed | The exact paper YOLO11 LECA YAML/source version is absent from the current project. |
| Not run | Paper-reproduction smoke, seed=42 reproduction, multi-seed study, ablations, stress tests, hard-case metrics. |

No numerical detection metric from the existing historical run directories is re-reported here because the architecture/split provenance is unresolved.

## Controlled mechanism-audit smoke (separate branch, not paper reproduction)

On `exp/mechanism-audit`, three topology-matched configurations were constructed without changing `paper-original`: Identity Baseline (2,624,080 parameters), ECA (2,624,116; +36), and LECA (2,624,140; +60 versus Baseline, +24 versus ECA). All place the selected module at the same eight C3k2 sites.

**Confirmed:** each configuration completed a seed-42, 1-epoch smoke on CUDA:0; pretrained transfer, checkpoint saving, training, validation, and local visual outputs succeeded. These 1-epoch metrics are intentionally not reported as method performance: the model is not converged and the historical validation split remains duplicated.

**Observed:** a corrected post-initialization LECA hook recorded all eight layers × eight aggregate statistics on the LECA smoke. It found no NaN or Inf values. Alpha, beta, and gamma moved from their initial values in all layers after one epoch. This shows the three factors participate in optimization; it does not establish that any factor detects reflection, weak bolts, or physical brightness.

Local-only outputs are in `runs_repro/mechanism_smoke/`, including normal training/validation images and LECA aggregate CSVs. The first LECA smoke lacked aggregate statistics because hooks were attached before the trainer copied the model; this was recorded, fixed, and repeated as `leca_stats_retry` rather than overwritten.
