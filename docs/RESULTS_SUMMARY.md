# Results summary

## Audit-stage result (not a reproduction result)

| Status | Result |
| --- | --- |
| Confirmed | The current formula body matches the requested LECA equations and has finite synthetic forward/backward gradients. |
| Confirmed | Current `yolo11.yaml` has eight implicit LECA modules, so it is not a baseline. |
| Confirmed | Current comparison YAMLs are structurally mismatched and lower-parameter; no fair Params/FLOPs comparison exists. |
| Confirmed | The historical canonical train/val split has five exact cross-split duplicates. |
| Confirmed | The exact paper YOLO11 LECA YAML/source version is absent from the current project. |
| Not run | Smoke test, seed=42 reproduction, multi-seed study, ablations, stress tests, hard-case metrics. |

No numerical detection metric from the existing historical run directories is re-reported here because the architecture/split provenance is unresolved.
