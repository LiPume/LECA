# Cleanup inventory

The repository has been made Git-clean without deleting evidence. The following local content is intentionally retained on disk but ignored by Git: raw/converted annotations (`auto_repair_screws/`, `collected_jsons/`, `ultralytics-main/dataset/`), all images/videos, weights, `runs/`, visualizations, caches, logs, and generated artifacts.

No existing data, original experiment output, or checkpoint was deleted. That is deliberate: deletion would violate the audit requirement to preserve the published experimental record. After the audit is complete, safe candidates for a separate user-approved local cleanup are Python caches, `.idea/`, generated label caches, and duplicate *derived* files; raw images/labels and historic runs remain evidence and need a retention decision, not automatic removal.
