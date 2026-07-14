<!-- Generated: concise, actionable guidance for AI coding agents working in this repo -->
# Copilot / AI agent instructions for car_bolt_detection

This repository is a small dataset/workspace for bolt / screw detection in car repair images. The guidance below helps an AI coding agent be immediately productive when making edits, adding models, or preparing experiments.

- Project type: lightweight image dataset + experiments. Key folder: `auto_repair_screws/` contains source images (JPEGs named like `微信图片_20251013...`).
- Big picture: there is no large app framework here — expect work to add scripts for dataset prep, training, or inference. Agents should create clear, minimal additions (scripts, README, requirements) and avoid large, opinionated frameworks.

Actionable rules for code edits

- Keep changes minimal and self-contained. Add new scripts under a top-level `scripts/` or `src/` directory.
- When adding Python code, include a `requirements.txt` with pinned minimal deps (e.g., numpy, opencv-python, torch) and a short `README.md` describing how to run the script.
- Prefer non-destructive edits: do not rename or move existing image files unless asked.

Patterns and examples

- Image naming: files under `auto_repair_screws/` use Chinese filenames starting with `微信图片_YYYYMMDD...`. When writing dataset loaders, treat filenames as opaque IDs and rely on file extension (`.jpg`) for discovery.
- Small dataset workflows: provide helper scripts for:
  - listing and sampling images (`scripts/list_images.py`)
  - splitting train/val (`scripts/split_dataset.py`) by random seed
  - visualizing images (`scripts/visualize_samples.py`) with optional bounding-box overlays

Developer workflows

- No CI detected. For local testing, recommend these commands (add to README):
  - Create venv and install: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt`
  - Run quick script: `python scripts/list_images.py --dir auto_repair_screws`

Integration notes

- There are no external API keys or services in the repo. If adding integrations (cloud storage, experiment tracking), put credentials into environment variables and update README with required env names.

When to ask the user

- If you need to change or remove existing images, confirm first.
- If you propose adding a model training pipeline (large changes), outline files and a short plan before implementing.

Files to reference when editing

- `auto_repair_screws/` — source images (treat as authoritative)
- newly added `scripts/` or `src/` — where agents should place runnable code

If anything in this guidance is unclear or you'd like the agent to follow stricter conventions (tests, CI, dataset formats like COCO/PascalVOC), tell me and I'll update this file.
