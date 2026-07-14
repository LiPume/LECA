#!/usr/bin/env python3
"""Local-only YOLO split audit. It writes aggregate JSON, never copies images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def dhash(path: Path, size: int = 8) -> str:
    with Image.open(path) as image:
        image = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    bits = [pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1] for row in range(size) for col in range(size)]
    return f"{sum(bit << index for index, bit in enumerate(bits)):0{size * size // 4}x}"


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="YOLO dataset YAML")
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_audit/summary.json"))
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    args = parser.parse_args()

    config = yaml.safe_load(args.data.read_text())
    root = Path(config["path"])
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    classes = set(config.get("names", {}).keys()) if isinstance(config.get("names"), dict) else set(range(len(config.get("names", []))))
    images, sha256, dhashes, labels = {}, defaultdict(list), {}, Counter()
    empty, invalid, missing_labels, corrupt = [], [], [], []

    for split in ("train", "val", "test"):
        value = config.get(split)
        if not value:
            continue
        image_dir = root / value
        paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        images[split] = [str(path.relative_to(root)) for path in paths]
        for image_path in paths:
            relative = str(image_path.relative_to(root))
            sha256[hashlib.sha256(image_path.read_bytes()).hexdigest()].append((split, relative))
            try:
                dhashes[(split, relative)] = dhash(image_path)
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as error:  # Record and continue; never silently skip.
                corrupt.append({"split": split, "image": relative, "error": str(error)})
            if root.name == "images":
                split_relative = image_path.relative_to(root)
                label_path = root.parent / "labels" / split_relative.with_suffix(".txt")
            else:
                split_relative = image_path.relative_to(root / "images")
                label_path = root / "labels" / split_relative.with_suffix(".txt")
            if not label_path.exists():
                missing_labels.append({"split": split, "image": relative})
                continue
            lines = [line.strip() for line in label_path.read_text(errors="replace").splitlines() if line.strip()]
            if not lines:
                empty.append({"split": split, "image": relative})
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                try:
                    category, x, y, width, height = map(float, parts)
                    is_invalid = len(parts) != 5 or category != int(category) or int(category) not in classes or any(
                        coordinate < 0 or coordinate > 1 for coordinate in (x, y, width, height)
                    ) or width <= 0 or height <= 0
                except ValueError:
                    is_invalid = True
                    category = None
                if is_invalid:
                    invalid.append({"split": split, "image": relative, "line": line_number, "value": line})
                else:
                    labels[int(category)] += 1

    exact_duplicates = [group for group in sha256.values() if len(group) > 1]
    near_duplicates = []
    keys = list(dhashes)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            distance = hamming(dhashes[left], dhashes[right])
            if distance <= args.near_duplicate_threshold:
                near_duplicates.append({"left": left, "right": right, "hamming": distance})

    summary = {
        "data_yaml": str(args.data),
        "resolved_root": str(root),
        "image_counts": {split: len(paths) for split, paths in images.items()},
        "class_instance_counts": dict(labels),
        "empty_labels": empty,
        "invalid_labels": invalid,
        "missing_labels": missing_labels,
        "corrupt_images": corrupt,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_pairs": near_duplicates,
        "note": "Image pixels are neither copied nor written; this file contains metadata only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("image_counts", "class_instance_counts", "exact_duplicate_groups", "near_duplicate_pairs")}, indent=2))


if __name__ == "__main__":
    main()
