#!/usr/bin/env python3
"""
rename_images.py

重命名 `auto_repair_screws/` 目录下的图片为格式：IMG_00001.jpg

用法：
  python rename_images.py         # 直接运行（会执行实际重命名）
  python rename_images.py --dry   # 仅显示将要做的更改（不修改文件）

脚本行为：
 - 按文件名的自然排序（可选按修改时间）处理图片。
 - 使用两阶段重命名（先临时名，再目标名）以避免命名冲突。
 - 支持常见图片扩展名（jpg,jpeg,png,JPG,JPEG,PNG）并将扩展名统一为小写jpg。
 - 会备份打印操作摘要。
"""

import argparse
import os
import re
import sys
import random
from pathlib import Path
from shutil import copy2


IMAGE_DIR = Path(__file__).resolve().parent / 'auto_repair_screws'
VALID_EXT = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def natural_sort_key(s: str):
    # split into ints and text for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def make_new_name(idx: int, prefix: str = 'screw_', digits: int = 3) -> str:
    return f"{prefix}{idx:0{digits}d}.jpg"


def gather_images(dirpath: Path):
    if not dirpath.exists():
        print(f"Directory not found: {dirpath}")
        sys.exit(1)
    files = [p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in {e.lower() for e in VALID_EXT}]
    # sort by natural filename order
    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def two_phase_rename(files, dry_run=False, prefix='screw_', digits=3, start_index=1):
    # Stage 1: compute mapping and check collisions
    mapping = {}
    for i, p in enumerate(files, start=start_index):
        new_name = make_new_name(i, prefix=prefix, digits=digits)
        mapping[p] = p.with_name(new_name)

    # Print plan
    print("Planned renames:")
    for src, dst in mapping.items():
        print(f"  {src.name} -> {dst.name}")

    if dry_run:
        print("\nDry run mode: no files will be changed.")
        return

    # Stage 2: perform safe two-step rename to avoid collisions
    temp_map = {}
    for src in mapping:
        temp = src.with_suffix(src.suffix + '.tmprename')
        src.rename(temp)
        temp_map[temp] = mapping[src]

    # Finalize
    for temp, final in temp_map.items():
        # ensure extension is .jpg
        final = final.with_suffix('.jpg')
        # if final exists, back it up (shouldn't happen due to temp strategy but just in case)
        if final.exists():
            backup = final.with_suffix(final.suffix + '.bak')
            final.rename(backup)
            print(f"Backed up existing {final.name} -> {backup.name}")
        temp.rename(final)

    print('\nRename complete.')


def main():
    parser = argparse.ArgumentParser(description='Rename images with numbering; supports shuffle and custom prefix')
    parser.add_argument('--dry', action='store_true', help='Dry run: only show planned renames')
    parser.add_argument('--by-time', action='store_true', help='Sort by modification time instead of filename')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle files before assigning numbers')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for shuffle')
    parser.add_argument('--dir', type=str, default=str(IMAGE_DIR), help='Directory containing images to rename')
    parser.add_argument('--prefix', type=str, default='screw_', help='Filename prefix (default: screw_)')
    parser.add_argument('--digits', type=int, default=3, help='Number of digits for numbering (default: 3 -> 001)')
    parser.add_argument('--start-index', type=int, default=1, help='Starting index for numbering (default: 1)')
    args = parser.parse_args()

    target_dir = Path(args.dir)
    files = gather_images(target_dir)

    # Sorting / shuffling logic
    if args.by_time and not args.shuffle:
        files.sort(key=lambda p: p.stat().st_mtime)
    if args.shuffle:
        if args.seed is not None:
            random.seed(args.seed)
        files = list(files)
        random.shuffle(files)

    if not files:
        print(f'No images found in {target_dir}')
        return

    two_phase_rename(files, dry_run=args.dry, prefix=args.prefix, digits=args.digits, start_index=args.start_index)


if __name__ == '__main__':
    main()
