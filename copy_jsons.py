#!/usr/bin/env python3
"""
copy_jsons.py

递归查找工作空间中的所有 .json 文件并复制到新建目录 `collected_jsons/`。
如果目标文件名冲突，会在文件名后添加后缀 `_1`, `_2`, ... 以避免覆盖。

用法:
  python copy_jsons.py

脚本假设在仓库根目录运行。
"""

import argparse
import shutil
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = ROOT / 'collected_jsons'


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(description='Copy all JSON files into a single folder')
    parser.add_argument('--src', type=str, default=str(ROOT), help='Source root to search for JSON files')
    parser.add_argument('--dest', type=str, default=str(DEFAULT_TARGET), help='Destination folder to copy JSON files into')
    parser.add_argument('--dry', action='store_true', help='Dry run: show planned copies but do not perform them')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    src_root = Path(args.src)
    target = Path(args.dest)

    if not src_root.exists():
        print(f'Source root not found: {src_root}')
        sys.exit(1)

    json_files = list(src_root.rglob('*.json'))
    if not json_files:
        print('No JSON files found.')
        return

    if args.dry:
        print(f'Dry run: would copy {len(json_files)} JSON files from {src_root} to {target}')
        for p in json_files:
            print('  ' + str(p))
        return

    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in json_files:
        dest = target / p.name
        dest = unique_dest(dest)
        shutil.copy2(p, dest)
        copied.append(dest.name)
        if args.verbose:
            print(f'Copied {p} -> {dest}')

    print(f'Copied {len(copied)} JSON files to {target}:')
    for name in copied:
        print('  ' + name)


if __name__ == '__main__':
    main()
