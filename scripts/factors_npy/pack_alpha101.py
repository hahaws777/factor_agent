#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pack alpha101 .npz files into 8 zip archives (~13 files each)."""
import os
import zipfile
from pathlib import Path
from tqdm import tqdm


def pack_alpha101_files():
    import os
    _root = Path(__file__).resolve().parents[2]
    os.chdir(_root)
    source_dir = Path("factors_by_type_npy/factors_by_type/alpha101")
    output_dir = Path("alpha101_zips")
    output_dir.mkdir(exist_ok=True)

    files = sorted(source_dir.glob("*.npz"))
    total_files = len(files)
    print(f"Found {total_files} files")

    files_per_group = 13
    num_groups = 8

    for group_idx in range(num_groups):
        start_idx = group_idx * files_per_group
        end_idx = min(start_idx + files_per_group, total_files)
        if start_idx >= total_files:
            break

        group_files = files[start_idx:end_idx]
        zip_filename = output_dir / f"alpha101_part{group_idx + 1:02d}.zip"
        print(f"\nPacking group {group_idx + 1} ({len(group_files)} files) -> {zip_filename.name}")

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tqdm(group_files, desc=f"Part {group_idx + 1}"):
                zipf.write(file_path, file_path.name)

        zip_size = zip_filename.stat().st_size / (1024 * 1024)
        print(f"Done: {zip_filename.name} ({zip_size:.2f} MB)")

    print(f"\nAll done — {num_groups} zip files written to {output_dir}")


if __name__ == "__main__":
    pack_alpha101_files()
