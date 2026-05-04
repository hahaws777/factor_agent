#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pack all factor directories into zip archives with parallel processing.
Each subdirectory is split into num_parts zip files.
"""
import math
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def pack_directory_files(source_dir, output_dir, num_parts=8, max_workers=4):
    """
    Pack all .npz files in source_dir into num_parts zip files.

    Parameters
    ----------
    source_dir  : source directory path
    output_dir  : output directory path
    num_parts   : number of zip files to create
    max_workers : thread pool size
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    files = sorted(source_path.glob("*.npz"))
    total_files = len(files)

    if total_files == 0:
        print(f"{source_dir}: no files found")
        return 0, 0

    dir_name = source_path.name
    files_per_part = math.ceil(total_files / num_parts)
    print(f"\n{dir_name}: {total_files} files → {num_parts} zips (~{files_per_part} each)")

    def create_zip(group_idx, group_files, zip_filename):
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in group_files:
                    zipf.write(file_path, file_path.name)
            zip_size = zip_filename.stat().st_size / (1024 * 1024)
            return group_idx, zip_filename, zip_size, True, None
        except Exception as e:
            return group_idx, zip_filename, 0, False, str(e)

    tasks = []
    for part_idx in range(num_parts):
        start_idx = part_idx * files_per_part
        end_idx = min(start_idx + files_per_part, total_files)
        if start_idx >= total_files:
            break
        group_files = files[start_idx:end_idx]
        zip_filename = output_path / f"{dir_name}_part{part_idx + 1:02d}.zip"
        tasks.append((part_idx + 1, group_files, zip_filename))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(create_zip, idx, f, zip_path): (idx, zip_path.name)
            for idx, f, zip_path in tasks
        }
        with tqdm(total=len(tasks), desc=dir_name) as pbar:
            for future in as_completed(future_to_task):
                idx, zip_name = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    group_idx, zip_filename, zip_size, success, error = result
                    if success:
                        pbar.set_postfix_str(f"Part {group_idx}: {zip_size:.1f}MB")
                    else:
                        print(f"Error: {zip_name} - {error}")
                except Exception as e:
                    print(f"Error: {zip_name} - {e}")
                finally:
                    pbar.update(1)

    successful = [r for r in results if r[3]]
    total_size = sum(r[2] for r in successful)
    print(f"{dir_name}: {len(successful)}/{len(tasks)} zips done, {total_size:.2f} MB total")
    return len(successful), total_size


def pack_all_factors(base_dir="factors_by_type", output_base="factors_zips", num_parts=8, max_workers=4):
    """
    Pack every subdirectory under base_dir into zip archives.

    Parameters
    ----------
    base_dir    : root directory of factor subdirectories
    output_base : output root directory
    num_parts   : zip files per subdirectory
    max_workers : thread pool size per subdirectory
    """
    base_path = Path(base_dir)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} directories to pack ({num_parts} zips each)")

    total_zips = 0
    total_size = 0
    for subdir in subdirs:
        output_dir = Path(output_base) / subdir.name
        zips, size = pack_directory_files(subdir, output_dir, num_parts, max_workers)
        total_zips += zips
        total_size += size

    print(f"\nAll done — {total_zips} zip files, {total_size / 1024:.2f} GB total")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    pack_all_factors(
        base_dir=str(Path("factors_by_type_npy/factors_by_type")),
        output_base="factors_zips",
        num_parts=8,
        max_workers=4,
    )
