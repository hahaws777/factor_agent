#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect a .npz factor file: keys, shapes, dtypes, NaN stats."""
import os
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT)
file_path = Path("factors_by_type_npy/factors_by_type/alpha101/WorldQuant_alpha027.npz")

size = file_path.stat().st_size
print(f"File size: {size / (1024*1024):.2f} MB ({size} bytes)")
print()

f = np.load(file_path, allow_pickle=True)

print("NPZ contents:")
print(f"Keys: {list(f.keys())}")
print()

for key in f.keys():
    try:
        arr = f[key]
        print(f"{key}:")
        if isinstance(arr, np.ndarray):
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            if arr.size > 0:
                if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
                    nan_count = np.isnan(arr).sum()
                    total_count = arr.size
                    valid_count = total_count - nan_count
                    nan_pct = (nan_count / total_count * 100) if total_count > 0 else 0
                    print(f"  NaN stats:")
                    print(f"    Total:  {total_count:,}")
                    print(f"    Valid:  {valid_count:,} ({100 - nan_pct:.2f}%)")
                    print(f"    NaN:    {nan_count:,} ({nan_pct:.2f}%)")

                if arr.ndim == 1:
                    print(f"  Head (first 5): {arr[:5]}")
                elif arr.ndim == 2:
                    print(f"  Head (first 5 rows):\n  {arr[:5]}")
                else:
                    print(f"  Head: {arr.flat[0]}")
        else:
            print(f"  Value: {arr}")
        print()
    except Exception as e:
        print(f"{key}: Error loading - {e}\n")

f.close()
