#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查看npz文件信息"""
import os
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_ROOT)
file_path = Path("factors_by_type_npy/factors_by_type/alpha101/WorldQuant_alpha027.npz")

# 文件大小
size = file_path.stat().st_size
print(f"文件大小: {size / (1024*1024):.2f} MB ({size} bytes)")
print()

# 加载npz文件
f = np.load(file_path, allow_pickle=True)

print("NPZ文件内容:")
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
                # 检查数值类型数组的缺失值
                if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
                    nan_count = np.isnan(arr).sum()
                    total_count = arr.size
                    valid_count = total_count - nan_count
                    nan_percentage = (nan_count / total_count * 100) if total_count > 0 else 0
                    print(f"  缺失值统计:")
                    print(f"    总数据量: {total_count:,}")
                    print(f"    有效值: {valid_count:,} ({100 - nan_percentage:.2f}%)")
                    print(f"    缺失值 (NaN): {nan_count:,} ({nan_percentage:.2f}%)")
                
                if arr.ndim == 1:
                    print(f"  Head (first 5 elements):")
                    print(f"  {arr[:5]}")
                elif arr.ndim == 2:
                    print(f"  Head (first 5 rows):")
                    print(f"  {arr[:5]}")
                else:
                    print(f"  Head (first element):")
                    print(f"  {arr.flat[0]}")
        else:
            print(f"  Value: {arr}")
        print()
    except Exception as e:
        print(f"{key}: Error loading - {e}")
        print()

f.close()

