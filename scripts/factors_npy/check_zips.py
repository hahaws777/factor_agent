#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查打包结果"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
base = Path("factors_zips")
dirs = [d for d in base.iterdir() if d.is_dir()]

print("打包结果统计:\n")
for d in sorted(dirs):
    zips = list(d.glob("*.zip"))
    total_size = sum(f.stat().st_size for f in zips) / (1024 * 1024)  # MB
    print(f"{d.name}: {len(zips)} 个zip文件, {total_size:.2f} MB")

total_zips = sum(len(list(d.glob("*.zip"))) for d in dirs)
total_size = sum(sum(f.stat().st_size for f in d.glob("*.zip")) for d in dirs) / (1024 * 1024 * 1024)  # GB
print(f"\n总计: {total_zips} 个zip文件, {total_size:.2f} GB")

