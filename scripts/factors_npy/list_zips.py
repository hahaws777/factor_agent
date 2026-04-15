#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查看各目录zip文件分布"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
base = Path("factors_zips")
dirs = sorted([d for d in base.iterdir() if d.is_dir()])

print("各目录zip文件分布:\n")
for d in dirs:
    zips = sorted(list(d.glob("*.zip")))
    print(f"{d.name}: {len(zips)} 个zip文件")
    for zip_file in zips:
        size = zip_file.stat().st_size / (1024 * 1024)
        print(f"  - {zip_file.name}: {size:.2f} MB")

