#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize zip archive count and size under factors_zips/."""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
base = Path("factors_zips")
dirs = [d for d in base.iterdir() if d.is_dir()]

print("Pack results:\n")
for d in sorted(dirs):
    zips = list(d.glob("*.zip"))
    total_size = sum(f.stat().st_size for f in zips) / (1024 * 1024)
    print(f"{d.name}: {len(zips)} zip files, {total_size:.2f} MB")

total_zips = sum(len(list(d.glob("*.zip"))) for d in dirs)
total_size = sum(sum(f.stat().st_size for f in d.glob("*.zip")) for d in dirs) / (1024 * 1024 * 1024)
print(f"\nTotal: {total_zips} zip files, {total_size:.2f} GB")
