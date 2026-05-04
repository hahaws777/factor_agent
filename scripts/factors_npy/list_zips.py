#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""List zip file distribution across factor subdirectories."""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
base = Path("factors_zips")
dirs = sorted([d for d in base.iterdir() if d.is_dir()])

print("Zip file distribution by directory:\n")
for d in dirs:
    zips = sorted(list(d.glob("*.zip")))
    print(f"{d.name}: {len(zips)} zip files")
    for zip_file in zips:
        size = zip_file.stat().st_size / (1024 * 1024)
        print(f"  - {zip_file.name}: {size:.2f} MB")
