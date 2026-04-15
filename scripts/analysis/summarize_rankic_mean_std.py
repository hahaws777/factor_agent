#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def main():
    from pathlib import Path
    import os
    _root = Path(__file__).resolve().parents[2]
    os.chdir(_root)
    base_dir = 'rankic_batch_results'
    pattern = os.path.join(base_dir, '**', '*_rankic.csv')
    files = sorted(glob.glob(pattern, recursive=True))

    print(f"Total files: {len(files)}")

    rows = []
    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, total=len(files), desc='Aggregating')

    for fp in iterator:
        try:
            df = pd.read_csv(fp, usecols=['rank_ic'])
            mean_rank_ic = df['rank_ic'].mean()
            std_rank_ic = df['rank_ic'].std()
            valid_days = df['rank_ic'].shape[0]

            rel = os.path.relpath(fp, base_dir)
            factor_name = os.path.splitext(os.path.basename(fp))[0].replace('_rankic', '')
            rows.append({
                'factor_file': rel,
                'factor_name': factor_name,
                'mean_rank_ic': mean_rank_ic,
                'rank_ic_std': std_rank_ic,
                'valid_days': valid_days,
            })
        except Exception as e:
            print(f"Error: {fp}: {e}")

    out_path = os.path.join(base_dir, 'rankic_mean_std_summary.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()


