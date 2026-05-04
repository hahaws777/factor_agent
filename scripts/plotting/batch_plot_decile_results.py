#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch decile cumulative return summary and plot generator.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def plot_decile_cumulative(cum_file, output_dir, factor_name):
    """
    Plot cumulative return curves (Q1..Q10 and LS) for one factor.

    Parameters
    ----------
    cum_file   : path to *_decile_cum.csv
    output_dir : output directory
    factor_name: factor display name
    """
    try:
        cum_df = pd.read_csv(cum_file)
    except Exception as e:
        print(f"Error reading {cum_file}: {e}")
        return False

    if 'date' in cum_df.columns:
        cum_df['date'] = pd.to_datetime(cum_df['date'])
        cum_df = cum_df.sort_values('date')
    else:
        print(f"No 'date' column in {cum_file}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # Identify quantile columns and LS
    cols = [c for c in cum_df.columns if c != 'date']
    ls_col = 'LS' if 'LS' in cols else None
    quant_cols = [c for c in cols if c != 'LS']
    quant_cols_sorted = sorted(quant_cols, key=lambda x: int(x[1:]) if x.startswith('Q') and x[1:].isdigit() else 0)

    if not quant_cols_sorted:
        print(f"No quantile columns found in {cum_file}")
        return False

    # Plot all quantiles + LS
    plt.figure(figsize=(12, 6))
    for c in quant_cols_sorted:
        if c in ['Q1', 'Q10']:
            continue
        plt.plot(cum_df['date'], cum_df[c], color='#999999', linewidth=1.0, alpha=0.8)

    if 'Q1' in quant_cols_sorted:
        plt.plot(cum_df['date'], cum_df['Q1'], color='#d62728', linewidth=1.8, label='Q1')
    if f"Q{len(quant_cols_sorted)}" in quant_cols_sorted:
        qh = f"Q{len(quant_cols_sorted)}"
        plt.plot(cum_df['date'], cum_df[qh], color='#1f77b4', linewidth=1.8, label=qh)
    if ls_col is not None and ls_col in cum_df.columns:
        plt.plot(cum_df['date'], cum_df[ls_col], color='#2ca02c', linewidth=2.0, label='LS')

    plt.title(f'Cumulative Return (Decile Portfolios) - {factor_name}', fontsize=13)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_all = os.path.join(output_dir, f"{factor_name}_decile_cum_all.png")
    plt.tight_layout()
    plt.savefig(out_all, dpi=150)
    plt.close()

    # Separate LS plot (if available)
    if ls_col is not None and ls_col in cum_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(cum_df['date'], cum_df[ls_col], color='#2ca02c', linewidth=2.0, label='LS')
        plt.title(f'Cumulative Return (Long-Short) - {factor_name}', fontsize=13)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_ls = os.path.join(output_dir, f"{factor_name}_decile_cum_LS.png")
        plt.tight_layout()
        plt.savefig(out_ls, dpi=150)
        plt.close()

    return True


def summarize_all_factors(input_dir, output_dir, summary_file='decile_summary_all_factors.csv'):
    """
    Aggregate terminal cumulative returns across all factor decile CSV files.

    Parameters
    ----------
    input_dir   : directory containing *_decile_cum.csv files
    output_dir  : output directory
    summary_file: filename for the summary CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    cum_files = sorted(glob.glob(os.path.join(input_dir, '*_decile_cum.csv')))
    total = len(cum_files)

    if total == 0:
        print(f"No decile cumulative files found in {input_dir}")
        return None

    print(f"Found {total} decile cumulative files")

    summary_data = []
    all_dates = set()

    iterator = cum_files
    if tqdm is not None:
        iterator = tqdm(cum_files, total=total, desc='Reading files')

    for cum_file in iterator:
        factor_name = os.path.basename(cum_file).replace('_decile_cum.csv', '')
        try:
            df = pd.read_csv(cum_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                all_dates.update(df['date'].values)

                last_row = df.iloc[-1]
                row_data = {
                    'factor_name': factor_name,
                    'valid_days': len(df),
                }
                for col in df.columns:
                    if col != 'date':
                        row_data[col] = last_row[col] if pd.notna(last_row[col]) else None

                summary_data.append(row_data)
        except Exception as e:
            print(f"Error processing {cum_file}: {e}")

    if not summary_data:
        print("No valid data found")
        return None

    summary_df = pd.DataFrame(summary_data)

    # Date-align all factors for a wide cumulative return matrix
    all_dates = sorted(all_dates)
    aligned_data = {'date': all_dates}

    for cum_file in cum_files:
        factor_name = os.path.basename(cum_file).replace('_decile_cum.csv', '')
        try:
            df = pd.read_csv(cum_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                for col in df.columns:
                    aligned_data[f"{factor_name}_{col}"] = df[col].reindex(all_dates, method='ffill').values
        except Exception as e:
            print(f"Error aligning {cum_file}: {e}")

    aligned_df = pd.DataFrame(aligned_data)
    aligned_path = os.path.join(output_dir, 'decile_cumulative_aligned_all_factors.csv')
    aligned_df.to_csv(aligned_path, index=False, encoding='utf-8-sig')
    print(f"Aligned data saved to: {aligned_path}")

    summary_path = os.path.join(output_dir, summary_file)
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"Summary saved to: {summary_path}")

    return summary_df


def batch_plot_all_factors(input_dir, output_dir):
    """
    Generate cumulative return plots for every *_decile_cum.csv in input_dir.

    Parameters
    ----------
    input_dir  : directory containing *_decile_cum.csv files
    output_dir : output directory for PNG files
    """
    os.makedirs(output_dir, exist_ok=True)

    cum_files = sorted(glob.glob(os.path.join(input_dir, '*_decile_cum.csv')))
    total = len(cum_files)

    if total == 0:
        print(f"No decile cumulative files found in {input_dir}")
        return

    print(f"Found {total} decile cumulative files")

    iterator = cum_files
    if tqdm is not None:
        iterator = tqdm(cum_files, total=total, desc='Plotting')

    success_count = 0
    error_count = 0

    for cum_file in iterator:
        factor_name = os.path.basename(cum_file).replace('_decile_cum.csv', '')
        if plot_decile_cumulative(cum_file, output_dir, factor_name):
            success_count += 1
        else:
            error_count += 1

    print(f"\nPlotting completed — success: {success_count}  errors: {error_count}")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    os.chdir(Path(__file__).resolve().parents[2])

    input_dir = sys.argv[1] if len(sys.argv) > 1 else 'decile_cpp_batch_results'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'decile_plots'
    summary_file = 'decile_summary_all_factors.csv'

    print("=" * 80)
    print("Batch Plot Decile Results")
    print("=" * 80)
    print(f"Input Dir : {input_dir}")
    print(f"Output Dir: {output_dir}")
    print("=" * 80)

    print("\n1. Summarizing all factors...")
    summarize_all_factors(input_dir, output_dir, summary_file)

    print("\n2. Generating plots for all factors...")
    batch_plot_all_factors(input_dir, output_dir)

    print("\nDone.")
