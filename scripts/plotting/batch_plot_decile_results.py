#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量汇总所有因子的累计收益并生成图表
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
    绘制单个因子的累计收益曲线图（Q1..Q10 与 LS），英文标签。
    
    参数:
        cum_file: 累计收益CSV文件路径
        output_dir: 输出目录
        factor_name: 因子名称
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
    
    # 识别分组列与LS
    cols = [c for c in cum_df.columns if c != 'date']
    ls_col = 'LS' if 'LS' in cols else None
    quant_cols = [c for c in cols if c != 'LS']
    quant_cols_sorted = sorted(quant_cols, key=lambda x: int(x[1:]) if x.startswith('Q') and x[1:].isdigit() else 0)
    
    if not quant_cols_sorted:
        print(f"No quantile columns found in {cum_file}")
        return False
    
    # 绘制全部分组 + LS
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
    
    # 单独绘制LS（若存在）
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
    汇总所有因子的累计收益数据
    
    参数:
        input_dir: 输入目录（包含所有 *_decile_cum.csv 文件）
        output_dir: 输出目录
        summary_file: 汇总文件名
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有累计收益文件
    cum_files = sorted(glob.glob(os.path.join(input_dir, '*_decile_cum.csv')))
    total = len(cum_files)
    
    if total == 0:
        print(f"No decile cumulative files found in {input_dir}")
        return None
    
    print(f"Found {total} decile cumulative files")
    
    # 汇总数据
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
                
                # 获取最后一天的累计收益
                last_row = df.iloc[-1]
                row_data = {
                    'factor_name': factor_name,
                    'valid_days': len(df)
                }
                
                # 添加各分组的最终累计收益
                for col in df.columns:
                    if col != 'date':
                        if pd.notna(last_row[col]):
                            row_data[col] = last_row[col]
                        else:
                            row_data[col] = None
                
                summary_data.append(row_data)
        except Exception as e:
            print(f"Error processing {cum_file}: {e}")
    
    if not summary_data:
        print("No valid data found")
        return None
    
    # 创建汇总DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # 按日期对齐所有因子的累计收益
    all_dates = sorted(all_dates)
    aligned_data = {'date': all_dates}
    
    # 为每个因子创建对齐的累计收益序列
    for cum_file in cum_files:
        factor_name = os.path.basename(cum_file).replace('_decile_cum.csv', '')
        try:
            df = pd.read_csv(cum_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                # 对齐到所有日期
                for col in df.columns:
                    col_name = f"{factor_name}_{col}"
                    aligned_series = df[col].reindex(all_dates, method='ffill')
                    aligned_data[col_name] = aligned_series.values
        except Exception as e:
            print(f"Error aligning {cum_file}: {e}")
    
    # 保存对齐后的完整数据（可选，可能很大）
    aligned_df = pd.DataFrame(aligned_data)
    aligned_path = os.path.join(output_dir, 'decile_cumulative_aligned_all_factors.csv')
    aligned_df.to_csv(aligned_path, index=False, encoding='utf-8-sig')
    print(f"Aligned data saved to: {aligned_path}")
    
    # 保存汇总统计
    summary_path = os.path.join(output_dir, summary_file)
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"Summary saved to: {summary_path}")
    
    return summary_df


def batch_plot_all_factors(input_dir, output_dir):
    """
    为所有因子生成累计收益图
    
    参数:
        input_dir: 输入目录（包含所有 *_decile_cum.csv 文件）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有累计收益文件
    cum_files = sorted(glob.glob(os.path.join(input_dir, '*_decile_cum.csv')))
    total = len(cum_files)
    
    if total == 0:
        print(f"No decile cumulative files found in {input_dir}")
        return
    
    print(f"Found {total} decile cumulative files")
    print(f"Generating plots...")
    
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
    
    print(f"\nPlotting completed!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])

    # 默认使用 decile_cpp_batch_results 目录
    input_dir = 'decile_cpp_batch_results'
    output_dir = 'decile_plots'
    summary_file = 'decile_summary_all_factors.csv'
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print("="*80)
    print("Batch Plot Decile Results")
    print("="*80)
    print(f"Input Dir: {input_dir}")
    print(f"Output Dir: {output_dir}")
    print("="*80)
    
    # 1. 汇总所有因子的累计收益
    print("\n1. Summarizing all factors...")
    summary_df = summarize_all_factors(input_dir, output_dir, summary_file)
    
    # 2. 为每个因子生成图表
    print("\n2. Generating plots for all factors...")
    batch_plot_all_factors(input_dir, output_dir)
    
    print("\nAll tasks completed!")

