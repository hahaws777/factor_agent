#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量分析多个因子的Rank IC
"""

import os
import glob
from factor_rankic_analysis import FactorRankICAnalyzer
import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def batch_analyze_factors(factor_dir, output_dir='rankic_batch_results', **kwargs):
    """
    批量分析因子
    
    参数:
        factor_dir: 因子目录路径
        output_dir: 输出目录
        **kwargs: 传递给calculate_rank_ic的其他参数
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有因子文件（递归）
    factor_files = sorted(glob.glob(os.path.join(factor_dir, '**', '*.pkl'), recursive=True))
    total = len(factor_files)
    print(f"Found {total} factor files in {factor_dir}")
    print("Starting batch analysis...\n")
    
    # 汇总结果
    summary_results = []
    
    iterator = factor_files
    if tqdm is not None:
        iterator = tqdm(factor_files, total=total, desc='Analyzing factors')

    for factor_file in iterator:
        factor_rel = os.path.relpath(factor_file, factor_dir)
        factor_name = os.path.splitext(os.path.basename(factor_file))[0]
        try:
            analyzer = FactorRankICAnalyzer('data.pkl')
            analyzer.load_market_data() \
                    .load_factor(factor_file) \
                    .merge_data() \
                    .calculate_rank_ic(**kwargs)

            results = analyzer.ic_results
            mean_ic = results['ic'].mean()
            mean_rank_ic = results['rank_ic'].mean()
            rank_ic_std = results['rank_ic'].std()
            rank_ic_ir = mean_rank_ic / rank_ic_std if rank_ic_std > 0 else 0

            # 逐因子实时打印 mean rankic 与 std rankic
            print(f"Done: {factor_rel} | mean_rankic: {mean_rank_ic:.6f} | std_rankic: {rank_ic_std:.6f}")

            # 保存结果（文件名包含相对路径信息中的目录，用下划线替代）
            safe_prefix = os.path.dirname(factor_rel).replace(os.sep, '_')
            safe_prefix = (safe_prefix + '_') if safe_prefix else ''
            output_file = os.path.join(output_dir, f'{safe_prefix}{factor_name}_rankic.csv')
            analyzer.save_results(output_file)

            summary_results.append({
                'factor_path': factor_rel,
                'factor_name': factor_name,
                'mean_ic': mean_ic,
                'mean_rank_ic': mean_rank_ic,
                'ic_std': results['ic'].std(),
                'rank_ic_std': rank_ic_std,
                'mean_p_value': results['p_value'].mean() if 'p_value' in results.columns else None,
                'mean_ic_p_value': results['ic_p_value'].mean() if 'ic_p_value' in results.columns else None,
                'ic_ir': mean_ic / results['ic'].std() if results['ic'].std() > 0 else 0,
                'rank_ic_ir': rank_ic_ir,
                'ic_win_rate': (results['ic'] > 0).mean(),
                'rank_ic_win_rate': (results['rank_ic'] > 0).mean(),
                'valid_days': len(results)
            })

        except Exception as e:
            print(f"Error analyzing {factor_rel}: {e}")
            summary_results.append({
                'factor_path': factor_rel,
                'factor_name': factor_name,
                'error': str(e)
            })
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_results)
    summary_file = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"\nBatch analysis completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    # 打印Top 10 Rank IC IR
    if 'rank_ic_ir' in summary_df.columns:
        top_factors = summary_df.nlargest(10, 'rank_ic_ir')
        print("\nTop 10 factors by Rank IC IR:")
        print(top_factors[['factor_name', 'mean_rank_ic', 'rank_ic_ir', 'rank_ic_win_rate']].to_string(index=False))
    
    return summary_df


if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])

    # 示例：默认递归分析 factors_by_type 下所有因子
    if len(sys.argv) > 1:
        factor_dir = sys.argv[1]
    else:
        factor_dir = 'factors_by_type'
    
    # 分析配置
    config = {
        'exclude_st': True,
        'exclude_limit_up': True,
        'exclude_limit_down': True,
        'exclude_suspended': True,
        'use_next_day_return': True
    }
    
    print("="*80)
    print("Batch Factor Rank IC Analysis")
    print("="*80)
    print(f"Factor Directory: {factor_dir}")
    print(f"Configuration: {config}")
    print("="*80)
    
    summary = batch_analyze_factors(factor_dir, **config)

