#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量分析多个因子的Rank IC
"""

import os
import glob
import time
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from factor_rankic_analysis import FactorRankICAnalyzer
import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_WORKER_ANALYZER = None


def _build_summary_record(analyzer, factor_rel, factor_name, elapsed):
    results = analyzer.ic_results
    mean_ic = results['ic'].mean()
    mean_rank_ic = results['rank_ic'].mean()
    rank_ic_std = results['rank_ic'].std()
    rank_ic_ir = mean_rank_ic / rank_ic_std if rank_ic_std > 0 else 0
    return {
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
        'valid_days': len(results),
        'elapsed_sec': elapsed,
    }


def _run_one_factor(analyzer, factor_file, factor_dir, output_dir, calc_kwargs):
    factor_rel = os.path.relpath(factor_file, factor_dir)
    factor_name = os.path.splitext(os.path.basename(factor_file))[0]
    t0 = time.perf_counter()
    analyzer.load_factor(factor_file) \
            .merge_data() \
            .calculate_rank_ic(**calc_kwargs)
    elapsed = time.perf_counter() - t0

    safe_prefix = os.path.dirname(factor_rel).replace(os.sep, '_')
    safe_prefix = (safe_prefix + '_') if safe_prefix else ''
    output_file = os.path.join(output_dir, f'{safe_prefix}{factor_name}_rankic.csv')
    analyzer.save_results(output_file)
    rec = _build_summary_record(analyzer, factor_rel, factor_name, elapsed)
    return rec, f"Done: {factor_rel} | mean_rankic: {rec['mean_rank_ic']:.6f} | std_rankic: {rec['rank_ic_std']:.6f} | sec: {elapsed:.2f}"


def _init_worker(data_pkl):
    global _WORKER_ANALYZER
    _WORKER_ANALYZER = FactorRankICAnalyzer(data_pkl).load_market_data()


def _run_one_factor_worker(task):
    factor_file, factor_dir, output_dir, calc_kwargs = task
    global _WORKER_ANALYZER
    if _WORKER_ANALYZER is None:
        _WORKER_ANALYZER = FactorRankICAnalyzer('data.pkl').load_market_data()
    factor_rel = os.path.relpath(factor_file, factor_dir)
    try:
        rec, log = _run_one_factor(_WORKER_ANALYZER, factor_file, factor_dir, output_dir, calc_kwargs)
        rec['error'] = None
        return rec, log
    except Exception as e:
        elapsed = np.nan
        return {
            'factor_path': factor_rel,
            'factor_name': os.path.splitext(os.path.basename(factor_file))[0],
            'error': str(e),
            'elapsed_sec': elapsed,
        }, f"Error analyzing {factor_rel}: {e}"


def batch_analyze_factors(
    factor_dir,
    output_dir='rankic_batch_results',
    data_pkl='data.pkl',
    factor_workers=1,
    **kwargs,
):
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

    factor_workers = int(max(1, factor_workers))
    calc_kwargs = dict(kwargs)
    if factor_workers > 1 and int(calc_kwargs.get('workers') or 0) > 1:
        print(
            "[warn] factor_workers>1 and workers>1 detected; "
            "this is two-level parallel and may oversubscribe CPU."
        )
    
    # 汇总结果
    summary_results = []

    if factor_workers == 1:
        # Single-process mode: reuse one analyzer instance for speed.
        analyzer = FactorRankICAnalyzer(data_pkl).load_market_data()
        iterator = factor_files
        if tqdm is not None:
            iterator = tqdm(factor_files, total=total, desc='Analyzing factors')
        for factor_file in iterator:
            factor_rel = os.path.relpath(factor_file, factor_dir)
            factor_name = os.path.splitext(os.path.basename(factor_file))[0]
            try:
                rec, log = _run_one_factor(analyzer, factor_file, factor_dir, output_dir, calc_kwargs)
                rec['error'] = None
                print(log)
                summary_results.append(rec)
            except Exception as e:
                print(f"Error analyzing {factor_rel}: {e}")
                summary_results.append({
                    'factor_path': factor_rel,
                    'factor_name': factor_name,
                    'error': str(e),
                    'elapsed_sec': np.nan,
                })
    else:
        max_workers = min(factor_workers, max(1, multiprocessing.cpu_count()))
        print(f"Factor-level parallel workers: {max_workers}")
        tasks = [(fp, factor_dir, output_dir, calc_kwargs) for fp in factor_files]
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(data_pkl,),
        ) as ex:
            futs = [ex.submit(_run_one_factor_worker, t) for t in tasks]
            iterator = as_completed(futs)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futs), desc='Analyzing factors(parallel)')
            for fut in iterator:
                rec, log = fut.result()
                print(log)
                summary_results.append(rec)
    
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
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    p = argparse.ArgumentParser(description='Batch Factor Rank IC Analysis')
    p.add_argument('factor_dir', nargs='?', default='factors_by_type', help='Factor directory (recursive *.pkl)')
    p.add_argument('--output-dir', type=str, default='rankic_batch_results', help='Output directory')
    p.add_argument('--data', type=str, default='data.pkl', help='Market data pkl path')
    p.add_argument('--factor-workers', type=int, default=1, help='Parallel workers across factors (level-1)')
    p.add_argument('--ic-workers', type=int, default=None, help='Parallel workers per factor/day IC calc (level-2)')
    p.add_argument('--backend', type=str, default='pandas', choices=['pandas', 'torch'], help='IC backend')
    p.add_argument('--device', type=str, default='auto', help='Torch device when --backend torch')
    p.add_argument('--no-filter-st', action='store_true', help='Do not exclude ST stocks')
    p.add_argument('--no-filter-limit-up', action='store_true', help='Do not exclude limit-up stocks')
    p.add_argument('--no-filter-limit-down', action='store_true', help='Do not exclude limit-down stocks')
    p.add_argument('--no-filter-suspended', action='store_true', help='Do not exclude suspended stocks')
    p.add_argument('--no-next-day', action='store_true', help='Use same-day return instead of next-day return')
    args = p.parse_args()

    config = {
        'exclude_st': not args.no_filter_st,
        'exclude_limit_up': not args.no_filter_limit_up,
        'exclude_limit_down': not args.no_filter_limit_down,
        'exclude_suspended': not args.no_filter_suspended,
        'use_next_day_return': not args.no_next_day,
        'workers': args.ic_workers,
        'backend': args.backend,
        'device': args.device,
    }

    print("="*80)
    print("Batch Factor Rank IC Analysis")
    print("="*80)
    print(f"Factor Directory: {args.factor_dir}")
    print(f"Configuration: {config}")
    print("="*80)
    
    summary = batch_analyze_factors(
        args.factor_dir,
        output_dir=args.output_dir,
        data_pkl=args.data,
        factor_workers=args.factor_workers,
        **config,
    )

