#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量分析多个因子的Rank IC
"""

import logging
import os
import glob
import time
import argparse
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from factor_rankic_analysis import FactorRankICAnalyzer
import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

log = logging.getLogger(__name__)

_WORKER_ANALYZER = None
_WORKER_DATA_PKL = None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _write_metadata(path, payload):
    p = os.path.abspath(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
    return p


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


def _output_exists(factor_file: str, factor_dir: str, output_dir: str) -> bool:
    factor_rel = os.path.relpath(factor_file, factor_dir)
    factor_name = os.path.splitext(os.path.basename(factor_file))[0]
    safe_prefix = os.path.dirname(factor_rel).replace(os.sep, '_')
    safe_prefix = (safe_prefix + '_') if safe_prefix else ''
    output_file = os.path.join(output_dir, f'{safe_prefix}{factor_name}_rankic.csv')
    return os.path.isfile(output_file)


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
    global _WORKER_ANALYZER, _WORKER_DATA_PKL
    _WORKER_DATA_PKL = data_pkl
    _WORKER_ANALYZER = FactorRankICAnalyzer(data_pkl).load_market_data()


def _run_one_factor_worker(task):
    factor_file, factor_dir, output_dir, calc_kwargs = task
    global _WORKER_ANALYZER, _WORKER_DATA_PKL
    if _WORKER_ANALYZER is None:
        raise RuntimeError(
            f"Worker analyzer not initialized. data_pkl={_WORKER_DATA_PKL!r}. "
            "This means _init_worker failed — check that data_pkl path is accessible from worker processes."
        )
    factor_rel = os.path.relpath(factor_file, factor_dir)
    try:
        rec, msg = _run_one_factor(_WORKER_ANALYZER, factor_file, factor_dir, output_dir, calc_kwargs)
        rec['error'] = None
        return rec, msg
    except Exception as e:
        log.error("Error analyzing %s: %s", factor_rel, e)
        return {
            'factor_path': factor_rel,
            'factor_name': os.path.splitext(os.path.basename(factor_file))[0],
            'error': str(e),
            'elapsed_sec': np.nan,
        }, f"Error analyzing {factor_rel}: {e}"


def batch_analyze_factors(
    factor_dir,
    output_dir='rankic_batch_results',
    data_pkl='data.pkl',
    factor_workers=1,
    metadata_out=None,
    skip_existing=False,
    **kwargs,
):
    """
    批量分析因子
    
    参数:
        factor_dir: 因子目录路径
        output_dir: 输出目录
        **kwargs: 传递给calculate_rank_ic的其他参数
    """
    
    t0 = time.perf_counter()
    ts0 = pd.Timestamp.now(tz='UTC')
    run_id = f"batch_rankic_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir_abs = os.path.abspath(output_dir)
    if metadata_out is None:
        metadata_out = os.path.join(output_dir_abs, "run_metadata.json")
    else:
        metadata_out = os.path.abspath(metadata_out)

    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        factor_files = sorted(glob.glob(os.path.join(factor_dir, '**', '*.pkl'), recursive=True))
        total = len(factor_files)
        log.info("Found %d factor files in %s", total, factor_dir)

        if skip_existing:
            before = len(factor_files)
            factor_files = [
                fp for fp in factor_files
                if not _output_exists(fp, factor_dir, output_dir)
            ]
            skipped = before - len(factor_files)
            log.info("Skipping %d already-completed factors (%d remaining)", skipped, len(factor_files))

        if not factor_files:
            log.info("Nothing to do — all factors already have output CSVs.")
            return pd.DataFrame()

        factor_workers = int(max(1, factor_workers))
        calc_kwargs = dict(kwargs)
        if factor_workers > 1 and int(calc_kwargs.get('workers') or 0) > 1:
            log.warning("factor_workers>1 and workers>1 detected; "
                        "two-level parallelism may oversubscribe CPU.")
        
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
                    rec, msg = _run_one_factor(analyzer, factor_file, factor_dir, output_dir, calc_kwargs)
                    rec['error'] = None
                    log.info(msg)
                    summary_results.append(rec)
                except Exception as e:
                    log.error("Error analyzing %s: %s", factor_rel, e)
                    summary_results.append({
                        'factor_path': factor_rel,
                        'factor_name': factor_name,
                        'error': str(e),
                        'elapsed_sec': np.nan,
                    })
        else:
            max_workers = min(factor_workers, max(1, multiprocessing.cpu_count()))
            log.info("Factor-level parallel workers: %d", max_workers)
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
                    rec, msg = fut.result()
                    log.info(msg)
                    summary_results.append(rec)
        
        # 保存汇总结果
        summary_df = pd.DataFrame(summary_results)
        summary_file = os.path.join(output_dir, 'summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        log.info("Batch analysis completed. Results: %s  Summary: %s", output_dir, summary_file)

        if 'rank_ic_ir' in summary_df.columns:
            top_factors = summary_df.nlargest(10, 'rank_ic_ir')
            log.info("Top 10 factors by Rank IC IR:\n%s",
                     top_factors[['factor_name', 'mean_rank_ic', 'rank_ic_ir', 'rank_ic_win_rate']].to_string(index=False))

        success_count = int((summary_df.get('error').isna()).sum()) if 'error' in summary_df.columns else int(len(summary_df))
        error_count = int(len(summary_df) - success_count)
        elapsed_arr = pd.to_numeric(summary_df.get('elapsed_sec'), errors='coerce') if 'elapsed_sec' in summary_df.columns else pd.Series(dtype=float)
        top_factor = None
        if 'rank_ic_ir' in summary_df.columns and len(summary_df) > 0:
            tmp = summary_df.copy()
            tmp['rank_ic_ir'] = pd.to_numeric(tmp['rank_ic_ir'], errors='coerce')
            tmp = tmp.sort_values('rank_ic_ir', ascending=False, na_position='last')
            if len(tmp) > 0:
                top_factor = {
                    "factor_name": str(tmp.iloc[0].get('factor_name')),
                    "rank_ic_ir": float(tmp.iloc[0].get('rank_ic_ir')) if pd.notna(tmp.iloc[0].get('rank_ic_ir')) else None,
                }

        ts1 = pd.Timestamp.now(tz='UTC')
        metadata = {
            "run_id": run_id,
            "status": "success",
            "script": "scripts/analysis/batch_factor_analysis.py",
            "started_at_utc": ts0.isoformat(),
            "ended_at_utc": ts1.isoformat(),
            "runtime_sec": float(time.perf_counter() - t0),
            "inputs": {
                "factor_dir": os.path.abspath(factor_dir),
                "output_dir": output_dir_abs,
                "data_pkl": os.path.abspath(data_pkl),
                "factor_workers": int(factor_workers),
                "ic_calc_kwargs": calc_kwargs,
            },
            "summary": {
                "total_factors": int(total),
                "success_count": success_count,
                "error_count": error_count,
                "mean_elapsed_sec": float(elapsed_arr.mean()) if len(elapsed_arr.dropna()) > 0 else None,
                "top_factor": top_factor,
            },
            "artifacts": {
                "summary_csv": os.path.abspath(summary_file),
            },
        }
        meta_path = _write_metadata(metadata_out, metadata)
        log.info("Run metadata saved to: %s", meta_path)
        return summary_df

    except Exception as e:
        ts1 = pd.Timestamp.now(tz='UTC')
        metadata = {
            "run_id": run_id,
            "status": "failed",
            "script": "scripts/analysis/batch_factor_analysis.py",
            "started_at_utc": ts0.isoformat(),
            "ended_at_utc": ts1.isoformat(),
            "runtime_sec": float(time.perf_counter() - t0),
            "inputs": {
                "factor_dir": os.path.abspath(factor_dir),
                "output_dir": output_dir_abs,
                "data_pkl": os.path.abspath(data_pkl),
                "factor_workers": int(max(1, factor_workers)),
                "ic_calc_kwargs": dict(kwargs),
            },
            "error": str(e),
        }
        meta_path = _write_metadata(metadata_out, metadata)
        log.error("Run failed. Metadata saved to: %s", meta_path)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
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
    p.add_argument('--metadata-out', type=str, default='', help='Optional metadata json output path')
    p.add_argument('--skip-existing', action='store_true', help='Skip factors whose output CSV already exists (resume interrupted runs)')
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

    log.info("Batch Factor Rank IC Analysis | dir=%s config=%s", args.factor_dir, config)

    summary = batch_analyze_factors(
        args.factor_dir,
        output_dir=args.output_dir,
        data_pkl=args.data,
        factor_workers=args.factor_workers,
        metadata_out=args.metadata_out or None,
        skip_existing=args.skip_existing,
        **config,
    )

