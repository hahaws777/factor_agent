#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Rank IC analysis tool.
Supports filtering ST stocks, limit-up/down stocks, and suspended stocks.
"""

import logging
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

torch = None

log = logging.getLogger(__name__)

MIN_STOCKS_PER_DAY = 50        # days with fewer stocks are dropped from IC
IC_VALID_BOUND = 1 - 1e-6      # |IC| must be below this to compute p-value
MIN_MERGE_OVERLAP = 0.05       # warn if merged rows < this fraction of factor rows
MARKET_REQUIRED_COLUMNS = [
    'order_book_id',
    'date',
    'close',
    'ST',
    'limit_up_flag',
    'limit_down_flag',
    'suspended',
]

def _parallel_ic_calc(args):
    d, g, factor_name = args
    n = len(g)
    if n < 3:
        return (d, n, np.nan, np.nan)
    ic_val = g[factor_name].corr(g['return'])
    g2 = g.copy()
    g2['f_rank'] = g2[factor_name].rank(method='average')
    g2['r_rank'] = g2['return'].rank(method='average')
    rank_ic_val = g2['f_rank'].corr(g2['r_rank'])
    return (d, n, ic_val, rank_ic_val)


def _corr_p_values(r_values, n_values) -> np.ndarray:
    """Two-sided p-values for correlations; scipy if present, normal approximation otherwise."""
    r = np.asarray(r_values, dtype=float)
    n = np.asarray(n_values, dtype=float)
    p_val = np.full(len(r), np.nan, dtype=float)
    valid = np.isfinite(r) & (np.abs(r) < IC_VALID_BOUND) & (n > 2)
    if not valid.any():
        return p_val
    df = n[valid] - 2
    rv = r[valid]
    t_stat = np.abs(rv) * np.sqrt(df / (1 - rv * rv))
    if student_t is not None:
        p_val[valid] = 2 * student_t.sf(t_stat, df)
    else:
        # The cross-section is large in this workflow, so this is a close fallback.
        p_val[valid] = np.array([math.erfc(float(t) / math.sqrt(2.0)) for t in t_stat], dtype=float)
    return p_val

def _parallel_decile_calc(args):
    d, g, factor_name, n_bins = args
    g = g.dropna(subset=[factor_name, 'return'])
    if g[factor_name].notna().sum() < n_bins:
        return (d, None)
    try:
        bins = pd.qcut(g[factor_name], q=n_bins, labels=False, duplicates='drop')
    except Exception:
        return (d, None)
    g = g.assign(bin=bins.astype(int))
    means = g.groupby('bin')['return'].mean()
    return (d, means.to_dict())


def _torch_corr(x: "torch.Tensor", y: "torch.Tensor") -> float:
    """Compute Pearson correlation for two 1D tensors."""
    x = x.float()
    y = y.float()
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    den = torch.sqrt(torch.sum(xv * xv) * torch.sum(yv * yv))
    if torch.isclose(den, torch.tensor(0.0, device=x.device)):
        return np.nan
    return float((torch.sum(xv * yv) / den).item())


def _torch_rank_average(x: "torch.Tensor") -> "torch.Tensor":
    """Average rank for ties — matches pandas rank(method='average')."""
    n = x.numel()
    order = torch.argsort(x, stable=True)
    sorted_x = x[order]
    new_group = torch.ones(n, dtype=torch.bool, device=x.device)
    new_group[1:] = sorted_x[1:] != sorted_x[:-1]
    group_id = torch.cumsum(new_group, dim=0) - 1
    sorted_rank = torch.arange(1, n + 1, device=x.device, dtype=torch.float32)
    group_rank_sum = torch.zeros(n, dtype=torch.float32, device=x.device)
    group_count = torch.zeros(n, dtype=torch.float32, device=x.device)
    group_rank_sum.scatter_add_(0, group_id, sorted_rank)
    group_count.scatter_add_(0, group_id, torch.ones(n, device=x.device))
    avg_rank = group_rank_sum / group_count.clamp(min=1)
    result = torch.empty(n, dtype=torch.float32, device=x.device)
    result[order] = avg_rank[group_id]
    return result


def _resolve_torch_device(device: str) -> str:
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except Exception as e:
            raise ImportError("PyTorch is not installed. Please install torch first.") from e
    if torch is None:
        raise ImportError("PyTorch is not installed. Please install torch first.")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def _torch_calc_rows_by_segments(
    data: pd.DataFrame,
    factor_name: str,
    resolved_device: str,
):
    """
    Torch path: move full columns to device once, then compute per-day slices.
    This avoids repeated CPU->GPU transfer inside day loops.
    """
    if data.empty:
        return []

    work = data[['date', factor_name, 'return']].sort_values('date')
    date_arr = work['date'].to_numpy()
    f_arr = work[factor_name].to_numpy(dtype=np.float32, copy=False)
    r_arr = work['return'].to_numpy(dtype=np.float32, copy=False)

    if len(work) == 0:
        return []

    boundaries = np.flatnonzero(date_arr[1:] != date_arr[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(work)]
    day_vals = date_arr[starts]

    # One-time transfer
    tf = torch.as_tensor(f_arr, device=resolved_device)
    tr = torch.as_tensor(r_arr, device=resolved_device)

    rows = []
    with torch.no_grad():
        for d, s, e in zip(day_vals, starts, ends):
            n = int(e - s)
            if n < 3:
                rows.append((d, n, np.nan, np.nan))
                continue
            x = tf[s:e]
            y = tr[s:e]
            ic_val = _torch_corr(x, y)
            rx = _torch_rank_average(x)
            ry = _torch_rank_average(y)
            rank_ic_val = _torch_corr(rx, ry)
            rows.append((d, n, ic_val, rank_ic_val))
    return rows

class FactorRankICAnalyzer:
    """Factor Rank IC analyzer."""

    def __init__(self, data_pkl='data.pkl'):
        """
        Parameters
        ----------
        data_pkl : path to market data pickle file
        """
        self.data_pkl = data_pkl
        self.market_data = None
        self.market_data_indexed = None
        self.factor_data = None
        self.merged_data = None
        self.ic_results = None
        self.backtest_daily_returns = None
        self.backtest_cum_returns = None

    def _prepare_market_index(self, copy_data=True, keep_required_only=True):
        """Prepare cached MultiIndex market table for faster joins.

        Rank IC and decile backtests only need close plus filter flags. Keeping
        the full market panel here creates a very large join/reset_index copy.
        """
        if self.market_data is None:
            raise ValueError("Market data is not loaded.")
        md = self.market_data
        if keep_required_only:
            keep = [c for c in MARKET_REQUIRED_COLUMNS if c in md.columns]
            missing = [c for c in ('order_book_id', 'date', 'close') if c not in keep]
            if missing:
                raise ValueError(f"Market data missing required columns: {missing}")
            md = md[keep]
        if copy_data:
            md = md.copy()
        md['date'] = pd.to_datetime(md['date']).dt.normalize()
        md['order_book_id'] = md['order_book_id'].astype(str)
        for col in ('ST', 'suspended'):
            if col in md.columns:
                md[col] = md[col].fillna(False).astype(bool)
        for col in ('limit_up_flag', 'limit_down_flag'):
            if col in md.columns:
                # Treat NaN as not limited, matching the existing `!= True` semantics.
                md[col] = md[col].eq(True)
        self.market_data = md
        self.market_data_indexed = md.set_index(['order_book_id', 'date']).sort_index()
        return self

    def set_market_data(self, market_data: pd.DataFrame, market_data_indexed: pd.DataFrame = None):
        """
        Set preloaded market data (for batch mode reuse).
        """
        self.market_data = market_data
        if market_data_indexed is not None:
            self.market_data_indexed = market_data_indexed
        else:
            self._prepare_market_index(copy_data=False)
        return self
        
    def load_market_data(self):
        """Load market data from pickle."""
        if self.market_data is not None and self.market_data_indexed is not None:
            log.info("Market data already loaded in memory; skip reloading.")
            return self

        log.info("Loading market data from %s...", self.data_pkl)
        if not os.path.isfile(self.data_pkl):
            raise FileNotFoundError(f"Market data file not found: {self.data_pkl}")

        self.market_data = pd.read_pickle(self.data_pkl)
        self._prepare_market_index()

        log.info("Market data loaded: shape=%s  date range: %s to %s  columns=%s",
                 self.market_data.shape,
                 self.market_data['date'].min(),
                 self.market_data['date'].max(),
                 list(self.market_data.columns))
        return self
    
    def load_factor(self, factor_file):
        """Load factor from a .pkl file."""
        log.info("Loading factor from %s...", factor_file)
        if not os.path.isfile(factor_file):
            raise FileNotFoundError(f"Factor file not found: {factor_file}")

        self.factor_data = pd.read_pickle(factor_file)
        self.factor_name = self.factor_data.columns[0]

        log.info("Factor loaded: shape=%s  name=%r  non-null=%.2f%%",
                 self.factor_data.shape,
                 self.factor_name,
                 self.factor_data[self.factor_name].notna().mean() * 100)
        return self
    
    def merge_data(self):
        """Merge market data with factor data."""
        log.info("Merging market and factor data...")

        if self.market_data is None:
            raise ValueError("Please load market data first.")
        if self.market_data_indexed is None:
            self._prepare_market_index()

        if not isinstance(self.factor_data.index, pd.MultiIndex):
            raise ValueError("Factor data must have MultiIndex (order_book_id, date)")

        factor_indexed = self.factor_data[[self.factor_name]].dropna()
        if factor_indexed.index.names != ['order_book_id', 'date']:
            factor_indexed = factor_indexed.copy()
            factor_indexed.index = factor_indexed.index.set_names(['order_book_id', 'date'])
        if not pd.api.types.is_datetime64_any_dtype(factor_indexed.index.get_level_values('date')):
            factor_df = factor_indexed.reset_index()
            factor_df['order_book_id'] = factor_df['order_book_id'].astype(str)
            factor_df['date'] = pd.to_datetime(factor_df['date']).dt.normalize()
            factor_indexed = factor_df.set_index(['order_book_id', 'date'])[[self.factor_name]]

        self.merged_data = self.market_data_indexed.join(factor_indexed, how='inner').reset_index()
        self.merged_data['order_book_id'] = self.merged_data['order_book_id'].astype('category')

        overlap = len(self.merged_data) / max(len(factor_indexed), 1)
        if overlap < MIN_MERGE_OVERLAP:
            log.warning("Low merge overlap: %.1f%% of factor rows matched market data. "
                        "Check date formats and stock ID conventions.", overlap * 100)

        mem_mb = self.merged_data.memory_usage(deep=True).sum() / 1024 ** 2
        log.info("Merged data shape: %s  memory=%.1f MB  columns=%s",
                 self.merged_data.shape, mem_mb, list(self.merged_data.columns))
        return self
    
    def calculate_rank_ic(self, 
                         exclude_st=True, 
                         exclude_limit_up=True, 
                         exclude_limit_down=True,
                         exclude_suspended=True,
                         use_next_day_return=True,
                         workers=None,
                         backend='pandas',
                         device='auto'):
        """
        Compute Rank IC (and Pearson IC) for the merged factor / return data.

        Parameters
        ----------
        exclude_st            : drop ST-flagged stocks from the signal day
        exclude_limit_up      : drop limit-up stocks from the signal day (and return day when next_day=True)
        exclude_limit_down    : drop limit-down stocks from the signal day (and return day when next_day=True)
        exclude_suspended     : drop suspended stocks
        use_next_day_return   : use T+1 return instead of same-day return

        Returns
        -------
        pd.DataFrame with per-day IC, rank_ic, p_value, n_stocks
        """
        log.info("Calculating Rank IC | exclude_st=%s limit_up=%s limit_down=%s "
                 "suspended=%s next_day=%s backend=%s%s",
                 exclude_st, exclude_limit_up, exclude_limit_down,
                 exclude_suspended, use_next_day_return, backend,
                 f" device={device}" if backend == 'torch' else "")
        
        data = self.merged_data.sort_values(['order_book_id', 'date']).copy()
        if use_next_day_return:
            by_stock = data.groupby('order_book_id', observed=True)
            data['return'] = by_stock['close'].shift(-1) / data['close'] - 1.0
            # Pre-compute next-day limit flags before any row drops
            if exclude_limit_up or exclude_limit_down:
                data['_next_luf'] = by_stock['limit_up_flag'].shift(-1)
                data['_next_ldf'] = by_stock['limit_down_flag'].shift(-1)
        else:
            data['return'] = data.groupby('order_book_id', observed=True)['close'].pct_change(1, fill_method=None)

        # Filter: ST and suspended are proper bool columns — astype(bool) is safe.
        # limit_up_flag / limit_down_flag are object dtype with possible NaN values;
        # NaN.astype(bool) == True, which would wrongly exclude non-limit rows.
        # Use == True to treat NaN as "not limit-up".
        if exclude_st:
            data = data[~data['ST'].astype(bool)]

        if exclude_limit_up:
            data = data[data['limit_up_flag'] != True]          # signal-day limit-up

        if exclude_limit_down:
            data = data[data['limit_down_flag'] != True]        # signal-day limit-down

        if exclude_suspended:
            data = data[~data['suspended'].astype(bool)]

        # Also exclude rows where the return is measured on a limit-up/down day:
        # that return is capped and not a clean alpha signal.
        if use_next_day_return:
            if exclude_limit_up and '_next_luf' in data.columns:
                data = data[data['_next_luf'] != True]
            if exclude_limit_down and '_next_ldf' in data.columns:
                data = data[data['_next_ldf'] != True]

        data = data.dropna(subset=[self.factor_name, 'return'])
        data = data[['date', self.factor_name, 'return']]
        
        log.info("Valid records after filtering: %d", len(data))

        num_days = data['date'].nunique()
        log.info("Processing %d trading days...", num_days)

        if backend == 'torch':
            resolved_device = _resolve_torch_device(device)
            log.info("Using torch backend on device: %s", resolved_device)
            rows = _torch_calc_rows_by_segments(data, self.factor_name, resolved_device)
        else:
            # Avoid materialising all per-day group DataFrames at once. On a
            # 20M-row panel, list(groupby(...)) can add several GB of peak RAM.
            tasks = (
                (d, g[[self.factor_name, 'return']].copy(), self.factor_name)
                for d, g in data.groupby('date', sort=True)
            )
            if workers is None or workers <= 1:
                rows = [_parallel_ic_calc(it) for it in tasks]
            else:
                max_workers = min(int(workers), max(1, min(multiprocessing.cpu_count(), 16)))
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    rows = list(ex.map(_parallel_ic_calc, tasks, chunksize=8))

        result = pd.DataFrame(rows, columns=['date', 'n_stocks', 'ic', 'rank_ic'])

        result = result[result['n_stocks'] >= MIN_STOCKS_PER_DAY].copy()

        # Drop degenerate days (n<3 or fully constant factor/return) rather than
        # filling NaN with 0, which would bias mean IC down and distort IR.
        result = result[result['rank_ic'].notna() & result['ic'].notna()].copy()

        result['p_value'] = _corr_p_values(result['rank_ic'].values, result['n_stocks'].values)
        result['ic_p_value'] = _corr_p_values(result['ic'].values, result['n_stocks'].values)

        result['date'] = pd.to_datetime(result['date'])
        self.ic_results = result[['date', 'ic', 'rank_ic', 'p_value', 'ic_p_value', 'n_stocks']].sort_values('date')
        
        log.info("Calculated IC for %d valid days", len(self.ic_results))
        
        return self
    
    def print_statistics(self):
        """Print IC / Rank IC summary statistics."""
        if self.ic_results is None or len(self.ic_results) == 0:
            print("No IC results to display")
            return
        
        print("\n" + "="*80)
        print("Factor Rank IC Statistics")
        print("="*80)
        print(f"Factor Name: {self.factor_name}")
        print(f"Total Valid Days: {len(self.ic_results)}")
        
        # IC stats
        ic_series = self.ic_results['ic']
        print(f"\nIC Statistics:")
        print(f"  Mean IC:       {ic_series.mean():.6f}")
        print(f"  Std IC:        {ic_series.std():.6f}")
        print(f"  Min IC:        {ic_series.min():.6f}")
        print(f"  Max IC:        {ic_series.max():.6f}")
        
        if ic_series.std() > 0:
            print(f"  IC IR:         {ic_series.mean() / ic_series.std():.6f}")
        
        # IC win rate
        print(f"  IC Win Rate:   {(ic_series > 0).mean()*100:.2f}%")
        print(f"  IC > 0.05:     {(ic_series > 0.05).mean()*100:.2f}%")
        print(f"  IC < -0.05:    {(ic_series < -0.05).mean()*100:.2f}%")
        
        # Rank IC stats
        rank_ic_series = self.ic_results['rank_ic']
        print(f"\nRank IC Statistics:")
        print(f"  Mean Rank IC:  {rank_ic_series.mean():.6f}")
        print(f"  Std Rank IC:   {rank_ic_series.std():.6f}")
        print(f"  Min Rank IC:   {rank_ic_series.min():.6f}")
        print(f"  Max Rank IC:   {rank_ic_series.max():.6f}")
        
        if rank_ic_series.std() > 0:
            print(f"  Rank IC IR:    {rank_ic_series.mean() / rank_ic_series.std():.6f}")
        
        # Rank IC win rate
        print(f"  Rank IC Win Rate:  {(rank_ic_series > 0).mean()*100:.2f}%")
        print(f"  Rank IC > 0.05:    {(rank_ic_series > 0.05).mean()*100:.2f}%")
        print(f"  Rank IC < -0.05:   {(rank_ic_series < -0.05).mean()*100:.2f}%")
        
        # Average stock count
        print(f"\nAverage stocks per day: {self.ic_results['n_stocks'].mean():.0f}")
        
        print("="*80)
        
        return self
    
    def save_results(self, output_file=None):
        """
        Save IC analysis results to CSV.

        Parameters
        ----------
        output_file : output path (auto-named if None)
        """
        if self.ic_results is None or len(self.ic_results) == 0:
            print("No results to save")
            return self

        if output_file is None:
            # Auto-generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'rankic_results_{self.factor_name}_{timestamp}.csv'
        
        self.ic_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_file}")
        
        return self

    def backtest_deciles(self,
                         n_bins=10,
                         exclude_st=True,
                         exclude_limit_up=True,
                         exclude_limit_down=True,
                         exclude_suspended=True,
                         use_next_day_return=True,
                         workers=None,
                         transaction_cost_bps=0):
        """
        Factor decile backtest (default 10 bins). Computes equal-weight group
        daily returns and long-short return (top_bin - bottom_bin).

        Parameters
        ----------
        n_bins              : number of quantile buckets (default 10)
        exclude_st / limit_up / limit_down / suspended : filter flags
        use_next_day_return : use T+1 return for realistic trade execution
        transaction_cost_bps: one-way cost in basis points. Applied as 2× every day
                              (i.e. assumes 100% daily turnover) — this is a conservative
                              upper bound. Real turnover is lower; use as a stress test.

        Results are stored in:
            self.backtest_daily_returns : DataFrame[date, Q1..Qn, LS]
            self.backtest_cum_returns   : DataFrame[date, Q1..Qn, LS]
        """
        if self.merged_data is None:
            raise ValueError("Please call merge_data() before backtesting.")

        data = self.merged_data.copy()

        data = data.sort_values(['order_book_id', 'date'])
        if use_next_day_return:
            by_stock = data.groupby('order_book_id', observed=True)
            data['return'] = by_stock['close'].shift(-1) / data['close'] - 1.0
            if exclude_limit_up or exclude_limit_down:
                data['_next_luf'] = by_stock['limit_up_flag'].shift(-1)
                data['_next_ldf'] = by_stock['limit_down_flag'].shift(-1)
        else:
            data['return'] = data.groupby('order_book_id', observed=True)['close'].pct_change(1, fill_method=None)

        if exclude_st:
            data = data[~data['ST'].astype(bool)]
        if exclude_limit_up:
            data = data[data['limit_up_flag'] != True]
        if exclude_limit_down:
            data = data[data['limit_down_flag'] != True]
        if exclude_suspended:
            data = data[~data['suspended'].astype(bool)]

        if use_next_day_return:
            if exclude_limit_up and '_next_luf' in data.columns:
                data = data[data['_next_luf'] != True]
            if exclude_limit_down and '_next_ldf' in data.columns:
                data = data[data['_next_ldf'] != True]

        data = data.dropna(subset=[self.factor_name, 'return'])
        if data.empty:
            raise ValueError("No valid data for backtest after filtering.")

        # Quantile-bin within each date. Keep this lazy to avoid materialising
        # thousands of copied daily frames before workers consume them.
        tasks = (
            (d, g[['order_book_id', self.factor_name, 'return']].copy(), self.factor_name, n_bins)
            for d, g in data.groupby('date', sort=True)
        )

        if workers is None or workers <= 1:
            rows = [_parallel_decile_calc(it) for it in tasks]
        else:
            max_workers = min(int(workers), max(1, min(multiprocessing.cpu_count(), 16)))
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                rows = list(ex.map(_parallel_decile_calc, tasks, chunksize=8))

        # Build DataFrame from per-day results
        records = []
        for d, means in rows:
            if means is None or len(means) == 0:
                continue
            rec = {'date': d}
            for k, v in means.items():
                rec[int(k)] = v
            records.append(rec)
        if not records:
            raise ValueError("All dates failed to form quantile bins; check factor distribution.")

        ret_tbl = pd.DataFrame(records).set_index('date').sort_index()

        # Rename bins to Q1..Qn (Q1 = bottom quantile, Qn = top quantile)
        col_map = {}
        for i, c in enumerate(ret_tbl.columns):
            col_map[c] = f"Q{i+1}"
        ret_tbl = ret_tbl.rename(columns=col_map)

        # Long-short: top quantile minus bottom quantile, net of round-trip cost.
        # Fixed-cost mode: assumes 100% daily turnover (conservative upper bound).
        # transaction_cost_bps is one-way; *2 for a full round-trip (long top + short bottom).
        q_low = f"Q1"
        q_high = f"Q{min(n_bins, len(ret_tbl.columns))}"
        daily_cost = 2.0 * transaction_cost_bps / 10_000
        if q_low in ret_tbl.columns and q_high in ret_tbl.columns:
            ret_tbl['LS'] = ret_tbl[q_high] - ret_tbl[q_low] - daily_cost
        else:
            ret_tbl['LS'] = np.nan

        # Cumulative compound return
        cum_tbl = (1.0 + ret_tbl.fillna(0)).cumprod() - 1.0

        ret_tbl = ret_tbl.reset_index()
        cum_tbl = cum_tbl.reset_index()
        ret_tbl['date'] = pd.to_datetime(ret_tbl['date'])
        cum_tbl['date'] = pd.to_datetime(cum_tbl['date'])
        self.backtest_daily_returns = ret_tbl
        self.backtest_cum_returns = cum_tbl

        print("\nDecile backtest completed:")
        print(f"  Trading days: {ret_tbl['date'].nunique()}")
        print(f"  Columns: {', '.join([c for c in ret_tbl.columns if c != 'date'])}")

        return self

    def save_backtest(self, output_dir=None, prefix=None):
        """
        Save decile daily returns and cumulative return curves to CSV.

        Parameters
        ----------
        output_dir : output directory (default: backtest_results/)
        prefix     : filename prefix (default: factor name)
        """
        if self.backtest_daily_returns is None or self.backtest_cum_returns is None:
            print("No backtest results to save")
            return self

        if output_dir is None:
            output_dir = 'backtest_results'
        os.makedirs(output_dir, exist_ok=True)

        if prefix is None:
            prefix = self.factor_name

        daily_path = os.path.join(output_dir, f"{prefix}_decile_daily_returns.csv")
        cum_path = os.path.join(output_dir, f"{prefix}_decile_cum_returns.csv")

        self.backtest_daily_returns.to_csv(daily_path, index=False, encoding='utf-8-sig')
        self.backtest_cum_returns.to_csv(cum_path, index=False, encoding='utf-8-sig')

        print(f"Saved daily returns to: {daily_path}")
        print(f"Saved cumulative returns to: {cum_path}")

        return self

    def plot_backtest_cumulative(self, from_file=None, output_dir=None, prefix=None):
        """
        Plot cumulative return curves for Q1..Qn and the LS spread.

        Parameters
        ----------
        from_file  : optional CSV path to read cumulative returns from directly
        output_dir : output directory (default: backtest_plots/)
        prefix     : filename prefix (default: factor name)
        """
        if from_file is not None:
            cum_df = pd.read_csv(from_file)
        else:
            if self.backtest_cum_returns is None:
                print("No cumulative backtest data. Run backtest_deciles() or provide from_file.")
                return self
            cum_df = self.backtest_cum_returns.copy()

        if 'date' in cum_df.columns:
            cum_df['date'] = pd.to_datetime(cum_df['date'])
            cum_df = cum_df.sort_values('date')

        if output_dir is None:
            output_dir = 'backtest_plots'
        os.makedirs(output_dir, exist_ok=True)
        if plt is None:
            print("matplotlib is not installed; skipping backtest plots.")
            return self

        if prefix is None:
            prefix = getattr(self, 'factor_name', 'factor')

        # Identify quantile columns and LS
        cols = [c for c in cum_df.columns if c != 'date']
        ls_col = 'LS' if 'LS' in cols else None
        quant_cols = [c for c in cols if c != 'LS']
        quant_cols_sorted = sorted(quant_cols, key=lambda x: int(x[1:]) if x.startswith('Q') and x[1:].isdigit() else 0)

        # Plot all quantiles + LS
        plt.figure(figsize=(12, 6))
        for c in quant_cols_sorted:
            plt.plot(cum_df['date'], cum_df[c], color='#999999', linewidth=1.0, alpha=0.8, label=c if c in ['Q1','Q10'] else None)
        if 'Q1' in quant_cols_sorted:
            plt.plot(cum_df['date'], cum_df['Q1'], color='#d62728', linewidth=1.8, label='Q1')
        if f"Q{len(quant_cols_sorted)}" in quant_cols_sorted:
            qh = f"Q{len(quant_cols_sorted)}"
            plt.plot(cum_df['date'], cum_df[qh], color='#1f77b4', linewidth=1.8, label=qh)
        if ls_col is not None and ls_col in cum_df.columns:
            plt.plot(cum_df['date'], cum_df[ls_col], color='#2ca02c', linewidth=2.0, label='LS')
        plt.title('Cumulative Return (Decile Portfolios)', fontsize=13)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_all = os.path.join(output_dir, f"{prefix}_decile_cum_all.png")
        plt.tight_layout()
        plt.savefig(out_all, dpi=150)
        plt.close()

        # Separate LS plot (if available)
        if ls_col is not None and ls_col in cum_df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(cum_df['date'], cum_df[ls_col], color='#2ca02c', linewidth=2.0, label='LS')
            plt.title('Cumulative Return (Long-Short)', fontsize=13)
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            out_ls = os.path.join(output_dir, f"{prefix}_decile_cum_LS.png")
            plt.tight_layout()
            plt.savefig(out_ls, dpi=150)
            plt.close()

        print(f"Saved plots to: {output_dir}")
        return self


def main():
    import argparse
    import os
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    os.chdir(_root)

    parser = argparse.ArgumentParser(description='Factor Rank IC Analysis Tool')
    parser.add_argument('--factor', type=str, required=True,
                       help='Factor file path (e.g., factors_by_type/alpha101/WorldQuant_alpha014.pkl)')
    parser.add_argument('--data', type=str, default='data.pkl',
                       help='Market data file path (default: data.pkl)')
    parser.add_argument('--no-filter-st', action='store_true',
                       help='Do not exclude ST stocks')
    parser.add_argument('--no-filter-limit-up', action='store_true',
                       help='Do not exclude limit-up stocks')
    parser.add_argument('--no-filter-limit-down', action='store_true',
                       help='Do not exclude limit-down stocks')
    parser.add_argument('--no-filter-suspended', action='store_true',
                       help='Do not exclude suspended stocks')
    parser.add_argument('--same-day', action='store_true',
                       help='Use same-day return instead of next-day forward return (default: next-day)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers for IC/backtest')
    parser.add_argument('--backend', type=str, default='pandas', choices=['pandas', 'torch'],
                       help='Compute backend for IC: pandas (default) or torch')
    parser.add_argument('--device', type=str, default='auto',
                       help='Torch device when --backend torch (e.g. auto/cuda/cuda:0/cpu)')
    parser.add_argument('--backtest-decile', action='store_true',
                       help='Run decile portfolio backtest with long-short output')
    parser.add_argument('--no-filter-st-bt', action='store_true',
                       help='(Backtest) Do not exclude ST stocks')
    parser.add_argument('--no-filter-limit-up-bt', action='store_true',
                       help='(Backtest) Do not exclude limit-up stocks')
    parser.add_argument('--no-filter-limit-down-bt', action='store_true',
                       help='(Backtest) Do not exclude limit-down stocks')
    parser.add_argument('--no-filter-suspended-bt', action='store_true',
                       help='(Backtest) Do not exclude suspended stocks')
    parser.add_argument('--no-next-day-bt', action='store_true',
                       help='(Backtest) Use same-day returns instead of next-day')
    parser.add_argument('--backtest-output-dir', type=str, default=None,
                       help='Directory to save backtest results (default: backtest_results)')
    parser.add_argument('--plot-backtest', action='store_true',
                       help='Plot cumulative returns of decile backtest')
    parser.add_argument('--plot-input', type=str, default=None,
                       help='Optional: cumulative returns csv to plot (overrides in-memory)')
    parser.add_argument('--plot-output-dir', type=str, default=None,
                       help='Directory to save plots (default: backtest_plots)')
    parser.add_argument('--transaction-cost-bps', type=float, default=0,
                       help='One-way transaction cost in basis points for L/S return (default: 0). '
                            'Applied as 2x daily; assumes 100%% daily turnover — conservative upper bound.')

    args = parser.parse_args()
    
    # Run analysis
    analyzer = FactorRankICAnalyzer(args.data)
    analyzer.load_market_data() \
            .load_factor(args.factor) \
            .merge_data() \
            .calculate_rank_ic(
                exclude_st=not args.no_filter_st,
                exclude_limit_up=not args.no_filter_limit_up,
                exclude_limit_down=not args.no_filter_limit_down,
                exclude_suspended=not args.no_filter_suspended,
                use_next_day_return=not args.same_day,
                workers=args.workers,
                backend=args.backend,
                device=args.device
            ) \
            .print_statistics() \
            .save_results(args.output)

    if args.backtest_decile:
        analyzer.backtest_deciles(
            n_bins=10,
            exclude_st=not args.no_filter_st_bt,
            exclude_limit_up=not args.no_filter_limit_up_bt,
            exclude_limit_down=not args.no_filter_limit_down_bt,
            exclude_suspended=not args.no_filter_suspended_bt,
            use_next_day_return=not args.no_next_day_bt,
            workers=args.workers,
            transaction_cost_bps=args.transaction_cost_bps,
        ).save_backtest(output_dir=args.backtest_output_dir)

    if args.plot_backtest:
        analyzer.plot_backtest_cumulative(
            from_file=args.plot_input,
            output_dir=args.plot_output_dir,
            prefix=getattr(analyzer, 'factor_name', 'factor')
        )
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.chdir(Path(__file__).resolve().parents[2])

    if len(sys.argv) == 1:
        print("="*80)
        print("Factor Rank IC Analysis Tool - Interactive Mode")
        print("="*80)
        
        # Example usage
        factor_file = 'factors_by_type/alpha101/WorldQuant_alpha002.pkl'
        
        print(f"\nExample: Analyzing factor {factor_file}")
        
        analyzer = FactorRankICAnalyzer('data.pkl')
        analyzer.load_market_data() \
                .load_factor(factor_file) \
                .merge_data() \
                .calculate_rank_ic(
                    exclude_st=True,
                    exclude_limit_up=True,
                    exclude_limit_down=True,
                    exclude_suspended=True,
                    use_next_day_return=True
                ) \
                .print_statistics() \
                .save_results()

        # Example: run 10-decile backtest
        analyzer.backtest_deciles(
            n_bins=10,
            exclude_st=True,
            exclude_limit_up=True,
            exclude_limit_down=True,
            exclude_suspended=True,
            use_next_day_return=True
        ).save_backtest()
        
        # print("\nExample analysis completed!")
        # print("\nFor command line usage:")
        # print("  python factor_rankic_analysis.py --factor <factor_file> [options]")
        # print("\nOptions:")
        # print("  --no-filter-st          : Do not exclude ST stocks")
        # print("  --no-filter-limit-up    : Do not exclude limit-up stocks")
        # print("  --no-filter-limit-down  : Do not exclude limit-down stocks")
        # print("  --no-filter-suspended   : Do not exclude suspended stocks")
        # print("  --same-day             : Use same-day return instead of next-day forward return")
        # print("  --output <file>         : Output file path")
    else:
        main()
