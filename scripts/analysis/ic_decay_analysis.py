#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC Decay Analysis: compute IC and Rank IC at multiple forward-return horizons.

Usage:
  python scripts/analysis/ic_decay_analysis.py --factor factors_by_type/alpha101/WorldQuant_alpha002.pkl
  python scripts/analysis/ic_decay_analysis.py --factor my_factor.pkl --horizons 1 5 10 20 60
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 5, 10, 20]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _compute_rank_ic_series(data: pd.DataFrame, factor_name: str, return_col: str) -> pd.Series:
    """Per-day Spearman rank IC between factor and a given return column."""
    rows = []
    for date, g in data.groupby("date", sort=True):
        g2 = g[[factor_name, return_col]].dropna()
        if len(g2) < 10:
            rows.append((date, np.nan))
            continue
        rho, _ = spearmanr(g2[factor_name], g2[return_col])
        rows.append((date, rho))
    s = pd.Series(dict(rows), name=return_col)
    s.index = pd.to_datetime(s.index)
    return s


class ICDecayAnalyzer:
    """Compute IC/Rank IC at multiple forward-return horizons to measure signal persistence."""

    def __init__(self, data_pkl: str = "data.pkl"):
        self.data_pkl = data_pkl
        self.market_data: pd.DataFrame | None = None
        self.factor_data: pd.DataFrame | None = None
        self.factor_name: str | None = None
        self.merged_data: pd.DataFrame | None = None
        self.decay_results: pd.DataFrame | None = None

    def load_market_data(self) -> "ICDecayAnalyzer":
        if self.market_data is not None:
            return self
        if not os.path.isfile(self.data_pkl):
            raise FileNotFoundError(f"Market data not found: {self.data_pkl}")
        log.info("Loading market data from %s", self.data_pkl)
        with open(self.data_pkl, "rb") as f:
            self.market_data = pickle.load(f)
        self.market_data["date"] = pd.to_datetime(self.market_data["date"]).dt.normalize()
        self.market_data["order_book_id"] = self.market_data["order_book_id"].astype(str)
        log.info("Market data loaded: %s rows", len(self.market_data))
        return self

    def load_factor(self, factor_file: str) -> "ICDecayAnalyzer":
        if not os.path.isfile(factor_file):
            raise FileNotFoundError(f"Factor file not found: {factor_file}")
        log.info("Loading factor from %s", factor_file)
        self.factor_data = pd.read_pickle(factor_file)
        self.factor_name = self.factor_data.columns[0]
        log.info("Factor: %r  shape=%s", self.factor_name, self.factor_data.shape)
        return self

    def _merge(self) -> pd.DataFrame:
        if self.merged_data is not None:
            return self.merged_data
        fac = self.factor_data.reset_index()
        fac.columns = ["order_book_id", "date", self.factor_name]
        fac["order_book_id"] = fac["order_book_id"].astype(str)
        fac["date"] = pd.to_datetime(fac["date"]).dt.normalize()

        mkt = self.market_data.sort_values(["order_book_id", "date"])
        self.merged_data = pd.merge(
            mkt[["order_book_id", "date", "close"]],
            fac,
            on=["order_book_id", "date"],
            how="inner",
        )
        log.info("Merged: %s rows", len(self.merged_data))
        return self.merged_data

    def compute_decay(
        self,
        horizons: list[int] = None,
        exclude_st: bool = True,
        exclude_limit_up: bool = True,
        exclude_limit_down: bool = True,
        exclude_suspended: bool = True,
        min_stocks: int = 30,
        workers: int | None = None,
    ) -> "ICDecayAnalyzer":
        horizons = horizons or DEFAULT_HORIZONS
        log.info("Computing IC decay for horizons %s", horizons)

        data = self._merge().copy()
        # Apply market filters using available columns
        for flag, col in [
            (exclude_st, "ST"),
            (exclude_limit_up, "limit_up_flag"),
            (exclude_limit_down, "limit_down_flag"),
            (exclude_suspended, "suspended"),
        ]:
            if flag and col in data.columns:
                data = data[~data[col].astype(bool)]

        data = data.sort_values(["order_book_id", "date"])

        # Precompute all forward-return columns at once
        for h in horizons:
            col = f"ret_{h}d"
            data[col] = (
                data.groupby("order_book_id")["close"]
                .pct_change(periods=h, fill_method=None)
                .shift(-h)
            )

        rows = []
        for h in horizons:
            ret_col = f"ret_{h}d"
            sub = data.dropna(subset=[self.factor_name, ret_col])
            if len(sub) == 0:
                log.warning("Horizon %dd: no valid data", h)
                rows.append(self._empty_row(h))
                continue

            # Per-day IC
            ic_vals, ric_vals, n_days_valid = [], [], 0
            for _, g in sub.groupby("date", sort=True):
                g2 = g[[self.factor_name, ret_col]].dropna()
                if len(g2) < min_stocks:
                    continue
                n_days_valid += 1
                ic_vals.append(g2[self.factor_name].corr(g2[ret_col]))
                rho, _ = spearmanr(g2[self.factor_name], g2[ret_col])
                ric_vals.append(rho)

            if not ic_vals:
                rows.append(self._empty_row(h))
                continue

            ic_arr = np.array(ic_vals)
            ric_arr = np.array(ric_vals)
            rows.append({
                "horizon": h,
                "mean_ic": float(np.nanmean(ic_arr)),
                "std_ic": float(np.nanstd(ic_arr)),
                "ic_ir": float(np.nanmean(ic_arr) / np.nanstd(ic_arr)) if np.nanstd(ic_arr) > 0 else np.nan,
                "ic_win_rate": float(np.nanmean(ic_arr > 0)),
                "mean_rank_ic": float(np.nanmean(ric_arr)),
                "std_rank_ic": float(np.nanstd(ric_arr)),
                "rank_ic_ir": float(np.nanmean(ric_arr) / np.nanstd(ric_arr)) if np.nanstd(ric_arr) > 0 else np.nan,
                "rank_ic_win_rate": float(np.nanmean(ric_arr > 0)),
                "n_valid_days": n_days_valid,
            })
            log.info("Horizon %2dd | mean_ric=%.4f  ir=%.3f  win=%.2f%%  days=%d",
                     h,
                     rows[-1]["mean_rank_ic"],
                     rows[-1]["rank_ic_ir"] if not np.isnan(rows[-1]["rank_ic_ir"]) else 0,
                     rows[-1]["rank_ic_win_rate"] * 100,
                     n_days_valid)

        self.decay_results = pd.DataFrame(rows)
        return self

    @staticmethod
    def _empty_row(h: int) -> dict:
        return {
            "horizon": h, "mean_ic": np.nan, "std_ic": np.nan, "ic_ir": np.nan,
            "ic_win_rate": np.nan, "mean_rank_ic": np.nan, "std_rank_ic": np.nan,
            "rank_ic_ir": np.nan, "rank_ic_win_rate": np.nan, "n_valid_days": 0,
        }

    def save(self, output_file: str | None = None) -> "ICDecayAnalyzer":
        if self.decay_results is None or self.decay_results.empty:
            log.warning("No decay results to save.")
            return self
        if output_file is None:
            output_file = f"{self.factor_name}_ic_decay.csv"
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        self.decay_results.to_csv(output_file, index=False, encoding="utf-8-sig")
        log.info("Decay results saved to: %s", output_file)
        return self

    def plot(self, output_dir: str | None = None, prefix: str | None = None) -> "ICDecayAnalyzer":
        if plt is None:
            log.warning("matplotlib not available — skipping plot.")
            return self
        if self.decay_results is None or self.decay_results.empty:
            return self

        output_dir = output_dir or "."
        os.makedirs(output_dir, exist_ok=True)
        prefix = prefix or self.factor_name or "factor"
        df = self.decay_results.copy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"IC Decay — {prefix}", fontsize=13)

        ax = axes[0]
        ax.bar(df["horizon"].astype(str), df["mean_rank_ic"], color="#1f77b4", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Forward horizon (days)")
        ax.set_ylabel("Mean Rank IC")
        ax.set_title("Mean Rank IC by Horizon")
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        ax.bar(df["horizon"].astype(str), df["rank_ic_ir"], color="#ff7f0e", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Forward horizon (days)")
        ax.set_ylabel("Rank IC IR")
        ax.set_title("Rank IC IR by Horizon")
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{prefix}_ic_decay.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info("Decay plot saved to: %s", out_path)
        return self

    def summary_str(self) -> str:
        if self.decay_results is None:
            return "No results."
        lines = [f"IC Decay — {self.factor_name}",
                 f"{'Horizon':>8}  {'MeanRIC':>8}  {'RIC_IR':>7}  {'WinRate':>8}  {'Days':>6}"]
        for _, row in self.decay_results.iterrows():
            lines.append(
                f"{int(row['horizon']):>7}d  "
                f"{row['mean_rank_ic']:>8.4f}  "
                f"{row['rank_ic_ir']:>7.3f}  "
                f"{row['rank_ic_win_rate']*100:>7.1f}%  "
                f"{int(row['n_valid_days']):>6}"
            )
        return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)

    p = argparse.ArgumentParser(description="IC Decay Analysis")
    p.add_argument("--factor", required=True, help="Factor .pkl path")
    p.add_argument("--data", default="data.pkl", help="Market data .pkl")
    p.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    p.add_argument("--output-dir", default="", help="Output directory for CSV and plot")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    factor_name = Path(args.factor).stem
    out_dir = args.output_dir or str(root / "ic_decay_results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{factor_name}_ic_decay.csv")

    ana = ICDecayAnalyzer(args.data)
    ana.load_market_data().load_factor(args.factor).compute_decay(horizons=args.horizons)
    ana.save(out_csv)
    if not args.no_plot:
        ana.plot(output_dir=out_dir, prefix=factor_name)
    print(ana.summary_str())


if __name__ == "__main__":
    main()
