#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
View composite factor return: read composite_factor_daily.csv and show
return summary + cumulative return plot. All labels in English.
Usage:
  python view_composite_returns.py
  python view_composite_returns.py --dir decile_cpp_batch_results
  python view_composite_returns.py --dir factor_composite_output
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def compute_and_plot(daily_path: Path, out_dir: Path = None) -> None:
    if out_dir is None:
        out_dir = daily_path.parent
    df = pd.read_csv(daily_path)
    if df.empty or "return_next" not in df.columns or "composite_score" not in df.columns:
        print("Need columns: date, composite_score, return_next")
        return
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["return_next", "composite_score"])
    df["daily_return_benchmark"] = df["return_next"]
    df["daily_return_strategy"] = np.sign(df["composite_score"]) * df["return_next"]
    df["cum_benchmark"] = (1 + df["daily_return_benchmark"]).cumprod()
    df["cum_strategy"] = (1 + df["daily_return_strategy"]).cumprod()
    df[["date", "daily_return_benchmark", "daily_return_strategy", "cum_benchmark", "cum_strategy"]].to_csv(
        out_dir / "composite_cumulative_return.csv", index=False
    )
    n = len(df)
    days_per_year = 252
    def _stats(r):
        r = np.asarray(r)
        total = (1 + r).prod() - 1
        ann = (1 + total) ** (days_per_year / n) - 1 if n > 0 else 0
        sharpe = (r.mean() / r.std() * np.sqrt(days_per_year)) if r.std() > 0 else 0
        cum = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        mdd = dd.min()
        return {"total_return": total, "annualized_return": ann, "sharpe_ratio": sharpe, "max_drawdown": mdd}
    s_bench = _stats(df["daily_return_benchmark"])
    s_strat = _stats(df["daily_return_strategy"])
    summary = pd.DataFrame([
        {"series": "benchmark_equal_weight", **s_bench},
        {"series": "strategy_long_short_by_score", **s_strat},
    ])
    summary.to_csv(out_dir / "composite_return_summary.csv", index=False)
    print("Composite factor return summary")
    print(summary.to_string(index=False))
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["date"], df["cum_benchmark"], label="Benchmark (equal-weight factors)", alpha=0.8)
        ax.plot(df["date"], df["cum_strategy"], label="Strategy (long/short by composite score)", alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.set_title("Composite factor: cumulative return")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "composite_cumulative_return.png", dpi=150)
        plt.close(fig)
        print(f"\nPlot saved: {out_dir / 'composite_cumulative_return.png'}")
    print(f"\nCumulative series saved: {out_dir / 'composite_cumulative_return.csv'}")


def main():
    import os
    _root = Path(__file__).resolve().parents[2]
    os.chdir(_root)
    p = argparse.ArgumentParser(description="View composite factor returns from composite_factor_daily.csv")
    p.add_argument("--dir", type=str, default="decile_cpp_batch_results", help="Directory containing composite_factor_daily.csv")
    args = p.parse_args()
    base = Path(args.dir)
    daily_path = base / "composite_factor_daily.csv"
    if not daily_path.exists():
        print(f"Not found: {daily_path}")
        print("Run factor_composite_pca_lgb.py first, or set --dir to the output folder.")
        return 1
    compute_and_plot(daily_path, base)
    return 0


if __name__ == "__main__":
    exit(main())
