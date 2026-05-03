#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor neutralization: remove market-wide and industry exposures from a factor
before IC computation to get a cleaner signal.

Three methods (applied in order, each optional):
  1. cross_demean   — subtract cross-sectional mean each day (removes market level)
  2. mktcap         — OLS residual against log(market_cap) each day
  3. industry       — subtract industry mean each day

All methods work at the (order_book_id, date) level.
Columns looked up in market_data:
  - market_cap or mkt_cap or total_market_cap or circulating_market_cap
  - industry or industry_code or sw_industry_code or cs_industry_code
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MKTCAP_COLS = ["market_cap", "mkt_cap", "total_market_cap", "circulating_market_cap", "float_market_cap"]
_INDUSTRY_COLS = ["industry", "industry_code", "sw_industry", "sw_industry_code",
                  "cs_industry", "cs_industry_code", "citics_industry"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ols_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y ~ x (intercept included). Returns y - y_hat."""
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 5:
        return y
    X = np.column_stack([np.ones(valid.sum()), x[valid]])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y[valid], rcond=None)
    except np.linalg.LinAlgError:
        return y
    y_hat = np.full_like(y, np.nan)
    y_hat[valid] = X @ beta
    residual = y.copy()
    residual[valid] = y[valid] - y_hat[valid]
    return residual


def neutralize_factor(
    factor_df: pd.DataFrame,
    market_df: pd.DataFrame,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """
    Neutralize factor against market exposures.

    Parameters
    ----------
    factor_df : DataFrame with MultiIndex (order_book_id, date), one column
    market_df : DataFrame with columns order_book_id, date, and optionally
                market_cap / industry columns
    methods   : list of methods to apply in order.
                Options: 'cross_demean', 'mktcap', 'industry'
                Default: ['cross_demean'] — always safe.
                'mktcap' and 'industry' silently skip if columns not found.

    Returns
    -------
    Neutralized factor DataFrame (same shape and index as input).
    """
    if methods is None:
        methods = ["cross_demean"]

    fname = factor_df.columns[0]
    fac = factor_df.reset_index()
    fac.columns = ["order_book_id", "date", fname]
    fac["order_book_id"] = fac["order_book_id"].astype(str)
    fac["date"] = pd.to_datetime(fac["date"]).dt.normalize()

    mkt = market_df.copy()
    mkt["order_book_id"] = mkt["order_book_id"].astype(str)
    mkt["date"] = pd.to_datetime(mkt["date"]).dt.normalize()

    mktcap_col = _find_col(mkt, _MKTCAP_COLS)
    industry_col = _find_col(mkt, _INDUSTRY_COLS)

    applied = []

    if "mktcap" in methods:
        if mktcap_col is None:
            log.warning("mktcap neutralization skipped: no market cap column found in market_data "
                        "(looked for: %s)", _MKTCAP_COLS)
        else:
            applied.append("mktcap")

    if "industry" in methods:
        if industry_col is None:
            log.warning("industry neutralization skipped: no industry column found in market_data "
                        "(looked for: %s)", _INDUSTRY_COLS)
        else:
            applied.append("industry")

    # Merge relevant columns
    merge_cols = ["order_book_id", "date"]
    if mktcap_col and "mktcap" in applied:
        merge_cols.append(mktcap_col)
    if industry_col and "industry" in applied:
        merge_cols.append(industry_col)

    if len(merge_cols) > 2:
        fac = pd.merge(fac, mkt[merge_cols], on=["order_book_id", "date"], how="left")

    result_vals = fac[fname].values.astype(float).copy()

    for method in methods:
        if method == "cross_demean":
            # Subtract daily cross-sectional mean
            daily_mean = fac.groupby("date")[fname].transform("mean")
            result_vals = result_vals - daily_mean.values
            log.info("Applied cross_demean neutralization")

        elif method == "mktcap" and "mktcap" in applied:
            log.info("Applying mktcap neutralization (column: %s)", mktcap_col)
            new_vals = result_vals.copy()
            for date, idx in fac.groupby("date").groups.items():
                y = result_vals[idx]
                raw_mc = fac.loc[idx, mktcap_col].values.astype(float)
                x = np.where(raw_mc > 0, np.log(raw_mc + 1e-10), np.nan)
                new_vals[idx] = _ols_residual(y, x)
            result_vals = new_vals

        elif method == "industry" and "industry" in applied:
            log.info("Applying industry neutralization (column: %s)", industry_col)
            tmp = pd.DataFrame({
                "date": fac["date"].values,
                "industry": fac[industry_col].values,
                "val": result_vals,
            })
            ind_mean = tmp.groupby(["date", "industry"])["val"].transform("mean")
            result_vals = result_vals - ind_mean.fillna(0).values

    fac[fname] = result_vals
    out = fac.set_index(["order_book_id", "date"])[[fname]]
    applied_str = ", ".join(methods) if methods else "none"
    log.info("Neutralization complete (%s). Non-null: %.1f%%",
             applied_str, out[fname].notna().mean() * 100)
    return out


def neutralize_from_file(
    factor_pkl: str,
    data_pkl: str,
    methods: list[str] | None = None,
    output_pkl: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: load from files, neutralize, optionally save."""
    import pickle
    import os

    with open(data_pkl, "rb") as f:
        market_df = pickle.load(f)
    factor_df = pd.read_pickle(factor_pkl)

    result = neutralize_factor(factor_df, market_df, methods=methods)

    if output_pkl:
        os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
        result.to_pickle(output_pkl)
        log.info("Neutralized factor saved to: %s", output_pkl)

    return result
