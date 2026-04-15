# -*- coding: utf-8 -*-
"""20-day trailing volatility: rolling std of daily simple returns from local data.pkl."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def compute_factor_df() -> pd.DataFrame:
    """MultiIndex (order_book_id, date), one column volatility_20d."""
    root = Path(__file__).resolve().parents[1]
    p = root / "data.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"Need {p}")
    with open(p, "rb") as f:
        mkt = pickle.load(f)
    need = {"order_book_id", "date", "close"}
    if not need.issubset(mkt.columns):
        raise ValueError(f"data.pkl must contain {need}; got {list(mkt.columns)}")
    mkt = mkt.sort_values(["order_book_id", "date"])
    mkt["ret"] = mkt.groupby("order_book_id", sort=False)["close"].pct_change(fill_method=None)
    mkt["volatility_20d"] = mkt.groupby("order_book_id", sort=False)["ret"].transform(
        lambda s: s.rolling(20, min_periods=15).std()
    )
    fac = mkt.dropna(subset=["volatility_20d"])[["order_book_id", "date", "volatility_20d"]]
    return fac.set_index(["order_book_id", "date"])[["volatility_20d"]]
