#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download A-share index daily OHLCV data via akshare and save as index_data.pkl.

Code conversion (akshare numeric → RiceQuant format):
  Shanghai indices (000xxx, 001xxx): code + ".XSHG"
  Shenzhen indices (399xxx):         code + ".XSHE"

Output:  <project_root>/index_data.pkl
  DataFrame columns: date, order_book_id, open, high, low, close,
                     volume, total_turnover, pct_change

Usage:
  python scripts/download/download_index_data.py
  python scripts/download/download_index_data.py --start 20100104 --end 20250731
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ── Index registry ────────────────────────────────────────────────────────────
# (akshare_code, rq_code, name)
INDICES = [
    ("000001", "000001.XSHG", "Shanghai Composite"),
    ("000300", "000300.XSHG", "CSI 300"),
    ("000905", "000905.XSHG", "CSI 500"),
    ("000852", "000852.XSHG", "CSI 1000"),
    ("000016", "000016.XSHG", "SSE 50"),
    ("399001", "399001.XSHE", "Shenzhen Component"),
    ("399006", "399006.XSHE", "ChiNext"),
]

_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "total_turnover",
    "涨跌幅": "pct_change",
}


def _rq_code(akshare_code: str) -> str:
    """Convert akshare numeric code to RiceQuant exchange-suffixed code."""
    code = akshare_code.strip()
    if code.startswith("39") or code.startswith("40"):
        return f"{code}.XSHE"
    return f"{code}.XSHG"


def download_index(akshare_code: str, rq_code: str, name: str,
                   start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    import akshare as ak

    for attempt in range(1, retries + 1):
        try:
            df = ak.index_zh_a_hist(
                symbol=akshare_code,
                period="daily",
                start_date=start,
                end_date=end,
            )
            if df is None or df.empty:
                print(f"  [{name}] empty response", flush=True)
                return None

            df = df.rename(columns=_COL_MAP)
            keep = [c for c in ["date", "open", "high", "low", "close",
                                 "volume", "total_turnover", "pct_change"] if c in df.columns]
            df = df[keep].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["order_book_id"] = rq_code

            for col in ["open", "high", "low", "close", "volume", "total_turnover", "pct_change"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            print(f"  [{name}] {rq_code}  {len(df)} rows  "
                  f"{df['date'].min().date()} → {df['date'].max().date()}", flush=True)
            return df

        except Exception as e:
            print(f"  [{name}] attempt {attempt}/{retries} failed: {e}", flush=True)
            if attempt < retries:
                time.sleep(3 * attempt)

    return None


def main():
    parser = argparse.ArgumentParser(description="Download A-share index data via akshare")
    parser.add_argument("--start", default="20100101", help="Start date YYYYMMDD")
    parser.add_argument("--end",   default="20251231", help="End date YYYYMMDD")
    parser.add_argument("--out",   default="", help="Output pickle path (default: <root>/index_data.pkl)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "index_data.pkl"

    print(f"Downloading {len(INDICES)} indices  {args.start} → {args.end}")
    frames: list[pd.DataFrame] = []

    for ak_code, rq_code, name in INDICES:
        df = download_index(ak_code, rq_code, name, args.start, args.end)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(1)   # be polite to akshare's data source

    if not frames:
        print("ERROR: no index data downloaded.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["order_book_id", "date"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(out_path)
    print(f"\nSaved {len(combined)} rows × {combined.shape[1]} cols → {out_path}")
    print("Indices included:", combined["order_book_id"].unique().tolist())

    # Quick sanity: CSI300 recent close
    csi300 = combined[combined["order_book_id"] == "000300.XSHG"]
    if not csi300.empty:
        last = csi300.sort_values("date").iloc[-1]
        print(f"CSI300 last close: {last['date'].date()}  {last['close']:.2f}")


if __name__ == "__main__":
    main()
