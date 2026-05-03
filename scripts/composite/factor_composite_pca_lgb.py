#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor composite using PCA + LightGBM.
- Load multiple factors (from CSV or factors_by_type/*.pkl).
- PCA for dimensionality reduction on factor matrix.
- LightGBM to predict forward return from PCA components (or raw factors).
- Output: composite factor score, metrics, and saved model.
"""

import logging
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_factor_csv(csv_path: str) -> pd.DataFrame:
    """Load factor data from CSV (e.g. factor_data.csv). Expect columns: date, order_book_id, close, factor_value[, more factor columns]."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def compute_forward_return(df: pd.DataFrame, close_col: str = "close", forward: int = 1) -> pd.Series:
    """Compute next-period return by (order_book_id, date)."""
    df = df.sort_values(["order_book_id", "date"])
    out = (
        df.groupby("order_book_id")[close_col]
        .pct_change(periods=forward)
        .shift(-forward)
    )
    return out


def load_factors_from_pkl(
    factor_dir: str,
    factor_names: list = None,
    max_factors: int = 50,
) -> pd.DataFrame:
    """
    Load multiple factor pkl files from factors_by_type structure.
    Returns DataFrame with MultiIndex (order_book_id, date) and one column per factor.
    """
    base = Path(factor_dir)
    if not base.exists():
        return None
    pkl_files = list(base.rglob("*.pkl"))
    if not pkl_files:
        return None
    if factor_names:
        pkl_files = [f for f in pkl_files if f.stem in factor_names]
    pkl_files = pkl_files[: max_factors]
    dfs = []
    for f in pkl_files:
        try:
            df = pd.read_pickle(f)
            if isinstance(df.index, pd.MultiIndex) or (hasattr(df, "index") and df.index.names and len(df.index.names) >= 2):
                df = df.reset_index()
            if "order_book_id" not in df.columns or "date" not in df.columns:
                log.warning("Skipping %s: missing order_book_id or date columns", f.name)
                continue
            col = [c for c in df.columns if c not in ("order_book_id", "date")][0]
            df = df[["order_book_id", "date", col]].rename(columns={col: f.stem})
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            dfs.append(df)
        except Exception as e:
            log.warning("Skipping %s: %s", f.name, e)
            continue
    if not dfs:
        return None
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on=["order_book_id", "date"], how="inner")
    return out


def load_factor_npz_single(path: Path) -> pd.DataFrame:
    """Load one factor npz (values, index_order_book_id, index_date) -> DataFrame with order_book_id, date, value."""
    data = np.load(path, allow_pickle=True)
    if "values" not in data or "index_order_book_id" not in data or "index_date" not in data:
        data.close()
        return None
    values = data["values"].reshape(-1).astype(np.float64)
    codes = data["index_order_book_id"]
    dates = data["index_date"]
    data.close()
    if len(values) != len(codes) or len(values) != len(dates):
        return None
    name = path.stem
    df = pd.DataFrame({
        "order_book_id": [str(c) for c in codes],
        "date": pd.to_datetime(dates).normalize(),
        name: values,
    })
    return df


def load_factors_from_npz(npz_base: str, max_factors: int = 50) -> pd.DataFrame:
    """
    Load stock-level factors from factors_by_type_npy/factors_by_type (or similar).
    Scans npz_base for */*.npz, each npz: values, index_order_book_id, index_date.
    Returns DataFrame with order_book_id, date, factor1, factor2, ...
    """
    base = Path(npz_base)
    if not base.exists():
        return None
    files = sorted(base.glob("*/*.npz"))[:max_factors]
    if not files:
        files = sorted(base.glob("*.npz"))[:max_factors]
    if not files:
        return None
    out = None
    for f in files:
        try:
            df = load_factor_npz_single(f)
            if df is None or len(df) == 0:
                continue
            if out is None:
                out = df
            else:
                out = out.merge(df, on=["order_book_id", "date"], how="inner")
        except Exception:
            continue
    return out


def load_market_return_for_npz(market_npz: Path = None, market_csv: Path = None) -> pd.DataFrame:
    """
    Load (order_book_id, date, return_next) for merging with npz factors.
    Prefer market_data_cpp.npz (stock_ids, date_ids, return_next, stock_code_list, date_values).
    Fallback: CSV with date, order_book_id, close -> compute return_next.
    """
    if market_npz and market_npz.exists():
        try:
            data = np.load(market_npz, allow_pickle=True)
            stock_code_list = data["stock_code_list"]
            if hasattr(stock_code_list.flat[0], "decode"):
                stock_codes = [c.decode("utf-8") if hasattr(c, "decode") else str(c) for c in stock_code_list]
            else:
                stock_codes = [str(c) for c in stock_code_list]
            date_values = data["date_values"]
            date_vals = pd.to_datetime(date_values.astype(str), format="%Y%m%d")
            stock_ids = data["stock_ids"]
            date_ids = data["date_ids"]
            return_next = data["return_next"].astype(np.float64)
            data.close()
            df = pd.DataFrame({
                "order_book_id": [stock_codes[i] for i in stock_ids],
                "date": date_vals[date_ids].values,
                "return_next": return_next,
            })
            return df.dropna(subset=["return_next"])
        except Exception:
            pass
    if market_csv and market_csv.exists():
        df = pd.read_csv(market_csv)
        if "date" not in df.columns or "order_book_id" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["order_book_id", "date"])
        df["return_next"] = df.groupby("order_book_id")["close"].pct_change().shift(-1)
        return df[["order_book_id", "date", "return_next"]].dropna()
    return None


def load_decile_daily_factors(decile_dir: str, max_factors: int = 200) -> pd.DataFrame:
    """
    Load all *_decile_daily.csv from decile_cpp_batch_results, merge on date.
    Each file: date, Q1..Q10, LS. Use LS as the factor signal per day.
    Returns DataFrame with columns: date, LS_<factor1>, LS_<factor2>, ...
    """
    base = Path(decile_dir)
    if not base.exists():
        return None
    files = sorted(
        f for f in base.glob("*_decile_daily.csv")
        if not f.stem.startswith("composite_factor")
    )[:max_factors]
    if not files:
        return None
    out = None
    for f in files:
        try:
            df = pd.read_csv(f)
            if "date" not in df.columns or "LS" not in df.columns:
                continue
            df = df[["date", "LS"]].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            name = f.stem.replace("_decile_daily", "")
            df = df.rename(columns={"LS": name})
            if out is None:
                out = df
            else:
                out = out.merge(df, on="date", how="outer", suffixes=("", "_r"))
        except Exception:
            continue
    if out is not None:
        out = out.sort_values("date").fillna(0)
    return out


def build_factor_matrix_and_target(
    csv_path: str = None,
    factor_dir: str = None,
    factors_npz_dir: str = None,
    market_npz: str = None,
    market_csv: str = None,
    decile_dir: str = None,
    max_factors: int = 30,
    forward_periods: int = 1,
) -> tuple:
    """
    Build (X: factor matrix, y: forward return, meta) from CSV, pkl, npz, or decile.
    - factors_npz_dir: factors_by_type_npy/factors_by_type -> 个股因子 npz，需配合 market 算个股收益。
    - decile_dir: 组合日频 LS，无个股。
    """
    # Source: 个股因子 npz (factors_by_type_npy/factors_by_type)
    if factors_npz_dir and Path(factors_npz_dir).exists():
        fdf = load_factors_from_npz(factors_npz_dir, max_factors=max_factors)
        if fdf is not None and len(fdf) > 0:
            factor_cols = [c for c in fdf.columns if c not in ("order_book_id", "date")]
            if not factor_cols:
                raise ValueError("No factor columns in npz files.")
            mkt = load_market_return_for_npz(
                Path(market_npz) if market_npz else None,
                Path(market_csv) if market_csv else None,
            )
            if mkt is None:
                raise FileNotFoundError(
                    "For npz factors need market data: set --market_npz (e.g. market_data_cpp.npz) or --market_csv (date, order_book_id, close)."
                )
            merged = fdf.merge(mkt, on=["order_book_id", "date"], how="inner")
            merged = merged.dropna(subset=factor_cols + ["return_next"])
            X = merged[factor_cols].astype(np.float64)
            y = merged["return_next"].values
            meta = merged[["date", "order_book_id"]].copy()
            return X, y, meta, factor_cols
    # Source: decile_cpp_batch_results (daily decile LS per factor)
    if decile_dir and Path(decile_dir).exists():
        df = load_decile_daily_factors(decile_dir, max_factors=max_factors)
        if df is not None and len(df) > 0:
            factor_cols = [c for c in df.columns if c != "date"]
            if not factor_cols:
                raise ValueError("No factor columns in decile files.")
            df = df.sort_values("date").dropna(subset=factor_cols, how="all")
            # Target: next-day mean LS across all factors.
            # Note: this uses shift(-1) on the already-aggregated daily series, so the
            # look-ahead is exactly one trading day — the same day's factor values are
            # used to predict the *following* day's returns, which is realistically
            # achievable (factor computed at close t, trade at open t+1).
            df["return_next"] = df[factor_cols].mean(axis=1).shift(-1)
            df = df.dropna(subset=["return_next"])
            X = df[factor_cols].astype(np.float64).fillna(0)
            y = df["return_next"].values
            meta = df[["date"]].copy()
            return X, y, meta, factor_cols
    if csv_path and Path(csv_path).exists():
        df = load_factor_csv(csv_path)
        # Forward return from close
        df["return_next"] = compute_forward_return(df, close_col="close", forward=forward_periods)
        # Factor columns: everything that looks like a factor
        exclude = {"date", "order_book_id", "close", "return_next", "is_st", "is_limit_up", "is_limit_down", "is_suspended"}
        factor_cols = [c for c in df.columns if c not in exclude and (c == "factor_value" or (np.issubdtype(df[c].dtype, np.floating) or np.issubdtype(df[c].dtype, np.integer)))]
        if not factor_cols and "factor_value" in df.columns:
            factor_cols = ["factor_value"]
        if not factor_cols:
            raise ValueError("No factor columns found in CSV. Need at least 'factor_value' or similar.")
        use = ["date", "order_book_id", "return_next"] + factor_cols
        df = df.dropna(subset=use)
        X = df[factor_cols].astype(np.float64)
        y = df["return_next"].values
        meta = df[["date", "order_book_id"]].copy()
        return X, y, meta, factor_cols
    # Try pkl
    if factor_dir:
        fdf = load_factors_from_pkl(factor_dir, max_factors=max_factors)
        if fdf is not None:
            # We need market data for return: try factor_data.csv for close/return only
            csv_path = Path("factor_data.csv")
            if csv_path.exists():
                mkt = load_factor_csv(str(csv_path))
                mkt["return_next"] = compute_forward_return(mkt, close_col="close", forward=forward_periods)
                mkt = mkt[["date", "order_book_id", "return_next"]].dropna()
                merged = fdf.merge(mkt, on=["order_book_id", "date"], how="inner")
            else:
                raise FileNotFoundError("For pkl factors, need factor_data.csv (with date, order_book_id, close) to compute return.")
            factor_cols = [c for c in merged.columns if c not in ("date", "order_book_id", "return_next")]
            merged = merged.dropna(subset=factor_cols + ["return_next"])
            X = merged[factor_cols].astype(np.float64)
            y = merged["return_next"].values
            meta = merged[["date", "order_book_id"]].copy()
            return X, y, meta, factor_cols
    raise FileNotFoundError(
        "Provide csv_path, factor_dir (pkl), factors_npz_dir (e.g. factors_by_type_npy/factors_by_type) + market_npz/market_csv, or decile_dir."
    )


def train_pca_lgb(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list,
    n_components: int = 10,
    time_col: str = "date",
    test_size: float = 0.2,
    test_start_date: str = None,
    lgb_params: dict = None,
) -> dict:
    """
    Split by time, fit PCA on train, then LightGBM on PCA features (or raw if n_components >= n_features).
    Returns dict with model, scaler, pca, predictions, metrics.
    """
    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    if n_components < 1:
        n_components = 1
    if time_col in meta.columns:
        dates = pd.to_datetime(meta[time_col])
        t_min, t_max = dates.min(), dates.max()
        if test_start_date is not None:
            split_time = pd.to_datetime(test_start_date)
            train_mask = dates < split_time
            test_mask = dates >= split_time
        else:
            split_time = t_min + (t_max - t_min) * (1 - test_size)
            train_mask = dates < split_time
            test_mask = dates >= split_time
    else:
        train_mask = np.ones(len(meta), dtype=bool)
        train_mask[int(len(meta) * (1 - test_size)):] = False
        test_mask = ~train_mask
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    meta_train = meta.loc[train_mask].reset_index(drop=True)
    meta_test = meta.loc[test_mask].reset_index(drop=True)
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    # LightGBM
    default_lgb = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
        "n_estimators": 200,
    }
    if lgb_params:
        default_lgb.update(lgb_params)
    if lgb is None:
        raise ImportError("Please install lightgbm: pip install lightgbm")
    pca_names = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
    train_data = lgb.Dataset(X_train_pca, label=y_train, feature_name=pca_names)
    model = lgb.train(
        default_lgb,
        train_data,
        valid_sets=[lgb.Dataset(X_test_pca, label=y_test, feature_name=pca_names)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_train = model.predict(X_train_pca)
    pred_test = model.predict(X_test_pca)
    # Metrics
    def _ic(u, v):
        return np.corrcoef(u, v)[0, 1] if np.std(u) > 0 and np.std(v) > 0 else np.nan
    def _rank_ic(u, v):
        from scipy.stats import spearmanr
        return spearmanr(u, v)[0] if len(np.unique(u)) > 1 and len(np.unique(v)) > 1 else np.nan
    ic_train = _ic(pred_train, y_train)
    ic_test = _ic(pred_test, y_test)
    ric_train = _rank_ic(pred_train, y_train)
    ric_test = _rank_ic(pred_test, y_test)
    return {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "feature_names": feature_names,
        "pca_names": pca_names,
        "n_components": n_components,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "ic_train": ic_train,
        "ic_test": ic_test,
        "rank_ic_train": ric_train,
        "rank_ic_test": ric_test,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def main():
    _root = Path(__file__).resolve().parents[2]
    os.chdir(_root)
    parser = argparse.ArgumentParser(description="Factor composite: PCA + LightGBM")
    parser.add_argument("--csv", type=str, default="", help="Factor CSV (date, order_book_id, close, factor columns)")
    parser.add_argument("--factor_dir", type=str, default="", help="Optional: factors_by_type dir to load multiple .pkl factors")
    parser.add_argument("--factors_npz_dir", type=str, default="", help="Stock-level factor npz dir (e.g. factors_by_type_npy/factors_by_type); need --market_npz or --market_csv for return_next")
    parser.add_argument("--market_npz", type=str, default="", help="Market npz with return_next (e.g. market_data_cpp.npz), used with --factors_npz_dir")
    parser.add_argument("--market_csv", type=str, default="", help="Market CSV with date, order_book_id, close (to compute return_next), used with --factors_npz_dir")
    parser.add_argument("--decile_dir", type=str, default="decile_cpp_batch_results", help="Decile daily CSVs dir (e.g. decile_cpp_batch_results)")
    parser.add_argument("--max_factors", type=int, default=200, help="Max number of factors to load (pkl, npz, or decile)")
    parser.add_argument("--n_components", type=int, default=20, help="PCA components")
    parser.add_argument("--forward", type=int, default=1, help="Forward periods for return")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio (ignored if --test_start_date set)")
    parser.add_argument("--test_start_date", type=str, default="2023-01-01", help="Test set start date: train before this, test from this (e.g. 2023-01-01). Empty to use test_ratio.")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: decile_dir when using decile_dir)")
    args = parser.parse_args()
    csv_path = args.csv if args.csv and Path(args.csv).exists() else None
    factor_dir = args.factor_dir if args.factor_dir and Path(args.factor_dir).exists() else None
    factors_npz_dir = args.factors_npz_dir if args.factors_npz_dir and Path(args.factors_npz_dir).exists() else None
    market_npz = args.market_npz if args.market_npz and Path(args.market_npz).exists() else None
    market_csv = args.market_csv if args.market_csv and Path(args.market_csv).exists() else None
    if not market_npz and not market_csv and factors_npz_dir:
        for p in [_root / "market_data_cpp.npz", Path("market_data_cpp.npz"), Path("e:/data/market_data_cpp.npz")]:
            if p.exists() and str(p).endswith(".npz"):
                market_npz = str(p)
                break
    decile_dir = args.decile_dir if args.decile_dir and Path(args.decile_dir).exists() else None
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif decile_dir:
        out_dir = Path(decile_dir)
    else:
        out_dir = Path("factor_composite_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building factor matrix and target...")
    X, y, meta, factor_cols = build_factor_matrix_and_target(
        csv_path=csv_path,
        factor_dir=factor_dir,
        factors_npz_dir=factors_npz_dir,
        market_npz=market_npz,
        market_csv=market_csv,
        decile_dir=decile_dir,
        max_factors=args.max_factors,
        forward_periods=args.forward,
    )
    print(f"Factors: {factor_cols}, shape = {X.shape}, target len = {len(y)}")
    n_components = min(args.n_components, X.shape[1])
    test_start_date = args.test_start_date.strip() or None
    result = train_pca_lgb(
        X.values if isinstance(X, pd.DataFrame) else X,
        y,
        meta,
        factor_cols,
        n_components=n_components,
        test_size=args.test_ratio,
        test_start_date=test_start_date,
    )
    print("\n--- PCA ---")
    print(f"Explained variance ratio (first 10): {result['explained_variance_ratio'][:10]}")
    print(f"Cumulative: {np.cumsum(result['explained_variance_ratio'])[:10]}")
    print("\n--- Train/Test (time-based split) ---")
    if test_start_date:
        print(f"Train = before {test_start_date}, Test = from {test_start_date}. Train days: {len(result['meta_train'])}, Test days: {len(result['meta_test'])}")
    else:
        print(f"Test set = last {args.test_ratio*100:.0f}% of dates. Train days: {len(result['meta_train'])}, Test days: {len(result['meta_test'])}")
    print("\n--- LightGBM (composite factor) ---")
    print(f"IC  train: {result['ic_train']:.4f}  test: {result['ic_test']:.4f}")
    print(f"Rank IC train: {result['rank_ic_train']:.4f}  test: {result['rank_ic_test']:.4f}")
    # Save composite factor (test set predictions with keys)
    comp_test = result["meta_test"].copy()
    comp_test["composite_score"] = result["pred_test"]
    comp_test["return_next"] = result["y_test"]
    comp_test.to_csv(out_dir / "composite_factor_test.csv", index=False)
    comp_train = result["meta_train"].copy()
    comp_train["composite_score"] = result["pred_train"]
    comp_train["return_next"] = result["y_train"]
    comp_train.to_csv(out_dir / "composite_factor_train.csv", index=False)
    result["model"].save_model(str(out_dir / "lgb_model.txt"))
    scaler = result["scaler"]
    np.savez(
        out_dir / "pca_scaler.npz",
        pca_components=result["pca"].components_,
        pca_explained_variance_ratio=result["pca"].explained_variance_ratio_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_names=np.array(result["feature_names"], dtype=object),
    )
    # Metrics summary (persist for comparison)
    metrics = pd.DataFrame([{
        "n_factors": len(result["feature_names"]),
        "n_components": result["n_components"],
        "test_ratio": args.test_ratio,
        "test_start_date": test_start_date or "",
        "ic_train": result["ic_train"],
        "ic_test": result["ic_test"],
        "rank_ic_train": result["rank_ic_train"],
        "rank_ic_test": result["rank_ic_test"],
        "train_days": len(result["meta_train"]),
        "test_days": len(result["meta_test"]),
    }])
    metrics.to_csv(out_dir / "composite_metrics_summary.csv", index=False)
    # Daily top 5 factors by that day's LS (best 5 for each date)
    save_top5_factors_daily(X, meta, factor_cols, out_dir)
    # Composite daily in same format as other decile_daily files (date, Q1..Q10, LS)
    all_dates = pd.concat([result["meta_train"], result["meta_test"]], ignore_index=True)
    pred_train, pred_test = result["pred_train"], result["pred_test"]
    y_train_arr, y_test_arr = result["y_train"], result["y_test"]
    all_scores = np.concatenate((pred_train, pred_test))
    all_returns = np.concatenate((y_train_arr, y_test_arr))
    idx = all_dates["date"].argsort()
    daily = pd.DataFrame({
        "date": pd.to_datetime(all_dates["date"].iloc[idx]).dt.strftime("%Y-%m-%d"),
        "composite_score": all_scores[idx],
        "return_next": all_returns[idx],
    })
    daily.to_csv(out_dir / "composite_factor_daily.csv", index=False)
    decile_daily = daily.copy()
    for q in range(1, 11):
        decile_daily[f"Q{q}"] = decile_daily["composite_score"]
    decile_daily["LS"] = decile_daily["composite_score"]
    decile_daily[["date"] + [f"Q{i}" for i in range(1, 11)] + ["LS"]].to_csv(
        out_dir / "composite_factor_decile_daily.csv", index=False
    )
    # Compute and save composite return stats + cumulative return plot
    compute_and_plot_composite_returns(daily, out_dir)
    print(f"\nSaved to {out_dir}:")
    print("  composite_factor_train.csv, composite_factor_test.csv")
    print("  composite_factor_daily.csv, composite_factor_decile_daily.csv")
    print("  composite_metrics_summary.csv, composite_return_summary.csv, composite_cumulative_return.csv")
    print("  composite_top5_factors_daily.csv, composite_cumulative_return.png, lgb_model.txt, pca_scaler.npz")
    return 0


def save_top5_factors_daily(X, meta: pd.DataFrame, factor_cols: list, out_dir: Path, top_k: int = 5) -> None:
    """
    For each date, save the top `top_k` factors by that day's factor value (LS in decile case).
    Output: composite_top5_factors_daily.csv with date, rank1_factor..rankK_factor, rank1_LS..rankK_LS.
    """
    X_arr = np.asarray(X)
    if X_arr.size == 0 or len(factor_cols) < top_k:
        return
    top_k = min(top_k, len(factor_cols))
    if "order_book_id" in meta.columns and meta["date"].nunique() < len(meta):
        # Multiple rows per date: aggregate by date (mean), then top-k per date
        uniq_dates = meta["date"].dropna().unique()
        rows = []
        for date in uniq_dates:
            mask = (meta["date"] == date).values
            row = np.nanmean(X_arr[mask], axis=0)
            idx = np.argsort(row)[-top_k:][::-1]
            names = [factor_cols[j] for j in idx]
            vals = [float(row[j]) for j in idx]
            rows.append({"date": pd.to_datetime(date).strftime("%Y-%m-%d"), **{f"rank{r+1}_factor": names[r] for r in range(top_k)}, **{f"rank{r+1}_LS": vals[r] for r in range(top_k)}})
    else:
        # One row per date (decile_dir)
        rows = []
        for i in range(len(meta)):
            row = X_arr[i]
            idx = np.argsort(row)[-top_k:][::-1]
            names = [factor_cols[j] for j in idx]
            vals = [float(row[j]) for j in idx]
            date = meta.iloc[i]["date"]
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            rows.append({"date": date_str, **{f"rank{r+1}_factor": names[r] for r in range(top_k)}, **{f"rank{r+1}_LS": vals[r] for r in range(top_k)}})
    pd.DataFrame(rows).to_csv(out_dir / "composite_top5_factors_daily.csv", index=False)


def compute_and_plot_composite_returns(daily: pd.DataFrame, out_dir: Path) -> None:
    """
    From daily (date, composite_score, return_next), compute benchmark & strategy returns,
    save summary + cumulative series + plot. All plot labels in English.
    """
    if daily.empty or "return_next" not in daily.columns or "composite_score" not in daily.columns:
        return
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["return_next", "composite_score"])
    # Benchmark: equal-weight factor portfolio (return_next is next-day mean LS)
    df["daily_return_benchmark"] = df["return_next"]
    # Strategy: long when composite_score > 0, short when < 0; daily ret = sign(score) * return_next
    df["daily_return_strategy"] = np.sign(df["composite_score"]) * df["return_next"]
    df["cum_benchmark"] = (1 + df["daily_return_benchmark"]).cumprod()
    df["cum_strategy"] = (1 + df["daily_return_strategy"]).cumprod()
    df[["date", "daily_return_benchmark", "daily_return_strategy", "cum_benchmark", "cum_strategy"]].to_csv(
        out_dir / "composite_cumulative_return.csv", index=False
    )
    # Summary stats (annualize with 252 trading days)
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
    print("\n--- Composite factor return summary ---")
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


if __name__ == "__main__":
    sys.exit(main())
