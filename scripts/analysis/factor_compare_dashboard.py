#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor compare dashboard script.

Given one new factor RankIC daily csv + baseline factor summary/daily files, output:
1) RankIC/IR ranking change
2) Correlation heatmap with top baseline factors
3) Segmented stability analysis
4) Recommendation for composite pool inclusion
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class FactorStats:
    factor_name: str
    mean_rank_ic: float
    rank_ic_std: float
    rank_ic_ir: float
    rank_ic_win_rate: float
    valid_days: int


def _safe_float(v: float) -> float:
    try:
        x = float(v)
    except Exception:
        return np.nan
    if np.isfinite(x):
        return x
    return np.nan


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


def infer_factor_name(path: Path) -> str:
    stem = path.stem
    return stem.replace("_rankic", "")


def resolve_factor_name(candidate: str, baseline_names: List[str]) -> str:
    """Try to map file-derived name to baseline naming style."""
    if candidate in baseline_names:
        return candidate
    # Prefer exact suffix matches, e.g. alpha101_WorldQuant_alpha016 -> WorldQuant_alpha016
    suffix_matches = [x for x in baseline_names if candidate.endswith(x)]
    if suffix_matches:
        return sorted(suffix_matches, key=len, reverse=True)[0]
    contains_matches = [x for x in baseline_names if x in candidate]
    if contains_matches:
        return sorted(contains_matches, key=len, reverse=True)[0]
    return candidate


def load_rankic_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "rank_ic"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in {path}: {sorted(miss)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    out["rank_ic"] = pd.to_numeric(out["rank_ic"], errors="coerce")
    out = out.dropna(subset=["rank_ic"])
    return out


def calc_stats(factor_name: str, daily_df: pd.DataFrame) -> FactorStats:
    s = daily_df["rank_ic"].astype(float)
    mean_rank_ic = _safe_float(s.mean())
    rank_ic_std = _safe_float(s.std())
    rank_ic_ir = _safe_float(mean_rank_ic / rank_ic_std) if rank_ic_std and rank_ic_std > 0 else np.nan
    rank_ic_win_rate = _safe_float((s > 0).mean())
    valid_days = int(s.notna().sum())
    return FactorStats(
        factor_name=factor_name,
        mean_rank_ic=mean_rank_ic,
        rank_ic_std=rank_ic_std,
        rank_ic_ir=rank_ic_ir,
        rank_ic_win_rate=rank_ic_win_rate,
        valid_days=valid_days,
    )


def load_baseline_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"factor_name", "mean_rank_ic", "rank_ic_std", "rank_ic_ir", "rank_ic_win_rate", "valid_days"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Baseline summary missing columns: {sorted(miss)}")
    out = df.copy()
    for c in ["mean_rank_ic", "rank_ic_std", "rank_ic_ir", "rank_ic_win_rate", "valid_days"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[out["factor_name"].notna()].copy()
    out = out[out["rank_ic_ir"].notna()].copy()
    return out


def rank_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.sort_values("rank_ic_ir", ascending=False).reset_index(drop=True).copy()
    out["rank_ir"] = np.arange(1, len(out) + 1)
    return out


def find_daily_rankic_file(daily_dir: Path, factor_name: str) -> Optional[Path]:
    candidates = sorted(daily_dir.glob(f"*{factor_name}*_rankic.csv"))
    if not candidates:
        return None
    exact = [p for p in candidates if p.stem.endswith(f"{factor_name}_rankic")]
    return exact[0] if exact else candidates[0]


def build_correlation_matrix(
    new_factor_name: str,
    new_daily: pd.DataFrame,
    baseline_ranked: pd.DataFrame,
    daily_dir: Path,
    top_k: int,
) -> Tuple[pd.DataFrame, List[str]]:
    top = baseline_ranked.head(top_k)["factor_name"].tolist()
    factor_series: Dict[str, pd.Series] = {
        new_factor_name: new_daily.set_index("date")["rank_ic"].rename(new_factor_name)
    }
    loaded: List[str] = []
    for fname in top:
        p = find_daily_rankic_file(daily_dir, fname)
        if p is None:
            continue
        try:
            d = load_rankic_daily(p)
        except Exception:
            continue
        factor_series[fname] = d.set_index("date")["rank_ic"].rename(fname)
        loaded.append(fname)

    merged = pd.concat(factor_series.values(), axis=1, join="inner").dropna(how="any")
    if merged.empty or merged.shape[1] < 2:
        return pd.DataFrame(), loaded
    corr = merged.corr(method="spearman")
    return corr, loaded


def save_heatmap(corr: pd.DataFrame, out_png: Path) -> None:
    if plt is None or corr.empty:
        return
    n = max(6, min(14, corr.shape[0] * 0.8))
    fig, ax = plt.subplots(figsize=(n, n))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Spearman Correlation Heatmap (Rank IC)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def segment_stability(daily: pd.DataFrame, n_segments: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = daily[["date", "rank_ic"]].dropna().sort_values("date").copy()
    if d.empty:
        return pd.DataFrame(), {}
    n_segments = max(2, int(n_segments))
    n_segments = min(n_segments, len(d))
    d["segment"] = pd.qcut(np.arange(len(d)), q=n_segments, labels=False, duplicates="drop") + 1
    d["segment"] = d["segment"].astype(int)

    rows = []
    for seg, g in d.groupby("segment", sort=True):
        s = g["rank_ic"]
        mean_v = _safe_float(s.mean())
        std_v = _safe_float(s.std())
        ir_v = _safe_float(mean_v / std_v) if std_v and std_v > 0 else np.nan
        rows.append(
            {
                "segment": int(seg),
                "start_date": g["date"].min().strftime("%Y-%m-%d"),
                "end_date": g["date"].max().strftime("%Y-%m-%d"),
                "n_days": int(len(g)),
                "mean_rank_ic": mean_v,
                "rank_ic_std": std_v,
                "rank_ic_ir": ir_v,
                "win_rate": _safe_float((s > 0).mean()),
            }
        )
    seg_df = pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)

    means = seg_df["mean_rank_ic"].astype(float)
    metrics = {
        "segment_count": int(len(seg_df)),
        "positive_segment_ratio": _safe_float((means > 0).mean()),
        "segment_mean_std": _safe_float(means.std()),
        "segment_mean_range": _safe_float(means.max() - means.min()),
        "segment_ir_mean": _safe_float(seg_df["rank_ic_ir"].mean()),
        "segment_win_rate_mean": _safe_float(seg_df["win_rate"].mean()),
    }
    return seg_df, metrics


def recommend(
    stats: FactorStats,
    corr: pd.DataFrame,
    stability_metrics: Dict[str, float],
) -> Dict[str, object]:
    corr_max_abs = np.nan
    if not corr.empty and stats.factor_name in corr.index:
        peers = corr.loc[stats.factor_name].drop(labels=[stats.factor_name], errors="ignore")
        if len(peers) > 0:
            corr_max_abs = _safe_float(np.nanmax(np.abs(peers.values)))

    score = 0.0
    reasons: List[str] = []

    if np.isfinite(stats.rank_ic_ir):
        if stats.rank_ic_ir >= 0.5:
            score += 35
            reasons.append("rank_ic_ir >= 0.50")
        elif stats.rank_ic_ir >= 0.3:
            score += 25
            reasons.append("rank_ic_ir in [0.30, 0.50)")
        elif stats.rank_ic_ir >= 0.15:
            score += 15
            reasons.append("rank_ic_ir in [0.15, 0.30)")

    if np.isfinite(stats.mean_rank_ic):
        if stats.mean_rank_ic >= 0.03:
            score += 20
            reasons.append("mean_rank_ic >= 0.03")
        elif stats.mean_rank_ic >= 0.015:
            score += 12
            reasons.append("mean_rank_ic in [0.015, 0.03)")
        elif stats.mean_rank_ic > 0:
            score += 6
            reasons.append("mean_rank_ic > 0")

    if np.isfinite(stats.rank_ic_win_rate):
        if stats.rank_ic_win_rate >= 0.58:
            score += 15
            reasons.append("rank_ic_win_rate >= 58%")
        elif stats.rank_ic_win_rate >= 0.54:
            score += 10
            reasons.append("rank_ic_win_rate in [54%, 58%)")
        elif stats.rank_ic_win_rate >= 0.51:
            score += 5
            reasons.append("rank_ic_win_rate in [51%, 54%)")

    pos_seg = _safe_float(stability_metrics.get("positive_segment_ratio", np.nan))
    seg_range = _safe_float(stability_metrics.get("segment_mean_range", np.nan))
    if np.isfinite(pos_seg):
        if pos_seg >= 0.75:
            score += 12
            reasons.append("positive segment ratio >= 75%")
        elif pos_seg >= 0.5:
            score += 6
            reasons.append("positive segment ratio in [50%, 75%)")
    if np.isfinite(seg_range) and seg_range <= 0.03:
        score += 8
        reasons.append("segment mean range <= 0.03")

    if np.isfinite(corr_max_abs):
        if corr_max_abs <= 0.5:
            score += 10
            reasons.append("max |corr| <= 0.50")
        elif corr_max_abs <= 0.75:
            score += 6
            reasons.append("max |corr| in (0.50, 0.75]")
        elif corr_max_abs >= 0.9:
            score -= 8
            reasons.append("max |corr| >= 0.90 (high redundancy)")

    score = float(max(0.0, min(100.0, score)))
    if score >= 70:
        decision = "in_pool"
    elif score >= 50:
        decision = "watchlist"
    else:
        decision = "eliminate"

    return {
        "decision": decision,
        "score": score,
        "max_abs_corr_with_top": corr_max_abs,
        "reasons": reasons,
        "thresholds": {
            "in_pool": "score >= 70",
            "watchlist": "50 <= score < 70",
            "eliminate": "score < 50",
        },
    }


def main() -> int:
    t0 = time.perf_counter()
    ts0 = pd.Timestamp.now(tz='UTC')
    run_id = f"factor_compare_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Compare new factor with baseline factors and output dashboard artifacts.")
    parser.add_argument("--new-rankic", type=str, required=True, help="Path to new factor daily rankic csv")
    parser.add_argument("--baseline-summary", type=str, default="rankic_batch_results/summary.csv", help="Baseline summary csv")
    parser.add_argument("--baseline-daily-dir", type=str, default="rankic_batch_results", help="Directory of baseline *_rankic.csv")
    parser.add_argument("--factor-name", type=str, default="", help="Optional name for new factor")
    parser.add_argument("--top-k", type=int, default=20, help="Top K baseline factors for correlation heatmap")
    parser.add_argument("--segments", type=int, default=4, help="Number of time segments for stability analysis")
    parser.add_argument("--out-dir", type=str, default="factor_compare_output", help="Output directory")
    parser.add_argument("--metadata-out", type=str, default="", help="Optional metadata json output path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_out = Path(args.metadata_out) if args.metadata_out else (out_dir / "run_metadata.json")
    if not metadata_out.is_absolute():
        metadata_out = (root / metadata_out).resolve()

    def save_meta(status: str, error: str = "", extra: dict | None = None):
        ts1 = pd.Timestamp.now(tz='UTC')
        payload = {
            "run_id": run_id,
            "status": status,
            "script": "scripts/analysis/factor_compare_dashboard.py",
            "started_at_utc": ts0.isoformat(),
            "ended_at_utc": ts1.isoformat(),
            "runtime_sec": float(time.perf_counter() - t0),
            "inputs": {
                "new_rankic": str(args.new_rankic),
                "baseline_summary": str(args.baseline_summary),
                "baseline_daily_dir": str(args.baseline_daily_dir),
                "factor_name": str(args.factor_name),
                "top_k": int(args.top_k),
                "segments": int(args.segments),
                "out_dir": str(out_dir),
            },
            "error": error or None,
        }
        if extra:
            payload.update(extra)
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Run metadata saved to: {metadata_out}")

    try:
        new_rankic_path = Path(args.new_rankic)
        if not new_rankic_path.is_absolute():
            new_rankic_path = (root / new_rankic_path).resolve()
        if not new_rankic_path.exists():
            raise FileNotFoundError(f"New factor rankic csv not found: {new_rankic_path}")

        baseline_summary_path = Path(args.baseline_summary)
        if not baseline_summary_path.is_absolute():
            baseline_summary_path = (root / baseline_summary_path).resolve()
        if not baseline_summary_path.exists():
            raise FileNotFoundError(f"Baseline summary not found: {baseline_summary_path}")

        baseline_daily_dir = Path(args.baseline_daily_dir)
        if not baseline_daily_dir.is_absolute():
            baseline_daily_dir = (root / baseline_daily_dir).resolve()
        if not baseline_daily_dir.exists():
            raise FileNotFoundError(f"Baseline daily dir not found: {baseline_daily_dir}")

        baseline_summary = load_baseline_summary(baseline_summary_path)
        new_daily = load_rankic_daily(new_rankic_path)
        raw_name = args.factor_name.strip() or infer_factor_name(new_rankic_path)
        factor_name = resolve_factor_name(raw_name, baseline_summary["factor_name"].astype(str).tolist())
        new_stats = calc_stats(factor_name, new_daily)
        before_rank = rank_table(baseline_summary)
        existed_before = factor_name in set(before_rank["factor_name"].astype(str))
        old_rank = None
        if existed_before:
            old_rank = int(before_rank.loc[before_rank["factor_name"] == factor_name, "rank_ir"].iloc[0])

        row_data = {c: np.nan for c in baseline_summary.columns}
        row_data.update(
            {
                "factor_name": new_stats.factor_name,
                "mean_rank_ic": new_stats.mean_rank_ic,
                "rank_ic_std": new_stats.rank_ic_std,
                "rank_ic_ir": new_stats.rank_ic_ir,
                "rank_ic_win_rate": new_stats.rank_ic_win_rate,
                "valid_days": new_stats.valid_days,
            }
        )
        row = pd.DataFrame([row_data], columns=baseline_summary.columns)
        merged = baseline_summary[baseline_summary["factor_name"] != factor_name].copy()
        merged = pd.concat([merged, row], ignore_index=True)
        after_rank = rank_table(merged)
        new_rank = int(after_rank.loc[after_rank["factor_name"] == factor_name, "rank_ir"].iloc[0])

        ranking_change = pd.DataFrame(
            [
                {
                    "factor_name": factor_name,
                    "old_rank_ir": old_rank,
                    "new_rank_ir": new_rank,
                    "rank_change": (old_rank - new_rank) if old_rank is not None else np.nan,
                    "mean_rank_ic": new_stats.mean_rank_ic,
                    "rank_ic_std": new_stats.rank_ic_std,
                    "rank_ic_ir": new_stats.rank_ic_ir,
                    "rank_ic_win_rate": new_stats.rank_ic_win_rate,
                    "valid_days": new_stats.valid_days,
                }
            ]
        )
        ranking_change.to_csv(out_dir / "ranking_change_new_factor.csv", index=False, encoding="utf-8-sig")

        before_simple = before_rank[["factor_name", "rank_ir", "rank_ic_ir"]].rename(
            columns={"rank_ir": "rank_before", "rank_ic_ir": "rank_ic_ir_before"}
        )
        after_simple = after_rank[["factor_name", "rank_ir", "rank_ic_ir"]].rename(
            columns={"rank_ir": "rank_after", "rank_ic_ir": "rank_ic_ir_after"}
        )
        delta = before_simple.merge(after_simple, on="factor_name", how="outer")
        delta["rank_shift"] = delta["rank_before"] - delta["rank_after"]
        impacted = delta[delta["rank_shift"].fillna(0) != 0].sort_values("rank_after", na_position="last")
        impacted.to_csv(out_dir / "ranking_shift_impacted.csv", index=False, encoding="utf-8-sig")
        after_rank.to_csv(out_dir / "ranking_after_merge.csv", index=False, encoding="utf-8-sig")

        corr, loaded_top = build_correlation_matrix(
            new_factor_name=factor_name,
            new_daily=new_daily,
            baseline_ranked=before_rank,
            daily_dir=baseline_daily_dir,
            top_k=max(1, int(args.top_k)),
        )
        corr_csv = out_dir / "correlation_top_factors.csv"
        corr.to_csv(corr_csv, encoding="utf-8-sig")
        save_heatmap(corr, out_dir / "correlation_top_factors_heatmap.png")

        seg_df, seg_metrics = segment_stability(new_daily, n_segments=args.segments)
        seg_df.to_csv(out_dir / "stability_segments.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([seg_metrics]).to_csv(out_dir / "stability_metrics.csv", index=False, encoding="utf-8-sig")

        recommendation = recommend(new_stats, corr, seg_metrics)
        recommendation_payload = {
            "factor_name": factor_name,
            "new_factor_stats": {
                "mean_rank_ic": new_stats.mean_rank_ic,
                "rank_ic_std": new_stats.rank_ic_std,
                "rank_ic_ir": new_stats.rank_ic_ir,
                "rank_ic_win_rate": new_stats.rank_ic_win_rate,
                "valid_days": new_stats.valid_days,
            },
            "ranking_change": {
                "old_rank_ir": old_rank,
                "new_rank_ir": new_rank,
                "rank_change": (old_rank - new_rank) if old_rank is not None else None,
            },
            "stability_metrics": seg_metrics,
            "correlation": {
                "top_loaded_count": len(loaded_top),
                "top_loaded_factors": loaded_top,
            },
            "recommendation": recommendation,
        }
        (out_dir / "recommendation.json").write_text(
            json.dumps(recommendation_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        save_meta(
            "success",
            extra={
                "factor_name": factor_name,
                "metrics": {
                    "mean_rank_ic": new_stats.mean_rank_ic,
                    "rank_ic_ir": new_stats.rank_ic_ir,
                    "rank_ic_win_rate": new_stats.rank_ic_win_rate,
                    "valid_days": new_stats.valid_days,
                    "new_rank_ir": int(new_rank),
                    "old_rank_ir": int(old_rank) if old_rank is not None else None,
                },
                "recommendation": recommendation,
                "artifacts": {
                    "output_dir": str(out_dir),
                    "ranking_change_csv": str((out_dir / "ranking_change_new_factor.csv").resolve()),
                    "correlation_csv": str((out_dir / "correlation_top_factors.csv").resolve()),
                    "stability_csv": str((out_dir / "stability_segments.csv").resolve()),
                    "recommendation_json": str((out_dir / "recommendation.json").resolve()),
                },
            },
        )

        print("Compare dashboard completed.")
        print(f"Output dir: {out_dir}")
        print(f"Factor: {factor_name}")
        print(f"Rank IC/IR rank: old={old_rank} -> new={new_rank}")
        print(f"Decision: {recommendation['decision']} (score={recommendation['score']:.1f})")
        print("Artifacts:")
        print("  ranking_change_new_factor.csv")
        print("  ranking_shift_impacted.csv")
        print("  ranking_after_merge.csv")
        print("  correlation_top_factors.csv")
        print("  correlation_top_factors_heatmap.png")
        print("  stability_segments.csv")
        print("  stability_metrics.csv")
        print("  recommendation.json")
        return 0
    except Exception as e:
        save_meta("failed", error=str(e), extra={"artifacts": {"output_dir": str(out_dir)}})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

