#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor quality screener: grade factors A/B/C/D/F based on IC/IR/win-rate thresholds.

Thresholds are calibrated for Chinese A-share daily factors. Adjust THRESHOLDS
to match your own standards.

Usage:
  from factor_screener import screen_ic_csv, screen_batch_summary, GRADES
  result = screen_ic_csv("rankic_batch_results/alpha002_rankic.csv")
  print(result["grade"], result["detail"])
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Quality thresholds ────────────────────────────────────────────────────────
# Each criterion contributes one "pass" point (max 5).
THRESHOLDS: dict[str, float] = {
    "mean_rank_ic":       0.020,   # mean Rank IC (absolute value)
    "rank_ic_ir":         0.300,   # Rank IC Information Ratio
    "rank_ic_win_rate":   0.520,   # fraction of days with positive Rank IC
    "valid_days":         100,     # minimum number of valid trading days
    "ic_ir":              0.200,   # Pearson IC IR (looser gate)
}

GRADE_MAP = {5: "A", 4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}

# ── Core grading ─────────────────────────────────────────────────────────────

def _grade(score: int) -> str:
    return GRADE_MAP.get(score, "F")


def grade_metrics(metrics: dict, thresholds: dict | None = None) -> dict:
    """
    Score a dict of factor metrics against thresholds.

    Returns dict with keys:
      grade, score, max_score, passed, failed, detail (per-criterion)
    """
    thr = thresholds or THRESHOLDS
    detail: dict[str, dict] = {}
    passed, failed = [], []

    for criterion, threshold in thr.items():
        val = metrics.get(criterion)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            detail[criterion] = {"value": None, "threshold": threshold, "pass": False, "reason": "missing"}
            failed.append(criterion)
            continue

        # For directional criteria use absolute value (some factors are negative)
        if criterion == "mean_rank_ic":
            ok = abs(float(val)) >= threshold
        elif criterion == "rank_ic_win_rate":
            # win_rate should be > 0.5; if mean_rank_ic < 0, win_rate < 0.5 is also acceptable
            mean_ric = metrics.get("mean_rank_ic", 0) or 0
            if float(mean_ric) < 0:
                ok = float(val) <= (1 - threshold)   # negative factor, low win_rate is good
            else:
                ok = float(val) >= threshold
        else:
            ok = float(val) >= threshold

        detail[criterion] = {"value": float(val), "threshold": threshold, "pass": ok}
        (passed if ok else failed).append(criterion)

    score = len(passed)
    return {
        "grade": _grade(score),
        "score": score,
        "max_score": len(thr),
        "passed": passed,
        "failed": failed,
        "detail": detail,
    }


def screen_ic_csv(
    ic_csv_path: str,
    factor_name: str | None = None,
    thresholds: dict | None = None,
) -> dict:
    """
    Screen a single factor from its Rank IC CSV (output of factor_rankic_analysis.py).

    Returns grading dict plus the raw metrics used.
    """
    try:
        df = pd.read_csv(ic_csv_path)
    except Exception as e:
        return {"grade": "F", "score": 0, "error": str(e), "factor_name": factor_name}

    metrics: dict = {}
    if "rank_ic" in df.columns:
        ric = pd.to_numeric(df["rank_ic"], errors="coerce")
        metrics["mean_rank_ic"] = float(ric.mean()) if ric.notna().any() else np.nan
        std = float(ric.std()) if ric.notna().any() else 0
        metrics["rank_ic_ir"] = metrics["mean_rank_ic"] / std if std > 0 else np.nan
        metrics["rank_ic_win_rate"] = float((ric > 0).mean()) if ric.notna().any() else np.nan
        metrics["rank_ic_std"] = std
    if "ic" in df.columns:
        ic = pd.to_numeric(df["ic"], errors="coerce")
        metrics["mean_ic"] = float(ic.mean()) if ic.notna().any() else np.nan
        std_ic = float(ic.std()) if ic.notna().any() else 0
        metrics["ic_ir"] = metrics["mean_ic"] / std_ic if std_ic > 0 else np.nan
    metrics["valid_days"] = len(df)

    result = grade_metrics(metrics, thresholds)
    result["metrics"] = metrics
    result["factor_name"] = factor_name or Path(ic_csv_path).stem
    result["ic_csv"] = str(ic_csv_path)
    return result


def screen_batch_summary(
    summary_csv_path: str,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Screen all factors in a batch summary CSV (output of batch_factor_analysis.py).

    Returns the summary DataFrame with added columns: grade, score, failed_criteria.
    """
    df = pd.read_csv(summary_csv_path)
    thr = thresholds or THRESHOLDS

    grades, scores, faileds = [], [], []
    for _, row in df.iterrows():
        metrics = {k: row.get(k) for k in thr}
        r = grade_metrics(metrics, thr)
        grades.append(r["grade"])
        scores.append(r["score"])
        faileds.append(", ".join(r["failed"]) if r["failed"] else "")

    df = df.copy()
    df["grade"] = grades
    df["quality_score"] = scores
    df["failed_criteria"] = faileds
    return df


def screen_decay_csv(
    decay_csv_path: str,
    primary_horizon: int = 1,
) -> dict:
    """
    Evaluate IC decay pattern.
    A well-decaying factor has peak IC at horizon=1 and decreasing IC for longer horizons.
    Returns: decay_grade (A/B/C/F), half_life (horizon where IC drops to 50% of peak).
    """
    try:
        df = pd.read_csv(decay_csv_path)
    except Exception as e:
        return {"decay_grade": "F", "error": str(e)}

    if "mean_rank_ic" not in df.columns or "horizon" not in df.columns:
        return {"decay_grade": "F", "error": "unexpected CSV format"}

    df = df.sort_values("horizon")
    peak_ric = df["mean_rank_ic"].abs().max()
    if peak_ric == 0 or np.isnan(peak_ric):
        return {"decay_grade": "F", "peak_rank_ic": 0, "half_life": None}

    # Half-life: first horizon where |IC| drops below 50% of peak
    half_life = None
    for _, row in df.iterrows():
        if abs(row["mean_rank_ic"]) < 0.5 * peak_ric:
            half_life = int(row["horizon"])
            break

    # Grade decay by how quickly signal fades
    h1_ric = df.loc[df["horizon"] == 1, "mean_rank_ic"].values
    h1_ric = float(h1_ric[0]) if len(h1_ric) > 0 else np.nan

    if half_life is None:
        decay_grade = "A"   # signal persists across all tested horizons
    elif half_life >= 10:
        decay_grade = "A"
    elif half_life >= 5:
        decay_grade = "B"
    elif half_life >= 2:
        decay_grade = "C"
    else:
        decay_grade = "D"   # very short-lived signal (day-trading only)

    return {
        "decay_grade": decay_grade,
        "peak_rank_ic": float(peak_ric),
        "half_life_days": half_life,
        "h1_rank_ic": float(h1_ric) if not np.isnan(h1_ric) else None,
        "horizons_tested": df["horizon"].tolist(),
    }


# ── Report helpers ────────────────────────────────────────────────────────────

def format_screening_report(result: dict, decay: dict | None = None) -> str:
    """Return a human-readable screening report string."""
    lines = []
    name = result.get("factor_name", "unknown")
    grade = result.get("grade", "?")
    score = result.get("score", 0)
    max_score = result.get("max_score", len(THRESHOLDS))

    lines.append(f"{'='*60}")
    lines.append(f"Factor: {name}")
    lines.append(f"Quality Grade: {grade}  ({score}/{max_score} criteria passed)")
    lines.append(f"{'='*60}")

    metrics = result.get("metrics", {})
    detail = result.get("detail", {})
    for criterion, info in detail.items():
        val = info.get("value")
        thr = info.get("threshold")
        ok = info.get("pass", False)
        flag = "PASS" if ok else "FAIL"
        val_str = f"{val:.4f}" if val is not None and not (isinstance(val, float) and np.isnan(val)) else "N/A"
        lines.append(f"  [{flag}] {criterion:<22} = {val_str}  (threshold: {thr})")

    if decay:
        lines.append("")
        lines.append(f"IC Decay Grade:   {decay.get('decay_grade', '?')}")
        lines.append(f"  Peak Rank IC:   {decay.get('peak_rank_ic', 0):.4f}")
        hl = decay.get("half_life_days")
        lines.append(f"  Half-life:      {hl}d" if hl else "  Half-life:      > all tested horizons")

    if result.get("failed"):
        lines.append(f"\nFailed criteria: {', '.join(result['failed'])}")

    return "\n".join(lines)


def save_screening_result(result: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Make JSON-serialisable
    safe = json.loads(json.dumps(result, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)
    log.info("Screening result saved to: %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Factor Quality Screener")
    p.add_argument("--ic-csv", type=str, default="", help="Single factor Rank IC CSV to screen")
    p.add_argument("--batch-summary", type=str, default="", help="Batch summary CSV from batch_factor_analysis.py")
    p.add_argument("--decay-csv", type=str, default="", help="IC decay CSV from ic_decay_analysis.py")
    p.add_argument("--out", type=str, default="", help="Output path for JSON result or graded CSV")
    args = p.parse_args()

    if args.ic_csv:
        result = screen_ic_csv(args.ic_csv)
        print(format_screening_report(result))
        if args.out:
            save_screening_result(result, args.out)

    if args.batch_summary:
        graded = screen_batch_summary(args.batch_summary)
        graded_sorted = graded.sort_values("quality_score", ascending=False)
        out_path = args.out or args.batch_summary.replace(".csv", "_graded.csv")
        graded_sorted.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Graded summary saved to: {out_path}")
        print("\nTop factors:")
        cols = [c for c in ["factor_name", "mean_rank_ic", "rank_ic_ir", "grade", "quality_score"] if c in graded_sorted.columns]
        print(graded_sorted[cols].head(20).to_string(index=False))

    if args.decay_csv:
        decay = screen_decay_csv(args.decay_csv)
        print(f"\nDecay grade: {decay['decay_grade']}  half-life: {decay.get('half_life_days')}d  "
              f"peak_ric: {decay.get('peak_rank_ic', 0):.4f}")


if __name__ == "__main__":
    main()
