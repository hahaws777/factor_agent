#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Research Orchestrator — end-to-end quant pipeline in one command.

Subcommands
-----------
run         Generate (or load) a factor → IC decay → quality screen → report
screen-dir  Batch Rank IC + grade all factors in a directory → leaderboard
decay       IC decay analysis for a single existing factor
leaderboard Rank all previously screened factors from a results dir
compare     Side-by-side metrics for multiple factor CSVs

Examples
--------
  # Full pipeline from natural language
  python agent/factor_orchestrator.py run -d "20-day momentum" --data data.pkl

  # Use an existing factor file
  python agent/factor_orchestrator.py run --factor-pkl factors_by_type/alpha101/WorldQuant_alpha002.pkl

  # Screen an entire factor library
  python agent/factor_orchestrator.py screen-dir factors_by_type/alpha101 --data data.pkl

  # IC decay for one factor
  python agent/factor_orchestrator.py decay --factor-pkl my_factor.pkl

  # Leaderboard across all screening runs
  python agent/factor_orchestrator.py leaderboard --results-dir screening_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

log = logging.getLogger(__name__)

REGISTRY_PATH = ROOT / "agent_runs" / "factor_registry.json"


# ── Factor Registry ───────────────────────────────────────────────────────────

def _load_registry() -> list[dict]:
    if REGISTRY_PATH.is_file():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_registry(entries: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2, default=str),
                              encoding="utf-8")


def _upsert_registry(entry: dict) -> None:
    entries = _load_registry()
    name = entry.get("factor_name", "")
    entries = [e for e in entries if e.get("factor_name") != name]
    entry["updated_at"] = datetime.utcnow().isoformat()
    entries.append(entry)
    _save_registry(entries)
    log.info("Registry updated for: %s", name)


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_report(
    factor_name: str,
    screening: dict,
    decay: dict | None,
    artifact_dir: Path,
    extra_notes: str = "",
) -> Path:
    from factor_screener import format_screening_report

    report_path = artifact_dir / f"{factor_name}_report.md"
    lines = [
        f"# Factor Report: {factor_name}",
        f"",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Artifact dir:** `{artifact_dir}`",
        f"",
        f"## Quality Screening",
        f"",
        f"```",
        format_screening_report(screening, decay),
        f"```",
        f"",
    ]

    metrics = screening.get("metrics", {})
    if metrics:
        lines += [
            "## Metrics Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
            lines.append(f"| {k} | {val_str} |")
        lines.append("")

    if decay and "horizons_tested" in decay:
        lines += [
            "## IC Decay",
            "",
            f"- Decay grade: **{decay.get('decay_grade', '?')}**",
            f"- Half-life: **{decay.get('half_life_days', 'N/A')} trading days**",
            f"- Peak Rank IC: **{decay.get('peak_rank_ic', 0):.4f}**",
            f"- Horizons tested: {decay['horizons_tested']}",
            "",
        ]
        decay_png = artifact_dir / f"{factor_name}_ic_decay.png"
        if decay_png.is_file():
            lines.append(f"![IC Decay]({decay_png.name})")
            lines.append("")

    if extra_notes:
        lines += ["## Notes", "", extra_notes, ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to: %s", report_path)
    return report_path


# ── Telegram notification (optional) ─────────────────────────────���───────────

def _notify(message: str) -> None:
    """Send a Telegram message if bot token + chat ID are set in environment."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    try:
        import urllib.request, urllib.parse
        body = json.dumps({"chat_id": chat_id, "text": message}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        log.info("Telegram notification sent.")
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Full pipeline: generate or load factor → IC + decay → screen → report."""
    from factor_screener import screen_ic_csv, screen_decay_csv
    from ic_decay_analysis import ICDecayAnalyzer

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ROOT / "agent_runs" / f"orch_{ts}"
    if not artifact_dir.is_absolute():
        artifact_dir = (ROOT / artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: obtain factor pickle ────────────────────��─────────────────────
    if args.factor_pkl:
        factor_pkl = Path(args.factor_pkl)
        if not factor_pkl.is_file():
            log.error("Factor file not found: %s", factor_pkl)
            return 1
        factor_name = factor_pkl.stem
        log.info("Using existing factor: %s", factor_pkl)
    else:
        if not args.describe:
            log.error("Provide --describe or --factor-pkl")
            return 1
        log.info("Generating factor from description: %r", args.describe)
        pipe_script = AGENT_DIR / "factor_agent_pipeline.py"
        cmd = [
            sys.executable, str(pipe_script),
            "-d", args.describe,
            "--data", args.data,
            "--artifact-dir", str(artifact_dir),
            "--provider", args.provider,
            "--model", args.model,
            "--no-backtest",
        ]
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            log.error("Factor generation pipeline failed (exit %d)", rc)
            return rc
        # Find the generated pkl
        pkls = sorted(artifact_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pkls:
            pkls = sorted((ROOT / "generated_factors").glob("*.pkl"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not pkls:
            log.error("Could not find generated factor pkl")
            return 1
        factor_pkl = pkls[0]
        factor_name = factor_pkl.stem
        log.info("Generated factor pickle: %s", factor_pkl)

    # ── Step 2: optional neutralization ──────────────────────────────────────
    if args.neutralize:
        log.info("Applying neutralization: %s", args.neutralize)
        from factor_neutralizer import neutralize_from_file
        neutral_pkl = artifact_dir / f"{factor_name}_neutral.pkl"
        try:
            neutralize_from_file(
                str(factor_pkl), args.data,
                methods=args.neutralize.split(","),
                output_pkl=str(neutral_pkl),
            )
            factor_pkl = neutral_pkl
            factor_name = factor_pkl.stem
        except Exception as e:
            log.warning("Neutralization failed (%s) — continuing with raw factor", e)

    # ── Step 3: Rank IC (1-day) ───────────────────────────────────────────────
    rankic_csv = artifact_dir / f"{factor_name}_rankic.csv"
    ana_script = ROOT / "scripts" / "analysis" / "factor_rankic_analysis.py"
    rc = subprocess.call([
        sys.executable, str(ana_script),
        "--factor", str(factor_pkl),
        "--data", args.data,
        "--output", str(rankic_csv),
        "--next-day",
    ], cwd=str(ROOT))
    if rc != 0:
        log.warning("Rank IC analysis returned non-zero exit %d — continuing", rc)

    # ── Step 4: IC Decay ──────────────────────────────────────────────────────
    decay_csv = artifact_dir / f"{factor_name}_ic_decay.csv"
    decay_result = {}
    try:
        decay_ana = ICDecayAnalyzer(args.data)
        decay_ana.load_market_data().load_factor(str(factor_pkl))
        decay_ana.compute_decay(horizons=args.horizons)
        decay_ana.save(str(decay_csv))
        decay_ana.plot(output_dir=str(artifact_dir), prefix=factor_name)
        decay_result = screen_decay_csv(str(decay_csv)) if decay_csv.is_file() else {}
        print(decay_ana.summary_str())
    except Exception as e:
        log.warning("IC decay failed: %s", e)

    # ── Step 5: Screen ────────────────────────────────────────────────────────
    screening = {}
    if rankic_csv.is_file():
        screening = screen_ic_csv(str(rankic_csv), factor_name=factor_name)
        from factor_screener import format_screening_report, save_screening_result
        print(format_screening_report(screening, decay_result or None))
        save_screening_result(screening, str(artifact_dir / f"{factor_name}_screening.json"))
    else:
        log.warning("No Rank IC CSV found — skipping screening")

    # ── Step 6: Report ────────────────────────────────────────────────────────
    report_path = _write_report(factor_name, screening, decay_result or None, artifact_dir)
    log.info("Full report: %s", report_path)

    # ── Step 7: Registry ──────────────────────────────────────────────────────
    registry_entry = {
        "factor_name": factor_name,
        "factor_pkl": str(factor_pkl),
        "grade": screening.get("grade", "?"),
        "quality_score": screening.get("score"),
        "decay_grade": decay_result.get("decay_grade"),
        "half_life_days": decay_result.get("half_life_days"),
        "metrics": screening.get("metrics", {}),
        "artifact_dir": str(artifact_dir),
        "describe": getattr(args, "describe", ""),
        "run_at": datetime.utcnow().isoformat(),
    }
    _upsert_registry(registry_entry)

    # ── Step 8: Notify ────────────────────────────────────────────────────────
    grade = screening.get("grade", "?")
    msg = (
        f"Factor run complete: {factor_name}\n"
        f"Grade: {grade}  ({screening.get('score', '?')}/{screening.get('max_score', '?')} criteria)\n"
        f"Mean Rank IC: {screening.get('metrics', {}).get('mean_rank_ic', float('nan')):.4f}\n"
        f"Rank IC IR: {screening.get('metrics', {}).get('rank_ic_ir', float('nan')):.3f}\n"
        f"Decay grade: {decay_result.get('decay_grade', 'N/A')}  "
        f"half-life: {decay_result.get('half_life_days', 'N/A')}d\n"
        f"Report: {report_path}"
    )
    _notify(msg)
    print(f"\nDone. Artifacts in: {artifact_dir}")
    return 0


def cmd_screen_dir(args: argparse.Namespace) -> int:
    """Batch Rank IC + grade all factors in a directory."""
    from factor_screener import screen_batch_summary, GRADE_MAP

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "screening_results" / ts
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "batch_summary.csv"
    batch_script = ROOT / "scripts" / "analysis" / "batch_factor_analysis.py"
    cmd = [
        sys.executable, str(batch_script),
        args.factor_dir,
        "--output-dir", str(out_dir),
        "--data", args.data,
        "--factor-workers", str(args.factor_workers),
        "--backend", args.backend,
        "--metadata-out", str(out_dir / "run_metadata.json"),
    ]
    if args.ic_workers is not None:
        cmd += ["--ic-workers", str(args.ic_workers)]
    if args.skip_existing:
        cmd.append("--skip-existing")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        log.error("Batch analysis failed (exit %d)", rc)
        return rc

    # Grade
    if not summary_csv.is_file():
        log.error("No summary.csv produced by batch analysis")
        return 1

    graded = screen_batch_summary(str(summary_csv))
    graded_path = out_dir / "leaderboard.csv"
    graded.sort_values("quality_score", ascending=False).to_csv(
        graded_path, index=False, encoding="utf-8-sig"
    )
    log.info("Leaderboard saved: %s", graded_path)

    # Summary stats
    grade_counts = graded["grade"].value_counts().to_dict()
    top5 = graded.sort_values("quality_score", ascending=False).head(5)
    cols = [c for c in ["factor_name", "mean_rank_ic", "rank_ic_ir", "grade", "quality_score"] if c in top5.columns]

    print("\n" + "="*60)
    print(f"Screen results: {len(graded)} factors")
    for g in sorted(GRADE_MAP.values(), key=lambda x: ["A","B","C","D","F"].index(x)):
        print(f"  Grade {g}: {grade_counts.get(g, 0)}")
    print("\nTop 5 factors:")
    print(top5[cols].to_string(index=False))

    # Update registry for each factor
    for _, row in graded.iterrows():
        ic_csv = out_dir / f"{row.get('factor_name', '')}_rankic.csv"
        _upsert_registry({
            "factor_name": row.get("factor_name"),
            "grade": row.get("grade"),
            "quality_score": row.get("quality_score"),
            "metrics": {k: row.get(k) for k in ["mean_rank_ic", "rank_ic_ir", "rank_ic_win_rate", "valid_days"]},
            "factor_dir": args.factor_dir,
            "artifact_dir": str(out_dir),
        })

    msg = (
        f"Screen-dir complete: {args.factor_dir}\n"
        f"{len(graded)} factors evaluated\n"
        + "\n".join(f"  Grade {g}: {grade_counts.get(g,0)}" for g in ["A","B","C","D","F"])
        + f"\nLeaderboard: {graded_path}"
    )
    _notify(msg)
    print(f"\nDone. Results in: {out_dir}")
    return 0


def cmd_decay(args: argparse.Namespace) -> int:
    """IC decay for a single factor."""
    from ic_decay_analysis import ICDecayAnalyzer
    from factor_screener import screen_decay_csv

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "ic_decay_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    factor_name = Path(args.factor_pkl).stem

    ana = ICDecayAnalyzer(args.data)
    ana.load_market_data().load_factor(args.factor_pkl)
    ana.compute_decay(horizons=args.horizons)
    decay_csv = out_dir / f"{factor_name}_ic_decay.csv"
    ana.save(str(decay_csv))
    ana.plot(output_dir=str(out_dir), prefix=factor_name)
    print(ana.summary_str())

    if decay_csv.is_file():
        decay = screen_decay_csv(str(decay_csv))
        print(f"\nDecay grade: {decay['decay_grade']}  "
              f"half-life: {decay.get('half_life_days', 'N/A')}d  "
              f"peak Rank IC: {decay.get('peak_rank_ic', 0):.4f}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Print leaderboard from factor registry or a results directory."""
    if args.from_registry:
        entries = _load_registry()
        if not entries:
            print("Factor registry is empty. Run 'factor_orchestrator.py run' first.")
            return 0
        df = pd.DataFrame(entries)
    else:
        results_dir = Path(args.results_dir)
        csvs = list(results_dir.glob("**/leaderboard.csv")) + list(results_dir.glob("**/batch_summary*.csv"))
        if not csvs:
            print(f"No leaderboard CSVs found in {results_dir}")
            return 1
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        df = df.drop_duplicates(subset=["factor_name"], keep="last")

    sort_col = "quality_score" if "quality_score" in df.columns else "rank_ic_ir"
    df = df.sort_values(sort_col, ascending=False, na_position="last")

    show_cols = [c for c in [
        "factor_name", "grade", "quality_score",
        "mean_rank_ic", "rank_ic_ir", "rank_ic_win_rate",
        "decay_grade", "half_life_days",
    ] if c in df.columns]

    n = args.top or len(df)
    print(f"\nTop {n} factors")
    print("="*80)
    print(df[show_cols].head(n).to_string(index=False))

    if args.out:
        df[show_cols].to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nSaved to: {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare multiple factor IC CSVs side by side."""
    from factor_screener import screen_ic_csv

    rows = []
    for path in args.ic_csvs:
        result = screen_ic_csv(path)
        m = result.get("metrics", {})
        rows.append({
            "factor": result.get("factor_name", Path(path).stem),
            "grade": result.get("grade"),
            "mean_rank_ic": m.get("mean_rank_ic"),
            "rank_ic_ir": m.get("rank_ic_ir"),
            "rank_ic_win_rate": m.get("rank_ic_win_rate"),
            "valid_days": m.get("valid_days"),
        })

    df = pd.DataFrame(rows).sort_values("rank_ic_ir", ascending=False, na_position="last")
    print(df.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved to: {args.out}")
    return 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(
        prog="factor_orchestrator",
        description="Factor Research Orchestrator",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── run ───────────────────��─────────────────────────────────���────────────
    p_run = sub.add_parser("run", help="Full pipeline: generate/load → decay → screen → report")
    p_run.add_argument("-d", "--describe", default="", help="Natural language factor description (triggers LLM generation)")
    p_run.add_argument("--factor-pkl", default="", help="Use existing factor .pkl instead of generating")
    p_run.add_argument("--data", default="data.pkl")
    p_run.add_argument("--provider", default="openai", choices=["openai", "anthropic"], help="LLM provider")
    p_run.add_argument("--model", default="gpt-4.1", help="LLM model for generation")
    p_run.add_argument("--artifact-dir", default="", help="Output directory (auto-named if empty)")
    p_run.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    p_run.add_argument("--neutralize", default="", help="Comma-separated neutralization methods: cross_demean,mktcap,industry")

    # ── screen-dir ───────────────────────────────────────────────────────────
    p_sd = sub.add_parser("screen-dir", help="Batch Rank IC + grade all factors in a directory")
    p_sd.add_argument("factor_dir", help="Directory of factor .pkl files")
    p_sd.add_argument("--data", default="data.pkl")
    p_sd.add_argument("--output-dir", default="")
    p_sd.add_argument("--factor-workers", type=int, default=1)
    p_sd.add_argument("--ic-workers", type=int, default=None, help="Per-factor IC worker threads (level-2)")
    p_sd.add_argument("--backend", default="pandas", choices=["pandas", "torch"])
    p_sd.add_argument("--skip-existing", action="store_true")

    # ── decay ────────────────────────────────────────────────────────────────
    p_decay = sub.add_parser("decay", help="IC decay analysis for a single factor")
    p_decay.add_argument("--factor-pkl", required=True)
    p_decay.add_argument("--data", default="data.pkl")
    p_decay.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    p_decay.add_argument("--output-dir", default="")

    # ── leaderboard ──────────────────────────────────────────────────────────
    p_lb = sub.add_parser("leaderboard", help="Rank all evaluated factors")
    p_lb.add_argument("--results-dir", default="screening_results")
    p_lb.add_argument("--from-registry", action="store_true", help="Use local factor registry instead")
    p_lb.add_argument("--top", type=int, default=20)
    p_lb.add_argument("--out", default="", help="Save leaderboard CSV")

    # ── compare ──────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Side-by-side comparison of multiple factors")
    p_cmp.add_argument("ic_csvs", nargs="+", help="Rank IC CSV paths")
    p_cmp.add_argument("--out", default="")

    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "screen-dir": cmd_screen_dir,
        "decay": cmd_decay,
        "leaderboard": cmd_leaderboard,
        "compare": cmd_compare,
    }
    rc = dispatch[args.cmd](args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
