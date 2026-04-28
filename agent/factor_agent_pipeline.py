#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After LLM generates factor code, load it, save .pkl, run scripts/analysis/factor_rankic_analysis.py
(Rank IC + optional decile backtest + optional cumulative plots).

Project root = parent of agent/ (e:\\data). Run:
  cd /d e:\\data
  python agent/factor_agent_pipeline.py -d "20-day momentum from close" --data data.pkl
  python agent/factor_agent_pipeline.py --preset volatility_20d --artifact-dir agent_runs/smoke

Requires generated module to define:
  def compute_factor_df() -> pd.DataFrame
  MultiIndex (order_book_id, date) with exactly one factor column (float), or
  columns order_book_id, date, <factor>.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))


def ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        if df.shape[1] != 1:
            raise ValueError("MultiIndex factor frame must have exactly one column.")
        out = df.copy()
        out = out.reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["order_book_id"] = out["order_book_id"].astype(str)
        fc = [c for c in out.columns if c not in ("order_book_id", "date")][0]
        return out.set_index(["order_book_id", "date"])[[fc]].sort_index()
    need = {"order_book_id", "date"}
    if not need.issubset(df.columns):
        raise ValueError(f"Wide format needs columns {need}, got {list(df.columns)}")
    fac_cols = [c for c in df.columns if c not in need]
    if len(fac_cols) != 1:
        raise ValueError(f"Expected one factor column besides order_book_id,date; got {fac_cols}")
    fc = fac_cols[0]
    out = (
        df[list(need) + [fc]]
        .dropna(subset=[fc])
        .assign(
            date=lambda x: pd.to_datetime(x["date"]).dt.normalize(),
            order_book_id=lambda x: x["order_book_id"].astype(str),
        )
    )
    return out.set_index(["order_book_id", "date"])[[fc]].sort_index()


def load_generated_module(py_path: Path):
    spec = importlib.util.spec_from_file_location("_factor_gen_mod", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_factor_gen_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_compute_and_pickle(py_path: Path, pkl_path: Path) -> str:
    mod = load_generated_module(py_path)
    if not hasattr(mod, "compute_factor_df"):
        raise RuntimeError(
            "Generated file must define compute_factor_df() -> pandas.DataFrame. "
            "Regenerate with: python agent/factor_code_agent.py ... (updated prompt)."
        )
    df = mod.compute_factor_df()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("compute_factor_df() must return pandas.DataFrame")
    fac = ensure_multiindex(df)
    fac.columns = [str(fac.columns[0])]
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    fac.to_pickle(pkl_path)
    print(f"Saved factor pickle: {pkl_path} shape={fac.shape} col={fac.columns[0]!r}")
    return str(fac.columns[0])


VOL_TEMPLATE = AGENT_DIR / "templates" / "factor_volatility_20d_from_datapkl.py"


def run_rank_ic_backtest(
    factor_pkl: Path,
    data_pkl: Path,
    out_csv: Path | None,
    next_day: bool,
    backtest_decile: bool,
    plot_backtest: bool,
    backtest_output_dir: Path | None,
    plot_output_dir: Path | None,
    workers: int | None,
    backend: str,
    device: str,
) -> int:
    ana = ROOT / "scripts" / "analysis" / "factor_rankic_analysis.py"
    cmd = [
        sys.executable,
        str(ana),
        "--factor",
        str(factor_pkl.resolve()),
        "--data",
        str(data_pkl.resolve()),
    ]
    if next_day:
        cmd.append("--next-day")
    if out_csv:
        cmd.extend(["--output", str(out_csv)])
    if backtest_decile:
        cmd.append("--backtest-decile")
    if plot_backtest and backtest_decile:
        cmd.append("--plot-backtest")
    if backtest_output_dir is not None:
        cmd.extend(["--backtest-output-dir", str(backtest_output_dir)])
    if plot_output_dir is not None:
        cmd.extend(["--plot-output-dir", str(plot_output_dir)])
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    if backend:
        cmd.extend(["--backend", str(backend)])
    if device:
        cmd.extend(["--device", str(device)])
    print("Running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main():
    p = argparse.ArgumentParser(description="Generate factor (via OpenAI) + pickle + rank IC backtest")
    p.add_argument("-d", "--describe", type=str, default="", help="Natural language factor description")
    p.add_argument(
        "--preset",
        type=str,
        choices=["volatility_20d"],
        default=None,
        help="Skip LLM: use bundled template (e.g. 20-day return volatility from data.pkl)",
    )
    p.add_argument("--data", type=str, default="data.pkl", help="Market pickle for factor_rankic_analysis")
    p.add_argument("--model", type=str, default="gpt-4.1")
    p.add_argument("--py-out", type=str, default="", help="Save generated .py path (default generated_factors/...)")
    p.add_argument("--pkl-out", type=str, default="", help="Save factor .pkl path (default beside .py)")
    p.add_argument("--rankic-out", type=str, default="", help="Rank IC csv output path")
    p.add_argument("--skip-generate", type=str, default="", help="Skip LLM; use existing .py path")
    p.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="If set, write rank_ic / backtest_results / backtest_plots under this directory",
    )
    p.add_argument("--no-next-day", action="store_true", help="Use same-day return in IC (default next-day)")
    p.add_argument("--no-backtest", action="store_true", help="Skip decile backtest step")
    p.add_argument("--no-plot-backtest", action="store_true", help="Skip decile cumulative plots (default: plot when backtest runs)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--backend", type=str, default="pandas", choices=["pandas", "torch"], help="IC backend")
    p.add_argument("--device", type=str, default="auto", help="Torch device when --backend torch")
    args = p.parse_args()

    skip = args.skip_generate.strip()
    has_d = bool(args.describe.strip())
    if skip:
        if args.preset is not None or has_d:
            p.error("Use only one of: --skip-generate, --preset, -d")
    elif args.preset is not None:
        if has_d:
            p.error("Do not combine --preset with -d")
    elif not has_d:
        p.error("Provide -d/--describe, --preset volatility_20d, or --skip-generate PATH")

    os.chdir(ROOT)

    from factor_code_agent import (  # noqa: E402
        default_user_prefix,
        call_openai,
        extract_python_code,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if skip:
        py_path = Path(skip)
        if not py_path.is_file():
            print(f"Not found: {py_path}", file=sys.stderr)
            return 1
    elif args.preset == "volatility_20d":
        if not VOL_TEMPLATE.is_file():
            print(f"Missing template {VOL_TEMPLATE}", file=sys.stderr)
            return 1
        gen_dir = ROOT / "generated_factors"
        gen_dir.mkdir(parents=True, exist_ok=True)
        py_path = Path(args.py_out) if args.py_out else gen_dir / f"{ts}_volatility_20d.py"
        py_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(VOL_TEMPLATE, py_path)
        hdr = f"# agent/factor_agent_pipeline.py preset volatility_20d — {datetime.now().isoformat(timespec='seconds')}\n\n"
        py_path.write_text(hdr + py_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Preset copied to: {py_path}")
    else:
        print("Calling OpenAI to generate factor code...", flush=True)
        raw = call_openai(default_user_prefix(args.describe), args.model)
        code = extract_python_code(raw)
        gen_dir = ROOT / "generated_factors"
        gen_dir.mkdir(parents=True, exist_ok=True)
        py_path = Path(args.py_out) if args.py_out else gen_dir / f"{ts}_factor_gen.py"
        py_path.parent.mkdir(parents=True, exist_ok=True)
        hdr = (
            f"# Generated by agent/factor_agent_pipeline.py — {datetime.now().isoformat(timespec='seconds')}\n"
            f"# Model: {args.model}\n\n"
        )
        py_path.write_text(hdr + code, encoding="utf-8")
        print(f"Saved: {py_path}")

    pkl_path = Path(args.pkl_out) if args.pkl_out else py_path.with_suffix(".pkl")
    try:
        run_compute_and_pickle(py_path, pkl_path)
    except Exception as e:
        print(f"compute_factor_df failed: {e}", file=sys.stderr)
        print("Fix the generated .py or rqdatac data access, then:", file=sys.stderr)
        print(f"  python agent/factor_agent_pipeline.py --skip-generate {py_path} --data {args.data}", file=sys.stderr)
        return 1

    data_pkl = Path(args.data)
    if not data_pkl.is_file():
        print(
            f"Missing market file {data_pkl.resolve()}: "
            "Rank IC needs data.pkl (columns: order_book_id, date, close, ST, limit_up_flag, ...).",
            file=sys.stderr,
        )
        print("Factor pickle is saved; run rank IC manually when data.pkl exists.", file=sys.stderr)
        return 0

    artifact = Path(args.artifact_dir) if args.artifact_dir.strip() else None
    if artifact is not None:
        artifact = (ROOT / artifact).resolve() if not artifact.is_absolute() else artifact
        artifact.mkdir(parents=True, exist_ok=True)
        rankic_out = Path(args.rankic_out) if args.rankic_out else artifact / f"{pkl_path.stem}_rankic.csv"
        bt_out = artifact / "backtest_results"
        plot_out = artifact / "backtest_plots"
    else:
        rankic_out = Path(args.rankic_out) if args.rankic_out else ROOT / "rankic_batch_results" / f"{pkl_path.stem}_rankic.csv"
        bt_out = None
        plot_out = None

    rankic_out.parent.mkdir(parents=True, exist_ok=True)
    do_bt = not args.no_backtest
    do_plot = do_bt and not args.no_plot_backtest
    rc = run_rank_ic_backtest(
        pkl_path,
        data_pkl,
        rankic_out,
        next_day=not args.no_next_day,
        backtest_decile=do_bt,
        plot_backtest=do_plot,
        backtest_output_dir=bt_out,
        plot_output_dir=plot_out,
        workers=args.workers,
        backend=args.backend,
        device=args.device,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
