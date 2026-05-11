#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local chat UI for the factor agent (Claude Code–style dark layout).

Run from project root:
  cd e:\\data
  pip install streamlit openai python-dotenv
  streamlit run agent/ui/streamlit_app.py
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    import job_queue as _jq
except Exception:
    _jq = None  # type: ignore[assignment]

import streamlit as st  # noqa: E402
from factor_code_agent import (  # noqa: E402
    SYSTEM_PROMPT,
    extract_python_code,
)

CHAT_SYSTEM = (
    SYSTEM_PROMPT
    + """

## Interactive chat mode (UI)
- You may use short replies in English or 中文 before/after code.
- When you generate or revise factor code, put the **entire** runnable module in **one** markdown ```python code block.
- Data: only `data.pkl` via `Path(__file__).resolve().parents[1] / "data.pkl"`. Do not suggest or use rqdatac/Tushare/cloud APIs.
- Pandas: never index groupby columns with `series.name` (often None). Assign expressions to `df["col_name"]` first, then `groupby("date")["col_name"]`.
- If the user only asks a conceptual question, answer without code.
"""
)

DOTENV = ROOT / ".env"
MINING_DIR = ROOT / "agent_runs" / "mining"
MINING_UI_STATE = MINING_DIR / "ui_state.json"
JOB_LOG_DIR = ROOT / "agent_runs" / "job_logs"
DEFAULT_CHAT_RENDER_LIMIT = 30


def _load_dotenv() -> None:
    if not DOTENV.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(DOTENV)
        return
    except ImportError:
        pass
    for line in DOTENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _bootstrap_api_key() -> tuple[bool, str]:
    """Ensure at least one supported API key (OpenAI or Anthropic) is available."""
    if os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True, ""
    _load_dotenv()
    if os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True, ""
    return False, "Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY found. Set one in .env or environment."


def _inject_theme_css():
    st.markdown(
        """
        <style>
        .stApp { background: #0f0f12; color: #e8e8ed; }
        [data-testid="stSidebar"] { background: #16161d; border-right: 1px solid #2a2a34; }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background: #1a1a22; border-radius: 12px; }
        div[data-testid="stChatMessageContent"] { font-size: 0.95rem; }
        h1, h2, h3 { color: #f0f0f5 !important; }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] { background: #1e1e26 !important; color: #e8e8ed !important; }
        hr { border-color: #2a2a34; }
        .order-board {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 14px 0;
        }
        .order-lane {
            border: 1px solid #303241;
            background: #171922;
            border-radius: 8px;
            padding: 10px;
            min-height: 132px;
        }
        .order-lane h4 {
            margin: 0 0 8px 0;
            color: #f0f0f5;
            font-size: 0.95rem;
            letter-spacing: 0;
        }
        .order-ticket {
            border: 1px solid #3a3d4e;
            background: #20232e;
            border-left: 4px solid #8aa4ff;
            border-radius: 6px;
            padding: 8px;
            margin-top: 7px;
        }
        .order-ticket.running { border-left-color: #f1c75b; }
        .order-ticket.success { border-left-color: #38d47a; }
        .order-ticket.failed { border-left-color: #ff6b6b; }
        .order-ticket.pending { border-left-color: #8aa4ff; }
        .order-ticket.cancelled { border-left-color: #9aa0aa; }
        .ticket-title { font-weight: 650; color: #f4f4f8; font-size: 0.9rem; }
        .ticket-meta { color: #a8adbd; font-size: 0.78rem; margin-top: 3px; overflow-wrap: anywhere; }
        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 12px 0;
        }
        .pipeline-cell {
            border: 1px solid #303241;
            background: #161820;
            border-radius: 8px;
            padding: 10px;
        }
        .pipeline-label { color: #9ca3b5; font-size: 0.78rem; }
        .pipeline-value { color: #f2f3f7; font-size: 1.08rem; font-weight: 700; margin-top: 3px; }
        @media (max-width: 900px) {
            .order-board, .pipeline-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_artifact_dir(artifact_dir: str) -> Path:
    p = Path(artifact_dir)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p.resolve()


def _resolve_existing_path(raw: str) -> Path:
    p = Path(str(raw))
    if p.exists():
        return p
    text = str(raw).replace("\\", "/")
    # Some artifacts are created by Windows Python and store E:/data paths;
    # convert the common project root back to WSL form when the UI runs there.
    for prefix in ("E:/data", "e:/data"):
        if text.startswith(prefix):
            candidate = ROOT / text[len(prefix):].lstrip("/")
            if candidate.exists():
                return candidate
    if not p.is_absolute():
        candidate = ROOT / p
        if candidate.exists():
            return candidate
    return p


def _tail_text(path: Path, max_lines: int = 160, max_bytes: int = 256 * 1024) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_bytes = min(size, int(max_bytes))
            if read_bytes <= 0:
                return ""
            f.seek(size - read_bytes)
            buf = f.read(read_bytes)
        lines = buf.decode("utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"Could not read log: {e}"
    return "\n".join(lines[-max_lines:])


def _process_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _is_miner_pid(pid: int | None) -> bool:
    """Return True only if `pid` is a live process whose cmdline contains 'alpha_miner'."""
    if not pid:
        return False
    try:
        pid = int(pid)
        os.kill(pid, 0)  # raises if process is gone
    except (OSError, ValueError):
        return False
    # /proc is Linux-only; skip the cmdline check on Windows/macOS.
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        return "alpha_miner" in cmdline
    except Exception:
        return True  # can't verify, allow the signal


def _read_mining_ui_state() -> dict:
    if not MINING_UI_STATE.is_file():
        return {}
    try:
        return json.loads(MINING_UI_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_mining_ui_state(run_dir: Path, pid: int | None = None) -> None:
    MINING_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_run_id": run_dir.name,
        "current_run_dir": str(run_dir.resolve()),
        "pid": int(pid) if pid else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    MINING_UI_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mining_run_dirs() -> list[Path]:
    if not MINING_DIR.is_dir():
        return []
    markers = {"checkpoint.json", "ui_alpha_miner.log", "config_used.yaml", "mining_report.md", "top_factors.csv"}
    dirs = []
    for d in MINING_DIR.iterdir():
        if not d.is_dir():
            continue
        if any((d / marker).exists() for marker in markers) or (d / "factors").is_dir():
            dirs.append(d)
    return sorted(
        dirs,
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def _render_pipeline_artifacts(art: Path) -> None:
    """Show Rank IC table summary, charts, and decile plot images after a successful run."""
    import pandas as pd

    st.subheader("Pipeline results")
    try:
        art_rel = art.relative_to(ROOT)
    except ValueError:
        art_rel = art
    st.caption(f"Output folder: `{art_rel}`")
    meta_path = art / "run_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            inputs = meta.get("inputs", {})
            horizon = "same-day diagnostic" if inputs.get("no_next_day") else "next-day forward"
            st.caption(f"Rank IC return horizon: `{horizon}`")
        except Exception:
            pass

    rankic_files = sorted(art.glob("*_rankic.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not rankic_files:
        st.info("No `*_rankic.csv` found in this folder yet.")
    else:
        ic_path = rankic_files[0]
        try:
            ic_rel = ic_path.relative_to(ROOT)
        except ValueError:
            ic_rel = ic_path
        st.markdown(f"**Rank IC file:** `{ic_rel}`")
        try:
            ic_df = pd.read_csv(ic_path, parse_dates=["date"])
        except Exception as e:
            st.warning(f"Could not read Rank IC CSV: {e}")
        else:
            if "rank_ic" in ic_df.columns:
                mean_r = float(ic_df["rank_ic"].mean())
                std_r = float(ic_df["rank_ic"].std())
                ir_r = mean_r / std_r if std_r and std_r > 0 else float("nan")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean Rank IC", f"{mean_r:.4f}")
                c2.metric("Rank IC IR", f"{ir_r:.4f}")
                if "ic" in ic_df.columns:
                    mean_i = float(ic_df["ic"].mean())
                    std_i = float(ic_df["ic"].std())
                    ir_i = mean_i / std_i if std_i and std_i > 0 else float("nan")
                    c3.metric("Mean IC", f"{mean_i:.4f}")
                    c4.metric("IC IR", f"{ir_i:.4f}")
                chart_cols = [c for c in ("rank_ic", "ic") if c in ic_df.columns]
                if chart_cols:
                    plot_df = ic_df.set_index("date")[chart_cols].sort_index()
                    st.markdown("**Daily IC / Rank IC**")
                    st.line_chart(plot_df)
            with st.expander("Rank IC table (first 200 rows)"):
                st.dataframe(ic_df.head(200), width="stretch")

    bt = art / "backtest_results"
    csvs = sorted(bt.glob("*.csv")) if bt.is_dir() else []
    cum_csvs = [p for p in csvs if "cum" in p.name.lower()]
    plot_dir = art / "backtest_plots"
    pngs = sorted(plot_dir.glob("*.png")) if plot_dir.is_dir() else []
    if pngs or cum_csvs:
        st.markdown("**Decile backtest plots**")
        _render_backtest_png_sections(
            plot_dir=plot_dir,
            pngs=pngs,
            cum_csvs=cum_csvs,
            key_prefix=f"{art.name}_pipeline",
        )

    if bt.is_dir():
        if csvs:
            with st.expander("Backtest CSV paths"):
                for c in csvs:
                    try:
                        st.text(str(c.relative_to(ROOT)))
                    except ValueError:
                        st.text(str(c))


def _render_batch_artifacts(out_dir: Path) -> None:
    import pandas as pd

    st.subheader("Batch analysis results")
    try:
        out_rel = out_dir.relative_to(ROOT)
    except ValueError:
        out_rel = out_dir
    st.caption(f"Output folder: `{out_rel}`")
    meta_path = out_dir / "run_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            config = meta.get("config", {})
            horizon = "next-day forward" if config.get("use_next_day_return", True) else "same-day diagnostic"
            st.caption(f"Rank IC return horizon: `{horizon}`")
        except Exception:
            pass
    summary_path = out_dir / "summary.csv"
    if not summary_path.is_file():
        st.info("No `summary.csv` found in this folder yet.")
        return
    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        st.warning(f"Could not read summary.csv: {e}")
        return
    st.markdown("**Batch summary (top by Rank IC IR)**")
    show = df.copy()
    if "rank_ic_ir" in show.columns:
        show["rank_ic_ir"] = pd.to_numeric(show["rank_ic_ir"], errors="coerce")
        show = show.sort_values("rank_ic_ir", ascending=False, na_position="last")
    cols = [c for c in ["factor_name", "mean_rank_ic", "rank_ic_ir", "rank_ic_win_rate", "valid_days", "elapsed_sec", "error"] if c in show.columns]
    if cols:
        st.dataframe(show[cols].head(200), width="stretch")
    else:
        st.dataframe(show.head(200), width="stretch")


def _backtest_factor_prefix(path: Path) -> str:
    name = path.name
    for suffix in (
        "_decile_daily_returns.csv",
        "_decile_cum_returns.csv",
        "_decile_cum_all.png",
        "_decile_cum_LS.png",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _backtest_factor_lookup(art: Path, candidates_df=None) -> dict[str, dict]:
    lookup: dict[str, dict] = {}

    meta_dir = art / "backtest_results"
    if meta_dir.is_dir():
        for meta_path in sorted(meta_dir.glob("*_factor_meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in (meta.get("name"), meta.get("factor_name"), meta_path.name.removesuffix("_factor_meta.json")):
                if key:
                    lookup[str(key)] = meta

    if candidates_df is not None and not getattr(candidates_df, "empty", True):
        for _, row in candidates_df.iterrows():
            row_dict = row.to_dict()
            name = str(row_dict.get("name") or "")
            if name:
                lookup[name] = row_dict
            pkl_path = str(row_dict.get("pkl_path") or "")
            if pkl_path:
                lookup[Path(pkl_path).stem] = row_dict

    return lookup


def _factor_meta_for_path(path: Path, lookup: dict[str, dict]) -> tuple[str, dict]:
    prefix = _backtest_factor_prefix(path)
    meta = lookup.get(prefix, {})
    if not meta:
        for name, candidate in lookup.items():
            if prefix == name or prefix.startswith(f"{name}_"):
                return prefix, candidate
    return prefix, meta


def _render_factor_formula(meta: dict, prefix: str, *, compact: bool = False) -> None:
    expr = str(meta.get("expression") or meta.get("canonical_expression") or "").strip()
    factor_name = str(meta.get("name") or meta.get("factor_name") or prefix)
    family = str(meta.get("family") or "unknown")
    if compact:
        if expr:
            st.caption(f"{factor_name} · {family}")
            st.code(expr, language="text")
        return

    st.markdown(f"**Backtest formula:** `{factor_name}` · `{family}`")
    if expr:
        st.code(expr, language="text")
    else:
        st.warning("Formula metadata not found for this backtest file.")
    if meta.get("economic_hypothesis"):
        st.caption(str(meta.get("economic_hypothesis")))


def _write_long_only_backtest_pngs(cum_csv: Path, plot_dir: Path, prefix: str) -> list[Path]:
    import pandas as pd
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    cum_df = pd.read_csv(cum_csv, parse_dates=["date"]).sort_values("date")
    q_cols = sorted(
        [c for c in cum_df.columns if c.startswith("Q") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    if not q_cols:
        return []

    written: list[Path] = []

    plt.figure(figsize=(12, 6))
    for c in q_cols:
        if c == "Q1":
            color, linewidth, alpha, label = "#d62728", 1.8, 0.95, c
        elif c == q_cols[-1]:
            color, linewidth, alpha, label = "#1f77b4", 2.0, 0.95, c
        else:
            color, linewidth, alpha, label = "#999999", 1.0, 0.75, c
        plt.plot(cum_df["date"], cum_df[c], color=color, linewidth=linewidth, alpha=alpha, label=label)
    plt.title("Cumulative Return (Long-Only Decile Portfolios)", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    out_all = plot_dir / f"{prefix}_decile_long_only_all.png"
    plt.tight_layout()
    plt.savefig(out_all, dpi=150)
    plt.close()
    written.append(out_all)

    top_q = q_cols[-1]
    plt.figure(figsize=(12, 6))
    plt.plot(cum_df["date"], cum_df[top_q], color="#1f77b4", linewidth=2.0, label=f"Long-only {top_q}")
    plt.title(f"Cumulative Return (Long-Only {top_q})", fontsize=13)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_top = plot_dir / f"{prefix}_decile_long_only_{top_q}.png"
    plt.tight_layout()
    plt.savefig(out_top, dpi=150)
    plt.close()
    written.append(out_top)
    return written


def _render_backtest_png_sections(
    *,
    plot_dir: Path,
    pngs: list[Path],
    cum_csvs: list[Path],
    key_prefix: str,
    lookup: dict[str, dict] | None = None,
) -> None:
    lookup = lookup or {}
    long_only_pngs = [p for p in pngs if "long_only" in p.name.lower()]
    long_short_pngs = [
        p for p in pngs
        if ("_ls" in p.name.lower() or "long-short" in p.name.lower())
        and p not in long_only_pngs
    ]
    other_pngs = [p for p in pngs if p not in long_only_pngs and p not in long_short_pngs]

    def _caption_for_png(png: Path) -> str:
        if lookup:
            prefix, meta = _factor_meta_for_path(png, lookup)
            caption_name = str(meta.get("name") or meta.get("factor_name") or prefix)
            return f"{png.name} · {caption_name}"
        return png.name

    def _render_png_grid(title: str, files: list[Path], expanded: bool = True) -> None:
        if not files:
            return
        with st.expander(title, expanded=expanded):
            cols = st.columns(2)
            for i, png in enumerate(files):
                cols[i % 2].image(str(png), caption=_caption_for_png(png), width="stretch")

    _render_png_grid("Long-only decile images", long_only_pngs, expanded=True)
    _render_png_grid("Long-short images", long_short_pngs, expanded=True)
    _render_png_grid("Other backtest images", other_pngs, expanded=False)

    if not long_only_pngs and cum_csvs:
        st.caption("No static long-only PNGs found for this backtest yet.")
        selected_for_png = st.selectbox(
            "Cumulative CSV for static long-only PNG generation",
            [p.name for p in cum_csvs],
            key=f"{key_prefix}_long_only_png_csv",
        )
        source_csv = next(p for p in cum_csvs if p.name == selected_for_png)
        prefix = _backtest_factor_prefix(source_csv)
        if st.button("Generate static long-only PNGs", key=f"{key_prefix}_make_long_only_png", width="stretch"):
            try:
                written = _write_long_only_backtest_pngs(source_csv, plot_dir, prefix)
                if written:
                    st.success(f"Generated {len(written)} long-only PNGs.")
                    st.rerun()
                else:
                    st.warning("No Q1..Qn columns found in the selected cumulative CSV.")
            except Exception as e:
                st.error(f"Could not generate long-only PNGs: {e}")


def _render_backtest_artifacts(art: Path, title: str = "Backtest results", candidates_df=None) -> None:
    import pandas as pd

    bt_dir = art / "backtest_results"
    plot_dir = art / "backtest_plots"
    csvs = sorted(bt_dir.glob("*.csv")) if bt_dir.is_dir() else []
    pngs = sorted(plot_dir.glob("*.png")) if plot_dir.is_dir() else []
    if not csvs and not pngs:
        st.info("No backtest result CSVs or plot images found yet.")
        return

    st.markdown(f"**{title}**")
    lookup = _backtest_factor_lookup(art, candidates_df)
    cum_csvs = [p for p in csvs if "cum" in p.name.lower()]
    daily_csvs = [p for p in csvs if "daily" in p.name.lower()]
    if cum_csvs:
        selected_csv = st.selectbox(
            "Cumulative return CSV",
            [p.name for p in cum_csvs],
            key=f"{art.name}_bt_cum_csv",
        )
        csv_path = next(p for p in cum_csvs if p.name == selected_csv)
        prefix, meta = _factor_meta_for_path(csv_path, lookup)
        _render_factor_formula(meta, prefix)
        try:
            cum_df = pd.read_csv(csv_path, parse_dates=["date"])
            chart_cols = [c for c in cum_df.columns if c != "date"]
            if chart_cols:
                st.line_chart(cum_df.set_index("date")[chart_cols].sort_index())
            with st.expander("Cumulative return table"):
                st.dataframe(cum_df.tail(200), width="stretch", hide_index=True)
        except Exception as e:
            st.warning(f"Could not read backtest CSV {csv_path.name}: {e}")

    if daily_csvs:
        with st.expander("Daily backtest return CSVs"):
            for p in daily_csvs:
                prefix, meta = _factor_meta_for_path(p, lookup)
                st.text(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
                _render_factor_formula(meta, prefix, compact=True)

    if pngs or cum_csvs:
        _render_backtest_png_sections(
            plot_dir=plot_dir,
            pngs=pngs,
            cum_csvs=cum_csvs,
            key_prefix=art.name,
            lookup=lookup,
        )

    if lookup and (csvs or pngs):
        mapped = []
        for p in sorted({_backtest_factor_prefix(x) for x in (csvs + pngs)}):
            meta = lookup.get(p, {})
            if not meta:
                for name, candidate in lookup.items():
                    if p == name or p.startswith(f"{name}_"):
                        meta = candidate
                        break
            mapped.append({
                "backtest_prefix": p,
                "factor": meta.get("name") or meta.get("factor_name") or "",
                "family": meta.get("family") or "",
                "expression": meta.get("expression") or meta.get("canonical_expression") or "",
            })
        with st.expander("Backtest formula map", expanded=False):
            st.dataframe(pd.DataFrame(mapped), width="stretch", hide_index=True)


def _extract_factor_expression_from_py(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    m = re.search(r"FACTOR_EXPRESSION\s*=\s*(['\"])(.*?)\1", text, flags=re.S)
    if m:
        return m.group(2).strip()
    m = re.search(r"Expression:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _factor_artifact_rows(run_dir: Path, candidates_df=None) -> list[dict]:
    by_name: dict[str, dict] = {}
    if candidates_df is not None and not getattr(candidates_df, "empty", True):
        for _, row in candidates_df.iterrows():
            row_dict = row.to_dict()
            name = str(row_dict.get("name") or "")
            if name:
                by_name[name] = row_dict

    factors_dir = run_dir / "factors"
    ic_dir = run_dir / "ic"
    names = set(by_name)
    if factors_dir.is_dir():
        names.update(p.stem for p in factors_dir.glob("*.py"))
        names.update(p.stem for p in factors_dir.glob("*.pkl"))
    if ic_dir.is_dir():
        for p in ic_dir.glob("*_rankic.csv"):
            names.add(p.name.removesuffix("_rankic.csv"))

    rows = []
    for name in sorted(names):
        meta = by_name.get(name, {})
        py_path = factors_dir / f"{name}.py"
        pkl_path = _resolve_existing_path(str(meta.get("pkl_path") or factors_dir / f"{name}.pkl"))
        ic_path = _resolve_existing_path(str(meta.get("ic_csv_path") or ic_dir / f"{name}_rankic.csv"))
        expression = str(meta.get("expression") or meta.get("canonical_expression") or "").strip()
        rows.append({
            "factor": name,
            "family": meta.get("family") or "",
            "grade": meta.get("grade") or "",
            "mean_rank_ic": meta.get("mean_rank_ic"),
            "rank_ic_ir": meta.get("rank_ic_ir"),
            "alpha_direction": meta.get("alpha_direction") or "",
            "expression": expression,
            "py": str(py_path.relative_to(ROOT)) if py_path.is_file() and py_path.is_relative_to(ROOT) else (str(py_path) if py_path.is_file() else ""),
            "pkl": str(pkl_path.relative_to(ROOT)) if pkl_path.is_file() and pkl_path.is_relative_to(ROOT) else (str(pkl_path) if pkl_path.is_file() else ""),
            "ic_csv": str(ic_path.relative_to(ROOT)) if ic_path.is_file() and ic_path.is_relative_to(ROOT) else (str(ic_path) if ic_path.is_file() else ""),
        })
    return rows


def _render_factor_artifacts(run_dir: Path, candidates_df=None, rows: "list[dict] | None" = None) -> None:
    import pandas as pd

    if rows is None:
        rows = _factor_artifact_rows(run_dir, candidates_df)
    if not rows:
        st.info("No factor .py/.pkl/Rank IC artifacts found yet.")
        return

    st.markdown("**Generated factor artifacts**")
    df_art = pd.DataFrame(rows)
    show_cols = [
        "factor", "family", "grade", "mean_rank_ic", "rank_ic_ir",
        "alpha_direction", "expression", "py", "pkl", "ic_csv",
    ]
    st.dataframe(df_art[[c for c in show_cols if c in df_art.columns]], width="stretch", hide_index=True)

    selected = st.selectbox(
        "Open generated factor",
        df_art["factor"].astype(str).tolist(),
        key=f"{run_dir.name}_artifact_factor_select",
    )
    row = df_art[df_art["factor"].astype(str) == selected].iloc[0].to_dict()
    selected_expr = str(row.get("expression") or "")
    py_path = ROOT / str(row.get("py") or "")
    if not selected_expr and py_path.is_file():
        selected_expr = _extract_factor_expression_from_py(py_path)
    st.markdown(f"**DSL formula for `{selected}`**")
    st.code(selected_expr, language="text")

    if py_path.is_file():
        with st.expander("Generated Python module"):
            st.code(py_path.read_text(encoding="utf-8", errors="replace")[:20000], language="python")

    ic_path = ROOT / str(row.get("ic_csv") or "")
    if ic_path.is_file():
        try:
            ic_df = pd.read_csv(ic_path, parse_dates=["date"])
            chart_cols = [c for c in ("rank_ic", "ic") if c in ic_df.columns]
            if chart_cols:
                st.markdown("**Daily IC / Rank IC for selected artifact**")
                st.line_chart(ic_df.set_index("date")[chart_cols].sort_index())
        except Exception as e:
            st.warning(f"Could not read Rank IC CSV: {e}")


def _html_escape(value: object) -> str:
    return (
        str(value if value is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _job_elapsed(job: dict) -> str:
    try:
        start = job.get("started_at") or job.get("created_at")
        end = job.get("finished_at") or datetime.now().astimezone().isoformat(timespec="seconds")
        if not start:
            return ""
        t0 = datetime.fromisoformat(str(start))
        t1 = datetime.fromisoformat(str(end))
        seconds = max(0, int((t1 - t0).total_seconds()))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m {seconds % 60}s"
    except Exception:
        return ""


def _factor_station(row: dict) -> tuple[str, str]:
    has_py = bool(row.get("py"))
    has_pkl = bool(row.get("pkl"))
    has_ic = bool(row.get("ic_csv"))
    if has_ic:
        return "Completed IC", "success"
    if has_pkl:
        return "Factor calculated", "running"
    if has_py:
        return "Formula ticketed", "pending"
    return "Waiting", "pending"


def _render_order_ticket(title: str, status: str, meta: list[str]) -> str:
    meta_html = "".join(f"<div class='ticket-meta'>{_html_escape(x)}</div>" for x in meta if x)
    return (
        f"<div class='order-ticket {_html_escape(status)}'>"
        f"<div class='ticket-title'>{_html_escape(title)}</div>"
        f"{meta_html}"
        "</div>"
    )


def _render_order_lane(title: str, tickets: list[str]) -> str:
    body = "".join(tickets) if tickets else "<div class='ticket-meta'>Empty</div>"
    return f"<div class='order-lane'><h4>{_html_escape(title)}</h4>{body}</div>"


def _render_calculation_pipeline_board(run_dir: Path, config: dict, candidates_df=None, factor_rows: "list[dict] | None" = None) -> None:
    st.markdown("**Alpha mining calculation board**")

    q_enabled = bool(config.get("pipeline_queue_enabled", True))
    outer_workers = int(config.get("outer_workers", 1) or 1)
    queue_size = int(config.get("pipeline_queue_size", 0) or max(2, outer_workers * 2))
    eval_workers = int(config.get("eval_workers", config.get("workers", 1)) or 1)
    if factor_rows is None:
        factor_rows = _factor_artifact_rows(run_dir, candidates_df)
    jobs = _jobs_for_run(run_dir.name, limit=30)

    pending_jobs = [j for j in jobs if j.get("status") == "pending"]
    running_jobs = [j for j in jobs if j.get("status") == "running"]
    done_jobs = [j for j in jobs if j.get("status") == "success"]
    bad_jobs = [j for j in jobs if j.get("status") in {"failed", "cancelled"}]

    accepted = sum(1 for r in factor_rows if r.get("ic_csv"))
    calculating = sum(1 for r in factor_rows if r.get("pkl") and not r.get("ic_csv"))
    ticketed = sum(1 for r in factor_rows if r.get("py") and not r.get("pkl"))
    errored = len(bad_jobs)

    st.markdown(
        f"""
        <div class="pipeline-grid">
          <div class="pipeline-cell"><div class="pipeline-label">Producer</div><div class="pipeline-value">LLM + recipe picker</div></div>
          <div class="pipeline-cell"><div class="pipeline-label">Task queue</div><div class="pipeline-value">{'On' if q_enabled else 'Off'} · cap {queue_size}</div></div>
          <div class="pipeline-cell"><div class="pipeline-label">Consumers</div><div class="pipeline-value">{outer_workers} workers</div></div>
          <div class="pipeline-cell"><div class="pipeline-label">IC calculators</div><div class="pipeline-value">{eval_workers} workers / factor</div></div>
        </div>
        <div class="pipeline-grid">
          <div class="pipeline-cell"><div class="pipeline-label">Formula tasks</div><div class="pipeline-value">{len(factor_rows)}</div></div>
          <div class="pipeline-cell"><div class="pipeline-label">Calculating</div><div class="pipeline-value">{calculating + ticketed}</div></div>
          <div class="pipeline-cell"><div class="pipeline-label">Completed IC</div><div class="pipeline-value">{accepted}</div></div>
          <div class="pipeline-cell"><div class="pipeline-label">Attention</div><div class="pipeline-value">{errored}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lanes = {
        "1. Queued": [
            _render_order_ticket(
                f"Job {j.get('id')} · {j.get('job_type')}",
                str(j.get("status") or "pending"),
                [str(j.get("run_id") or ""), str(j.get("created_at") or "")],
            )
            for j in pending_jobs[:5]
        ],
        "2. Calculating": [
            _render_order_ticket(
                f"Job {j.get('id')} · {j.get('job_type')}",
                str(j.get("status") or "running"),
                [f"PID {j.get('pid')}" if j.get("pid") else "", f"elapsed {_job_elapsed(j)}"],
            )
            for j in running_jobs[:5]
        ],
        "3. Completed": [
            _render_order_ticket(
                str(r.get("factor") or ""),
                _factor_station(r)[1],
                [
                    _factor_station(r)[0],
                    str(r.get("expression") or ""),
                    f"Rank IC {float(r['mean_rank_ic']):.4f}" if r.get("mean_rank_ic") not in (None, "") else "",
                ],
            )
            for r in [x for x in factor_rows if x.get("ic_csv")][:5]
        ],
        "4. Attention": [
            _render_order_ticket(
                f"Job {j.get('id')} · {j.get('job_type')}",
                str(j.get("status") or "failed"),
                [str(j.get("error") or "cancelled")[:140], str(j.get("finished_at") or "")],
            )
            for j in bad_jobs[:5]
        ],
    }
    st.markdown(
        "<div class='order-board'>"
        + "".join(_render_order_lane(title, tickets) for title, tickets in lanes.items())
        + "</div>",
        unsafe_allow_html=True,
    )

    if factor_rows:
        with st.expander("Factor calculation tasks", expanded=True):
            import pandas as pd

            ticket_df = pd.DataFrame([
                {
                    "ticket": r.get("factor"),
                    "station": _factor_station(r)[0],
                    "family": r.get("family"),
                    "grade": r.get("grade"),
                    "mean_rank_ic": r.get("mean_rank_ic"),
                    "expression": r.get("expression"),
                }
                for r in factor_rows
            ])
            st.dataframe(ticket_df, width="stretch", hide_index=True)


def _render_mining_run(run_dir: Path) -> None:
    """Display alpha miner run status and top factors."""
    import pandas as pd

    st.subheader(f"Mining run: {run_dir.name}")
    cp = run_dir / "checkpoint.json"
    if not cp.is_file():
        st.warning("Checkpoint not found yet. If the run just started, this is normal until the first generation finishes.")
        _render_calculation_pipeline_board(run_dir, config={}, candidates_df=None)
        _render_factor_artifacts(run_dir)
        _render_backtest_artifacts(run_dir, title="Run-level decile backtest results")
        _render_run_jobs(run_dir.name)
        log_path = run_dir / "ui_alpha_miner.log"
        if log_path.is_file():
            st.markdown("**Mining process log**")
            st.code(_tail_text(log_path), language="text")
        return

    state = _cached_checkpoint(str(cp))
    if state is None:
        st.warning("Could not read checkpoint.")
        return

    config = state.get("config", {})
    candidates = state.get("all_candidates", [])
    survivors = set(state.get("survivors", []))
    df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations done", state.get("generations_done", 0))
    col2.metric("Best grade", state.get("best_grade", "?"))
    col3.metric("Best mean Rank IC", f"{state.get('best_mean_ric', 0):.4f}")
    col4.metric("Total evaluated", len(candidates))

    factor_rows = _factor_artifact_rows(run_dir, candidates_df=df if not df.empty else None)
    _render_calculation_pipeline_board(run_dir, config, candidates_df=df, factor_rows=factor_rows)

    if not df.empty:
        _render_factor_artifacts(run_dir, candidates_df=df, rows=factor_rows)

        if "error" in df.columns:
            error_s = df["error"].fillna("").astype(str)
            rejected = int((error_s != "").sum())
        else:
            rejected = 0
        diversity_rejected = int(df.get("diversity_rejected", pd.Series([False] * len(df))).fillna(False).astype(bool).sum())
        accepted = max(0, len(df) - rejected - diversity_rejected)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accepted / clean", accepted)
        m2.metric("Errors / unsafe", rejected)
        m3.metric("Diversity rejected", diversity_rejected)
        m4.metric("Survivors", len(survivors))

        show_rejected = st.checkbox("Show rejected / errored candidates", value=True, key=f"{run_dir.name}_show_rejected")
        family_filter = "all"
        if "family" in df.columns:
            families = ["all"] + sorted(x for x in df["family"].fillna("unknown").astype(str).unique() if x)
            family_filter = st.selectbox("Family filter", families, key=f"{run_dir.name}_family_filter")

        df_show = df.copy()
        if not show_rejected and "error" in df_show.columns:
            df_show = df_show[df_show["error"].fillna("").astype(str) == ""]
        if family_filter != "all" and "family" in df_show.columns:
            df_show = df_show[df_show["family"].fillna("unknown").astype(str) == family_filter]
        if "mean_rank_ic" in df_show.columns:
            df_show["mean_rank_ic"] = pd.to_numeric(df_show["mean_rank_ic"], errors="coerce")
            df_show = df_show.sort_values("mean_rank_ic", key=abs, ascending=False, na_position="last")
        if "name" in df_show.columns:
            df_show["survivor"] = df_show["name"].isin(survivors)

        show_cols = [
            "survivor", "name", "generation", "origin", "family", "grade", "mean_rank_ic",
            "rank_ic_ir", "rank_ic_win_rate", "recent_rank_ic", "train_rank_ic",
            "validation_rank_ic", "test_rank_ic", "alpha_direction", "recommendation",
            "expression", "max_similarity", "most_similar_to", "error",
        ]
        show_cols = [c for c in show_cols if c in df_show.columns]
        st.markdown("**Evaluated factors and formulas**")
        st.dataframe(df_show[show_cols].head(200), width="stretch", hide_index=True)

        detail_names = df_show["name"].astype(str).tolist() if "name" in df_show.columns else []
        if detail_names:
            selected = st.selectbox("Inspect one factor", detail_names, key=f"{run_dir.name}_factor_detail")
            row = df[df["name"].astype(str) == selected].iloc[0].to_dict()
            d1, d2 = st.columns([1, 1])
            with d1:
                st.markdown("**Formula / hypothesis**")
                st.code(str(row.get("expression") or row.get("canonical_expression") or ""), language="text")
                if row.get("economic_hypothesis"):
                    st.write(row.get("economic_hypothesis"))
                if row.get("why_not_duplicate"):
                    st.caption(f"Why not duplicate: {row.get('why_not_duplicate')}")
            with d2:
                st.markdown("**Diagnostics**")
                diag = {
                    "grade": row.get("grade"),
                    "mean_rank_ic": row.get("mean_rank_ic"),
                    "rank_ic_ir": row.get("rank_ic_ir"),
                    "win_rate": row.get("rank_ic_win_rate"),
                    "alpha_direction": row.get("alpha_direction"),
                    "recommendation": row.get("recommendation"),
                    "max_similarity": row.get("max_similarity"),
                    "most_similar_to": row.get("most_similar_to"),
                    "error": row.get("error"),
                }
                st.json(diag)

            ic_path = _resolve_existing_path(str(row.get("ic_csv_path") or ""))
            if ic_path.is_file():
                try:
                    ic_df = pd.read_csv(ic_path, parse_dates=["date"])
                    chart_cols = [c for c in ("rank_ic", "ic") if c in ic_df.columns]
                    if chart_cols:
                        st.markdown("**Selected factor daily IC**")
                        st.line_chart(ic_df.set_index("date")[chart_cols].sort_index())
                except Exception as e:
                    st.warning(f"Could not read selected factor IC CSV: {e}")

            st.markdown("**Selected factor backtest**")
            pkl_path = _resolve_existing_path(str(row.get("pkl_path") or ""))
            if pkl_path.is_file():
                if st.button("Run selected factor backtest + plots", key=f"{run_dir.name}_{selected}_bt", width="stretch"):
                    bt_rankic = run_dir / "ic" / f"{selected}_bt_rankic.csv"
                    bt_meta_dir = run_dir / "backtest_results"
                    bt_meta_dir.mkdir(parents=True, exist_ok=True)
                    bt_meta = {
                        "name": selected,
                        "factor_name": selected,
                        "family": row.get("family") or "",
                        "expression": row.get("expression") or row.get("canonical_expression") or "",
                        "economic_hypothesis": row.get("economic_hypothesis") or "",
                        "expected_sign": row.get("expected_sign") or "",
                        "alpha_direction": row.get("alpha_direction") or "",
                    }
                    (bt_meta_dir / f"{selected}_factor_meta.json").write_text(
                        json.dumps(bt_meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "analysis" / "factor_rankic_analysis.py"),
                        "--factor", str(pkl_path),
                        "--data", str(st.session_state.get("alpha_mining_data") or "data.pkl"),
                        "--output", str(bt_rankic),
                        "--workers", str(int(st.session_state.get("sb_pipeline_workers", 1))),
                        "--backtest-decile",
                        "--plot-backtest",
                        "--backtest-output-dir", str(run_dir / "backtest_results"),
                        "--plot-output-dir", str(run_dir / "backtest_plots"),
                    ]
                    if _jq is not None:
                        job_id = _jq.submit(
                            "factor_backtest",
                            {"cmd": cmd, "cwd": str(ROOT), "artifact_dir": str(run_dir)},
                            run_id=run_dir.name,
                        )
                        st.session_state["mining_backtest_job_id"] = job_id
                        _invalidate_job_caches()
                        st.success(f"Backtest queued as job `{job_id}`")
                    else:
                        st.error("job_queue not available — cannot submit backtest job.")
            else:
                st.info("Selected factor pickle not found yet; backtest can run after factor evaluation writes the .pkl.")

            _render_job_card(
                st.session_state.get("mining_backtest_job_id"),
                artifact_key="mining_run_dir",
            )

    top_csv = run_dir / "top_factors.csv"
    if top_csv.is_file():
        with st.expander("Top survivors (top_factors.csv)"):
            try:
                top_df = pd.read_csv(top_csv)
                st.dataframe(top_df, width="stretch")
            except Exception as e:
                st.warning(f"Could not read top_factors.csv: {e}")

    _render_backtest_artifacts(run_dir, title="Run-level decile backtest results", candidates_df=df)
    _render_run_jobs(run_dir.name)

    report = run_dir / "mining_report.md"
    if report.is_file():
        with st.expander("Mining report (markdown)"):
            st.markdown(report.read_text(encoding="utf-8"))

    log_path = run_dir / "ui_alpha_miner.log"
    if log_path.is_file():
        with st.expander("Mining process log"):
            st.code(_tail_text(log_path), language="text")
    elif _jq is not None:
        # Job-queue runs write to job_logs/{id}.log — surface the latest mining job log.
        mining_jobs = [j for j in _jobs_for_run(run_dir.name, limit=5) if j.get("job_type") in ("mining_start", "mining_resume")]
        if mining_jobs:
            lp = Path(str(mining_jobs[0].get("log_path") or ""))
            if lp.is_file():
                with st.expander("Mining process log (job queue)"):
                    st.code(_tail_text(lp), language="text")


@st.cache_data(ttl=2)
def _cached_get_job(job_id: int) -> "dict | None":
    if _jq is None:
        return None
    try:
        return _jq.get_job(job_id)
    except Exception:
        return None


@st.cache_data(ttl=2)
def _cached_list_jobs(limit: int = 8) -> list:
    if _jq is None:
        return []
    try:
        return _jq.list_jobs(limit=limit)
    except Exception:
        return []


@st.cache_data(ttl=2)
def _cached_list_jobs_for_run(run_id: str, limit: int = 12) -> list:
    if _jq is None:
        return []
    try:
        return _jq.list_jobs_for_run(run_id, limit=limit)
    except Exception:
        return []


@st.cache_data(ttl=10)
def _cached_mining_run_dirs() -> list[str]:
    """Cached filesystem scan — TTL 10s avoids rescanning on every rerun."""
    return [str(p) for p in _mining_run_dirs()]


@st.cache_data(ttl=5)
def _cached_checkpoint(cp_str: str) -> "dict | None":
    """Cache checkpoint.json parse — large JSON, only changes between generations."""
    cp = Path(cp_str)
    if not cp.is_file():
        return None
    try:
        return json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _invalidate_job_caches() -> None:
    """Clear all job-related caches after a mutation (submit / cancel)."""
    _cached_get_job.clear()
    _cached_list_jobs.clear()
    _cached_list_jobs_for_run.clear()


def _render_job_card(job_id: int, artifact_key: str = "") -> None:
    """Show a compact inline status card for a queued job, plus a log tail."""
    if _jq is None or job_id is None:
        return
    job = _cached_get_job(job_id)
    if job is None:
        return

    status = job["status"]
    elapsed = ""
    if job.get("started_at") and job.get("finished_at"):
        try:
            t0 = datetime.fromisoformat(job["started_at"])
            t1 = datetime.fromisoformat(job["finished_at"])
            elapsed = f" · {int((t1 - t0).total_seconds())}s"
        except Exception:
            pass

    pid_info = f"PID {job['pid']}" if job.get("pid") else ""
    label = f"Job `{job_id}` [{status}]{elapsed} {pid_info}".strip()

    if status == "success":
        st.success(label)
        # Restore artifact dir so pipeline/batch result panels appear.
        if artifact_key:
            try:
                params = json.loads(job.get("params_json") or "{}")
            except Exception:
                params = {}
            art = params.get("artifact_dir", "")
            if art:
                st.session_state[artifact_key] = str(_resolve_artifact_dir(art))
            run_id = job.get("run_id") or params.get("run_id", "")
            if run_id and artifact_key == "mining_run_dir":
                st.session_state["mining_run_dir"] = str(MINING_DIR / run_id)
    elif status == "running":
        st.info(label)
        if st.button("Cancel", key=f"cancel_job_{job_id}"):
            _jq.cancel_job(job_id)
            _invalidate_job_caches()
            st.rerun()
    elif status == "failed":
        st.error(f"Job `{job_id}` failed — {job.get('error') or 'see log below'}")
    elif status == "cancelled":
        st.warning(label)
    else:
        st.info(f"Job `{job_id}` pending — start the worker to process.")

    lp = job.get("log_path") or ""
    if lp and Path(lp).is_file():
        with st.expander(f"Log · job {job_id}", expanded=(status == "failed")):
            st.code(_tail_text(Path(lp), max_lines=80), language="text")


def _render_recent_jobs_sidebar() -> None:
    """Compact expander showing the 8 most recent jobs."""
    if _jq is None:
        return
    jobs = _cached_list_jobs(limit=8)
    if not jobs:
        return
    if not jobs:
        return
    import pandas as pd
    with st.expander("Recent jobs", expanded=False):
        df = pd.DataFrame(jobs)[["id", "job_type", "status", "run_id", "created_at"]]
        df.columns = ["#", "type", "status", "run_id", "submitted"]
        st.dataframe(df, width="stretch", hide_index=True)


def _job_params(job: dict) -> dict:
    try:
        return json.loads(job.get("params_json") or "{}")
    except Exception:
        return {}


def _artifact_key_for_job(job: dict) -> str:
    job_type = str(job.get("job_type") or "")
    if job_type == "pipeline":
        return "pipeline_artifact_dir"
    if job_type == "batch_analysis":
        return "batch_artifact_dir"
    if job_type in {"mining_start", "mining_resume", "factor_backtest"}:
        return "mining_run_dir"
    return ""


def _render_chat_jobs_panel() -> None:
    """Main-area job queue view for Chat / Pipeline so refreshes do not hide jobs."""
    if _jq is None:
        st.info("Job queue is not available in this environment.")
        return

    import pandas as pd

    jobs = _cached_list_jobs(limit=50)
    with st.expander("Jobs", expanded=True):
        top_left, top_right = st.columns([1, 1])
        with top_left:
            st.caption("Queued, running, and completed work from the local SQLite job queue.")
        with top_right:
            if st.button("Refresh jobs", key="chat_jobs_refresh", width="stretch"):
                _invalidate_job_caches()
                st.rerun()

        if not jobs:
            st.info("No jobs found yet.")
            return

        job_types = sorted({str(j.get("job_type") or "unknown") for j in jobs})
        selected_types = st.multiselect(
            "Job types",
            job_types,
            default=job_types,
            key="chat_jobs_type_filter",
        )
        shown_jobs = [j for j in jobs if str(j.get("job_type") or "unknown") in set(selected_types)]
        if not shown_jobs:
            st.info("No jobs match the selected filters.")
            return

        status_counts = {}
        for job in shown_jobs:
            status = str(job.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        metric_cols = st.columns(4)
        for i, status in enumerate(["pending", "running", "success", "failed"]):
            metric_cols[i].metric(status.title(), status_counts.get(status, 0))

        rows = []
        for job in shown_jobs:
            params = _job_params(job)
            rows.append({
                "#": job.get("id"),
                "type": job.get("job_type"),
                "status": job.get("status"),
                "run_id": job.get("run_id") or params.get("run_id") or "",
                "artifact": params.get("artifact_dir") or "",
                "submitted": job.get("created_at") or "",
                "finished": job.get("finished_at") or "",
                "error": job.get("error") or "",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        selected_job_id = st.selectbox(
            "Inspect job",
            [int(j["id"]) for j in shown_jobs],
            format_func=lambda jid: next(
                (
                    f"job {jid} · {j.get('job_type')} · {j.get('status')}"
                    for j in shown_jobs
                    if int(j["id"]) == int(jid)
                ),
                f"job {jid}",
            ),
            key="chat_jobs_inspect",
        )
        selected_job = next((j for j in shown_jobs if int(j["id"]) == int(selected_job_id)), {})
        _render_job_card(int(selected_job_id), artifact_key=_artifact_key_for_job(selected_job))


def _jobs_for_run(run_id: str, limit: int = 12) -> list[dict]:
    if _jq is None or not run_id:
        return []
    return _cached_list_jobs_for_run(run_id, limit=limit)


def _render_run_jobs(run_id: str) -> None:
    jobs = _jobs_for_run(run_id)
    if not jobs:
        return
    import pandas as pd

    with st.expander("Jobs and logs for this run", expanded=True):
        df = pd.DataFrame(jobs)[["id", "job_type", "status", "created_at", "finished_at", "error"]]
        df.columns = ["#", "type", "status", "submitted", "finished", "error"]
        st.dataframe(df, width="stretch", hide_index=True)
        selected_job = st.selectbox(
            "Inspect job log",
            [int(j["id"]) for j in jobs],
            format_func=lambda jid: f"job {jid}",
            key=f"{run_id}_job_log_select",
        )
        job = next((j for j in jobs if int(j["id"]) == int(selected_job)), None)
        lp = Path(str(job.get("log_path") or "")) if job else Path()
        if lp.is_file():
            st.code(_tail_text(lp, max_lines=120), language="text")
        elif job:
            st.caption("No log file written yet.")


def _render_alpha_mining_console(_w_max: int, default_data_pkl: str) -> None:
    st.subheader("Alpha Mining Console")
    st.caption("Start/resume mining runs, inspect the producer-consumer pipeline, and review generated formulas.")

    with st.expander("Prepared formula recipe library", expanded=False):
        try:
            import pandas as pd
            from factor_recipe_library import PREPARED_FACTOR_RECIPES

            recipe_df = pd.DataFrame([
                {
                    "recipe_id": r.recipe_id,
                    "family": r.family,
                    "name": r.name,
                    "expected_sign": r.expected_sign,
                    "expression": r.expression,
                }
                for r in PREPARED_FACTOR_RECIPES
            ])
            st.dataframe(recipe_df, width="stretch", hide_index=True)
            st.caption("DSL mining asks the LLM to choose recipe_id; the system expands the expression locally.")
        except Exception as e:
            st.warning(f"Could not load prepared recipe library: {e}")

    st.markdown("**Run controls**")
    mining_dir = MINING_DIR
    ui_state = _read_mining_ui_state()
    default_run_id = f"mining_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    c1, c2 = st.columns([1, 1])
    with c1:
        st.text_input("Run ID", value=default_run_id, key="alpha_mining_run_id")
        st.text_input("Config", value=str(AGENT_DIR / "alpha_mining_config.yaml"), key="alpha_mining_config")
        st.text_input("Mining data", value=default_data_pkl, key="alpha_mining_data")
    with c2:
        st.number_input("Generations", min_value=1, max_value=50, value=1, step=1, key="alpha_mining_generations")
        st.number_input("Factors / gen", min_value=1, max_value=100, value=3, step=1, key="alpha_mining_per_gen")
        st.number_input("Outer consumers", min_value=1, max_value=_w_max, value=1, step=1, key="alpha_mining_outer_workers")

    mining_job_id = st.session_state.get("mining_job_id")
    if mining_job_id and _jq is not None:
        _render_job_card(mining_job_id, artifact_key="mining_run_dir")

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("Start alpha mining", width="stretch"):
            run_id = str(st.session_state.get("alpha_mining_run_id") or default_run_id).strip()
            run_dir = mining_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(AGENT_DIR / "alpha_miner.py"),
                "start",
                "--run-id", run_id,
                "--config", str(st.session_state.get("alpha_mining_config") or AGENT_DIR / "alpha_mining_config.yaml"),
                "--generations", str(int(st.session_state.get("alpha_mining_generations", 1))),
                "--per-gen", str(int(st.session_state.get("alpha_mining_per_gen", 3))),
                "--model", str(st.session_state.get("sb_model", "gpt-4.1")),
                "--provider", str(st.session_state.get("sb_provider", "openai")),
                "--outer-workers", str(int(st.session_state.get("alpha_mining_outer_workers", 1))),
                "--data", str(st.session_state.get("alpha_mining_data") or default_data_pkl),
            ]
            if _jq is not None:
                job_id = _jq.submit("mining_start", {"cmd": cmd, "cwd": str(ROOT)}, run_id=run_id)
                st.session_state["mining_job_id"] = job_id
                st.session_state["mining_run_dir"] = str(run_dir)
                _write_mining_ui_state(run_dir, ui_state.get("pid"))
                _invalidate_job_caches()
                _cached_mining_run_dirs.clear()
                st.success(f"Mining queued as job `{job_id}` (run: {run_id})")
            else:
                st.error("job_queue not available — cannot submit job.")
    with b2:
        if st.button("Refresh mining view", width="stretch"):
            _cached_checkpoint.clear()
            _cached_mining_run_dirs.clear()
            _invalidate_job_caches()
            st.rerun()
    st.caption("Jobs run through the SQLite queue. Use each job card's Cancel button for queued/running work.")

    run_dirs = [Path(p) for p in _cached_mining_run_dirs()]
    if run_dirs:
        st.markdown("**Existing runs**")
        run_names = [d.name for d in run_dirs[:50]]
        current_name = Path(str(st.session_state.get("mining_run_dir", ""))).name
        default_index = run_names.index(current_name) if current_name in run_names else 0
        selected_run = st.selectbox("Select run", run_names, index=default_index, key="sb_mining_run")
        selected_dir = mining_dir / selected_run
        can_resume = (selected_dir / "checkpoint.json").is_file()
        if not can_resume:
            st.caption("Selected run has no checkpoint yet, so resume is disabled. You can still view files/logs for diagnosis.")
        v1, v2 = st.columns([1, 1])
        with v1:
            if st.button("View selected run", width="stretch"):
                st.session_state["mining_run_dir"] = str(selected_dir)
                _write_mining_ui_state(selected_dir, ui_state.get("pid"))
        with v2:
            if st.button("Resume selected run", width="stretch", disabled=not can_resume):
                run_dir = selected_dir
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(AGENT_DIR / "alpha_miner.py"),
                    "resume",
                    "--run-id", selected_run,
                    "--config", str(st.session_state.get("alpha_mining_config") or AGENT_DIR / "alpha_mining_config.yaml"),
                ]
                if _jq is not None:
                    job_id = _jq.submit("mining_resume", {"cmd": cmd, "cwd": str(ROOT)}, run_id=selected_run)
                    st.session_state["mining_job_id"] = job_id
                    st.session_state["mining_run_dir"] = str(run_dir)
                    _write_mining_ui_state(run_dir, ui_state.get("pid"))
                    _invalidate_job_caches()
                    st.success(f"Resume queued as job `{job_id}`")
                else:
                    st.error("job_queue not available — cannot submit job.")
    else:
        st.caption("No mining runs found.")


@st.cache_resource
def _init_db_once() -> None:
    """Run once per server process — opens SQLite, sets WAL, creates table/indexes."""
    if _jq is not None:
        try:
            _jq.init_db()
        except Exception:
            pass


@st.cache_resource
def _worker_counts() -> tuple[int, int]:
    """Cache cpu_count so multiprocessing.cpu_count() is not called on every rerun."""
    nc = int(multiprocessing.cpu_count() or 4)
    return min(32, max(4, nc)), max(1, min(8, nc))


@st.cache_resource
def _ensure_dirs() -> Path:
    """Create persistent output dirs once per server start."""
    gen = ROOT / "generated_factors"
    gen.mkdir(parents=True, exist_ok=True)
    MINING_DIR.mkdir(parents=True, exist_ok=True)
    return gen


def _stream_assistant(messages: list, model: str, temperature: float, provider: str = "openai"):
    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        with client.messages.stream(
            model=model,
            max_tokens=8096,
            system=system_msg,
            messages=user_msgs,
            temperature=temperature,
        ) as stream:
            for token in stream.text_stream:
                if token:
                    yield token
    else:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token


def _render_chat_pipeline_sidebar(gen_dir: Path, last_code, _w_max: int, _w_default: int) -> None:
    """Sidebar content that is only relevant in Chat / Pipeline view."""
    st.divider()
    st.subheader("Pipeline (optional)")
    data_pkl = st.text_input("Market pickle", value="data.pkl", key="sb_data")
    artifact_dir = st.text_input(
        "Artifact dir (under project root)",
        value=st.session_state.artifact_default,
        key="sb_art",
    )
    st.selectbox(
        "Rank IC backend",
        ["pandas", "torch"],
        key="sb_pipeline_backend",
        help="pandas: default CPU path; torch: supports CUDA/CPU device selection.",
    )
    st.text_input(
        "Torch device",
        value="auto",
        key="sb_pipeline_device",
        help="Used when backend=torch. Example: auto / cuda / cuda:0 / cpu",
    )
    st.slider(
        "Parallel workers (Level-2: per-day IC/decile)",
        min_value=1,
        max_value=_w_max,
        value=_w_default,
        key="sb_pipeline_workers",
        help="Per-trading-day IC/decile tasks run in a process pool when >1. Factor pickle step stays single-process.",
    )
    st.selectbox(
        "Rank IC return horizon",
        ["next-day forward", "same-day diagnostic"],
        key="sb_pipeline_return_horizon",
        help="Default is T signal vs T+1 return. Same-day is only for diagnostics.",
    )
    st.divider()
    st.subheader("Export & run")
    save_name = st.text_input("Save as (.py)", value=st.session_state.save_stub, key="sb_save")
    save_path = gen_dir / save_name if save_name.endswith(".py") else gen_dir / f"{save_name}.py"
    st.caption("Latest extractable `compute_factor_df` module is used for save/run.")
    model = st.session_state.sb_model
    if st.button("Save latest code → generated_factors", width="stretch", disabled=not last_code):
        hdr = (
            f"# Saved from Factor Agent UI — {datetime.now().isoformat(timespec='seconds')}\n"
            f"# Model: {model}\n\n"
        )
        save_path.write_text(hdr + last_code, encoding="utf-8")
        st.success(f"Wrote {save_path.relative_to(ROOT)}")

    if st.button("Run pipeline (pickle + Rank IC + backtest + plots)", width="stretch"):
        if not last_code:
            st.warning("No `compute_factor_df` in recent assistant replies.")
        else:
            hdr = (
                f"# Saved from Factor Agent UI — {datetime.now().isoformat(timespec='seconds')}\n"
                f"# Model: {model}\n\n"
            )
            save_path.write_text(hdr + last_code, encoding="utf-8")
            pipe = AGENT_DIR / "factor_agent_pipeline.py"
            cmd = [
                sys.executable,
                str(pipe),
                "--skip-generate",
                str(save_path),
                "--data",
                data_pkl,
                "--artifact-dir",
                artifact_dir,
            ]
            _workers = int(st.session_state.get("sb_pipeline_workers", _w_default))
            if _workers > 1:
                cmd.extend(["--workers", str(_workers)])
            _backend = str(st.session_state.get("sb_pipeline_backend", "pandas"))
            _device = str(st.session_state.get("sb_pipeline_device", "auto")).strip() or "auto"
            _ui_provider = str(st.session_state.get("sb_provider", "openai"))
            cmd.extend(["--backend", _backend, "--device", _device, "--provider", _ui_provider])
            if st.session_state.get("sb_pipeline_return_horizon") == "same-day diagnostic":
                cmd.append("--no-next-day")
            if _jq is not None:
                job_id = _jq.submit(
                    "pipeline",
                    {"cmd": cmd, "cwd": str(ROOT), "artifact_dir": artifact_dir},
                )
                st.session_state["pipeline_job_id"] = job_id
                _invalidate_job_caches()
                st.success(f"Queued as job `{job_id}` — the worker will run it.")
            else:
                st.error("job_queue not available — cannot submit job.")

    _render_job_card(
        st.session_state.get("pipeline_job_id"),
        artifact_key="pipeline_artifact_dir",
    )
    st.divider()
    st.subheader("Batch analysis (multi-level parallel)")
    batch_factor_dir = st.text_input(
        "Factor dir",
        value="factors_by_type",
        key="batch_factor_dir",
    )
    batch_out_dir = st.text_input(
        "Batch output dir",
        value="rankic_batch_results",
        key="batch_out_dir",
    )
    st.selectbox(
        "Batch backend",
        ["pandas", "torch"],
        key="batch_backend",
        help="Applied to each factor's Rank IC computation.",
    )
    st.text_input(
        "Batch torch device",
        value="auto",
        key="batch_device",
    )
    st.slider(
        "Factor workers (Level-1: cross-factor parallel)",
        min_value=1,
        max_value=_w_max,
        value=max(1, min(4, _w_default)),
        key="batch_factor_workers",
        help="Run multiple factor files in parallel.",
    )
    st.slider(
        "IC workers (Level-2: per-factor/day parallel)",
        min_value=1,
        max_value=_w_max,
        value=_w_default,
        key="batch_ic_workers",
        help="Workers used inside each factor for per-day IC/decile tasks.",
    )
    st.selectbox(
        "Batch return horizon",
        ["next-day forward", "same-day diagnostic"],
        key="batch_return_horizon",
        help="Default is T signal vs T+1 return. Same-day is only for diagnostics.",
    )
    if st.button("Run batch analysis", width="stretch"):
        batch_script = ROOT / "scripts" / "analysis" / "batch_factor_analysis.py"
        _factor_workers = int(st.session_state.get("batch_factor_workers", 1))
        _ic_workers = int(st.session_state.get("batch_ic_workers", _w_default))
        _batch_backend = str(st.session_state.get("batch_backend", "pandas"))
        _batch_device = str(st.session_state.get("batch_device", "auto")).strip() or "auto"
        cmd = [
            sys.executable,
            str(batch_script),
            batch_factor_dir,
            "--output-dir",
            batch_out_dir,
            "--data",
            data_pkl,
            "--factor-workers",
            str(_factor_workers),
            "--ic-workers",
            str(_ic_workers),
            "--backend",
            _batch_backend,
            "--device",
            _batch_device,
        ]
        if st.session_state.get("batch_return_horizon") == "same-day diagnostic":
            cmd.append("--no-next-day")
        if _jq is not None:
            job_id = _jq.submit(
                "batch_analysis",
                {"cmd": cmd, "cwd": str(ROOT), "artifact_dir": batch_out_dir},
            )
            st.session_state["batch_job_id"] = job_id
            _invalidate_job_caches()
            st.success(f"Queued as job `{job_id}` — the worker will run it.")
        else:
            st.error("job_queue not available — cannot submit job.")

    _render_job_card(
        st.session_state.get("batch_job_id"),
        artifact_key="batch_artifact_dir",
    )
    _render_recent_jobs_sidebar()
    st.divider()
    if st.button("New conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()


def main():
    st.set_page_config(
        page_title="Factor Agent",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme_css()
    _init_db_once()

    ok, err = _bootstrap_api_key()
    if not ok:
        st.error(err)
        st.stop()

    gen_dir = _ensure_dirs()
    _w_max, _w_default = _worker_counts()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "artifact_default" not in st.session_state:
        st.session_state.artifact_default = f"agent_runs/chat_ui_{datetime.now().strftime('%Y%m%d')}"
    if "save_stub" not in st.session_state:
        st.session_state.save_stub = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    if "sb_provider" not in st.session_state:
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
        has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        st.session_state.sb_provider = "anthropic" if (has_anthropic and not has_openai) else "openai"
    if "sb_model" not in st.session_state:
        st.session_state.sb_model = "gpt-4.1"
    if "chat_render_limit" not in st.session_state:
        st.session_state.chat_render_limit = DEFAULT_CHAT_RENDER_LIMIT
    st.title("Factor Agent")
    st.caption("Chat · generate `compute_factor_df()` · optional Rank IC / decile backtest")

    if "mining_run_dir" not in st.session_state:
        ui_state = _read_mining_ui_state()
        saved_run_value = str(ui_state.get("current_run_dir") or "").strip()
        saved_run_dir = Path(saved_run_value) if saved_run_value else None
        if saved_run_dir is not None and saved_run_dir.is_dir():
            st.session_state["mining_run_dir"] = str(saved_run_dir)
        else:
            latest_runs = _cached_mining_run_dirs()
            if latest_runs:
                st.session_state["mining_run_dir"] = latest_runs[0]

    with st.sidebar:
        ui_view = st.radio(
            "目录",
            ["Chat / Pipeline", "Alpha Mining Console"],
            key="ui_view",
        )

    if ui_view == "Chat / Pipeline":
        render_limit = int(st.session_state.get("chat_render_limit", DEFAULT_CHAT_RENDER_LIMIT))
        all_messages = st.session_state.messages
        shown_messages = all_messages[-render_limit:] if render_limit > 0 else all_messages
        hidden_count = max(0, len(all_messages) - len(shown_messages))
        if hidden_count > 0:
            st.caption(
                f"Showing last {len(shown_messages)} messages for smoother UI "
                f"(hidden older messages: {hidden_count})."
            )
        for m in shown_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Describe a factor, or ask for a revision…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            api_messages = [{"role": "system", "content": CHAT_SYSTEM}]
            for m in st.session_state.messages:
                api_messages.append({"role": m["role"], "content": m["content"]})

            with st.chat_message("assistant"):
                text_holder = st.empty()
                collected: list[str] = []
                for token in _stream_assistant(
                    api_messages,
                    st.session_state.sb_model,
                    float(st.session_state.sb_temp),
                    provider=st.session_state.get("sb_provider", "openai"),
                ):
                    collected.append(token)
                    text_holder.markdown("".join(collected))
                full = "".join(collected)
                st.session_state.messages.append({"role": "assistant", "content": full})
    else:
        _render_alpha_mining_console(_w_max, str(st.session_state.get("sb_data", "data.pkl")))

    # Only needed in Chat/Pipeline view — skip the scan entirely in Mining Console.
    last_code = None
    if ui_view == "Chat / Pipeline":
        for m in reversed(st.session_state.messages):
            if m["role"] != "assistant":
                continue
            code = extract_python_code(m["content"])
            if "def compute_factor_df" in code:
                last_code = code
                break

    _provider_models = {
        "openai": ["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"],
    }

    with st.sidebar:
        st.subheader("Model")
        provider = st.selectbox(
            "Provider",
            ["openai", "anthropic"],
            key="sb_provider",
        )
        model_choices = _provider_models.get(provider, ["gpt-4.1"])
        # Reset model if it doesn't belong to the selected provider
        if st.session_state.get("sb_model") not in model_choices:
            st.session_state.sb_model = model_choices[0]
        st.selectbox(
            "Model",
            model_choices,
            key="sb_model",
        )
        st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="sb_temp")
        if ui_view == "Chat / Pipeline":
            st.slider(
                "Chat messages to render",
                min_value=10,
                max_value=200,
                value=int(st.session_state.get("chat_render_limit", DEFAULT_CHAT_RENDER_LIMIT)),
                step=10,
                key="chat_render_limit",
                help="Only render the latest N chat messages on each rerun to reduce UI lag.",
            )
        if ui_view == "Chat / Pipeline":
            _render_chat_pipeline_sidebar(gen_dir, last_code, _w_max, _w_default)
        else:
            _render_recent_jobs_sidebar()

    # Must run AFTER sidebar so session state set by sidebar buttons is visible.
    # Each block is gated to its own view — the other view's renders are completely
    # skipped, which eliminates the main source of slowness: checkpoint.json parsing,
    # glob scans, and CSV reads running on every chat interaction.
    if ui_view == "Chat / Pipeline":
        st.divider()
        _render_chat_jobs_panel()

        art_key = st.session_state.get("pipeline_artifact_dir")
        if art_key:
            art_show = Path(art_key)
            if art_show.is_dir():
                st.divider()
                _render_pipeline_artifacts(art_show)

        batch_key = st.session_state.get("batch_artifact_dir")
        if batch_key:
            batch_show = Path(batch_key)
            if batch_show.is_dir():
                st.divider()
                _render_batch_artifacts(batch_show)
    else:
        mining_key = st.session_state.get("mining_run_dir")
        if mining_key:
            mining_show = Path(mining_key)
            if mining_show.is_dir():
                st.divider()
                _render_mining_run(mining_show)


if __name__ == "__main__":
    main()
