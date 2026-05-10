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

import multiprocessing
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

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
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_artifact_dir(artifact_dir: str) -> Path:
    p = Path(artifact_dir)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p.resolve()


def _tail_text(path: Path, max_lines: int = 160) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
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


def _read_mining_ui_state() -> dict:
    if not MINING_UI_STATE.is_file():
        return {}
    try:
        import json
        return json.loads(MINING_UI_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_mining_ui_state(run_dir: Path, pid: int | None = None) -> None:
    import json
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
    import json
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
                st.dataframe(ic_df.head(200), use_container_width=True)

    plot_dir = art / "backtest_plots"
    if plot_dir.is_dir():
        pngs = sorted(plot_dir.glob("*.png"))
        if pngs:
            st.markdown("**Decile backtest plots**")
            cols = st.columns(2)
            for i, png in enumerate(pngs):
                cols[i % 2].image(str(png), caption=png.name, use_container_width=True)

    bt = art / "backtest_results"
    if bt.is_dir():
        csvs = sorted(bt.glob("*.csv"))
        if csvs:
            with st.expander("Backtest CSV paths"):
                for c in csvs:
                    try:
                        st.text(str(c.relative_to(ROOT)))
                    except ValueError:
                        st.text(str(c))


def _render_batch_artifacts(out_dir: Path) -> None:
    import json
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
        st.dataframe(show[cols].head(200), use_container_width=True)
    else:
        st.dataframe(show.head(200), use_container_width=True)


def _render_mining_run(run_dir: Path) -> None:
    """Display alpha miner run status and top factors."""
    import json
    import pandas as pd

    st.subheader(f"Mining run: {run_dir.name}")
    cp = run_dir / "checkpoint.json"
    if not cp.is_file():
        st.warning("Checkpoint not found yet. If the run just started, this is normal until the first generation finishes.")
        log_path = run_dir / "ui_alpha_miner.log"
        if log_path.is_file():
            st.markdown("**Mining process log**")
            st.code(_tail_text(log_path), language="text")
        return

    try:
        state = json.loads(cp.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not read checkpoint: {e}")
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

    q_enabled = bool(config.get("pipeline_queue_enabled", True))
    outer_workers = int(config.get("outer_workers", 1) or 1)
    queue_size = int(config.get("pipeline_queue_size", 0) or max(2, outer_workers * 2))
    eval_workers = int(config.get("eval_workers", config.get("workers", 1)) or 1)
    with st.expander("Producer / Consumer pipeline", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Producer", "LLM + mutations")
        p2.metric("Queue", f"{queue_size}" if q_enabled else "off")
        p3.metric("Consumers", outer_workers)
        p4.metric("IC workers / factor", eval_workers)
        st.caption(
            "Producer creates safe DSL/Python candidates, the bounded queue buffers them, "
            "consumer workers run safety validation, factor evaluation, Rank IC/backtest, then write checkpoint and ledger."
        )

    if not df.empty:
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
        st.dataframe(df_show[show_cols].head(200), use_container_width=True, hide_index=True)

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

            ic_path = Path(str(row.get("ic_csv_path") or ""))
            if ic_path.is_file():
                try:
                    ic_df = pd.read_csv(ic_path, parse_dates=["date"])
                    chart_cols = [c for c in ("rank_ic", "ic") if c in ic_df.columns]
                    if chart_cols:
                        st.markdown("**Selected factor daily IC**")
                        st.line_chart(ic_df.set_index("date")[chart_cols].sort_index())
                except Exception as e:
                    st.warning(f"Could not read selected factor IC CSV: {e}")

    top_csv = run_dir / "top_factors.csv"
    if top_csv.is_file():
        with st.expander("Top survivors (top_factors.csv)"):
            try:
                top_df = pd.read_csv(top_csv)
                st.dataframe(top_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not read top_factors.csv: {e}")

    report = run_dir / "mining_report.md"
    if report.is_file():
        with st.expander("Mining report (markdown)"):
            st.markdown(report.read_text(encoding="utf-8"))

    log_path = run_dir / "ui_alpha_miner.log"
    if log_path.is_file():
        with st.expander("Mining process log"):
            st.code(_tail_text(log_path), language="text")


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
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
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


def main():
    st.set_page_config(
        page_title="Factor Agent",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme_css()

    ok, err = _bootstrap_api_key()
    if not ok:
        st.error(err)
        st.stop()

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
    if "sb_temp" not in st.session_state:
        st.session_state.sb_temp = 0.2
    if "batch_factor_dir" not in st.session_state:
        st.session_state.batch_factor_dir = "factors_by_type"
    if "batch_out_dir" not in st.session_state:
        st.session_state.batch_out_dir = f"rankic_batch_results"

    st.title("Factor Agent")
    st.caption("Chat · generate `compute_factor_df()` · optional Rank IC / decile backtest")

    gen_dir = ROOT / "generated_factors"
    gen_dir.mkdir(parents=True, exist_ok=True)
    MINING_DIR.mkdir(parents=True, exist_ok=True)

    if "mining_run_dir" not in st.session_state:
        ui_state = _read_mining_ui_state()
        saved_run_value = str(ui_state.get("current_run_dir") or "").strip()
        saved_run_dir = Path(saved_run_value) if saved_run_value else None
        if saved_run_dir is not None and saved_run_dir.is_dir():
            st.session_state["mining_run_dir"] = str(saved_run_dir)
        else:
            latest_runs = _mining_run_dirs()
            if latest_runs:
                st.session_state["mining_run_dir"] = str(latest_runs[0])

    with st.sidebar:
        ui_view = st.radio(
            "目录",
            ["Chat / Pipeline", "Alpha Mining Console"],
            key="ui_view",
        )

    if ui_view == "Chat / Pipeline":
        for m in st.session_state.messages:
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
        st.subheader("Alpha Mining Console")
        st.caption("Use the Alpha Miner controls in the sidebar to start/resume runs. Results update from checkpoint files.")

    last_code = None
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
        st.divider()
        st.subheader("Pipeline (optional)")
        data_pkl = st.text_input("Market pickle", value="data.pkl", key="sb_data")
        artifact_dir = st.text_input(
            "Artifact dir (under project root)",
            value=st.session_state.artifact_default,
            key="sb_art",
        )
        _nc = int(multiprocessing.cpu_count() or 4)
        _w_max = min(32, max(4, _nc))
        _w_default = max(1, min(8, _nc))
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
        if st.button("Save latest code → generated_factors", use_container_width=True, disabled=not last_code):
            hdr = (
                f"# Saved from Factor Agent UI — {datetime.now().isoformat(timespec='seconds')}\n"
                f"# Model: {model}\n\n"
            )
            save_path.write_text(hdr + last_code, encoding="utf-8")
            st.success(f"Wrote {save_path.relative_to(ROOT)}")

        if st.button("Run pipeline (pickle + Rank IC + backtest + plots)", use_container_width=True):
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
                with st.spinner("Running pipeline…"):
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                st.code(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), language="text")
                if proc.returncode == 0:
                    st.success("Pipeline finished OK.")
                    st.session_state["pipeline_artifact_dir"] = str(_resolve_artifact_dir(artifact_dir))
                else:
                    st.error(f"Exit code {proc.returncode}")
        st.divider()
        st.subheader("Batch analysis (multi-level parallel)")
        batch_factor_dir = st.text_input(
            "Factor dir",
            value=st.session_state.batch_factor_dir,
            key="batch_factor_dir",
        )
        batch_out_dir = st.text_input(
            "Batch output dir",
            value=st.session_state.batch_out_dir,
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
        if st.button("Run batch analysis", use_container_width=True):
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
            with st.spinner("Running batch analysis…"):
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            st.code(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), language="text")
            if proc.returncode == 0:
                st.success("Batch analysis finished OK.")
                st.session_state["batch_artifact_dir"] = str(_resolve_artifact_dir(batch_out_dir))
            else:
                st.error(f"Exit code {proc.returncode}")
        st.divider()
        st.subheader("Alpha Miner runs")
        mining_dir = MINING_DIR
        ui_state = _read_mining_ui_state()
        default_run_id = f"mining_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.text_input("Run ID", value=default_run_id, key="alpha_mining_run_id")
        st.text_input("Config", value=str(AGENT_DIR / "alpha_mining_config.yaml"), key="alpha_mining_config")
        mcols = st.columns(2)
        with mcols[0]:
            st.number_input("Generations", min_value=1, max_value=50, value=1, step=1, key="alpha_mining_generations")
            st.number_input("Outer consumers", min_value=1, max_value=_w_max, value=1, step=1, key="alpha_mining_outer_workers")
        with mcols[1]:
            st.number_input("Factors / gen", min_value=1, max_value=100, value=3, step=1, key="alpha_mining_per_gen")
            st.text_input("Mining data", value=data_pkl, key="alpha_mining_data")

        proc = st.session_state.get("alpha_mining_proc")
        if _process_running(proc):
            st.caption(f"Running PID: `{proc.pid}`")
            if st.button("Stop current mining process", use_container_width=True):
                proc.terminate()
                st.warning("Stop signal sent.")
        elif _pid_running(ui_state.get("pid")):
            st.caption(f"Running PID from saved state: `{ui_state.get('pid')}`")
            if st.button("Stop saved mining process", use_container_width=True):
                try:
                    os.kill(int(ui_state["pid"]), signal.SIGTERM)
                    st.warning("Stop signal sent to saved PID.")
                except Exception as e:
                    st.error(f"Could not stop saved PID: {e}")
        else:
            if proc is not None:
                rc = proc.poll()
                st.caption(f"Last mining process exited: `{rc}`")

        if st.button("Start alpha mining", use_container_width=True):
            run_id = str(st.session_state.get("alpha_mining_run_id") or default_run_id).strip()
            run_dir = mining_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "ui_alpha_miner.log"
            cmd = [
                sys.executable,
                str(AGENT_DIR / "alpha_miner.py"),
                "start",
                "--run-id",
                run_id,
                "--config",
                str(st.session_state.get("alpha_mining_config") or AGENT_DIR / "alpha_mining_config.yaml"),
                "--generations",
                str(int(st.session_state.get("alpha_mining_generations", 1))),
                "--per-gen",
                str(int(st.session_state.get("alpha_mining_per_gen", 3))),
                "--model",
                str(st.session_state.get("sb_model", "gpt-4.1")),
                "--provider",
                str(st.session_state.get("sb_provider", "openai")),
                "--outer-workers",
                str(int(st.session_state.get("alpha_mining_outer_workers", 1))),
                "--data",
                str(st.session_state.get("alpha_mining_data") or data_pkl),
            ]
            log_f = open(log_path, "a", encoding="utf-8", buffering=1)
            log_f.write(f"\n\n[{datetime.now().isoformat(timespec='seconds')}] RUN {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            st.session_state["alpha_mining_proc"] = proc
            st.session_state["mining_run_dir"] = str(run_dir)
            _write_mining_ui_state(run_dir, proc.pid)
            st.success(f"Alpha mining started: {run_id}")

        run_dirs = _mining_run_dirs()
        if run_dirs:
            run_names = [d.name for d in run_dirs[:20]]
            current_name = Path(str(st.session_state.get("mining_run_dir", ""))).name
            default_index = run_names.index(current_name) if current_name in run_names else 0
            selected_run = st.selectbox("Select run", run_names, index=default_index, key="sb_mining_run")
            c_view, c_resume = st.columns(2)
            with c_view:
                if st.button("View", use_container_width=True):
                    selected_dir = mining_dir / selected_run
                    st.session_state["mining_run_dir"] = str(selected_dir)
                    _write_mining_ui_state(selected_dir, ui_state.get("pid"))
            with c_resume:
                if st.button("Resume", use_container_width=True):
                    run_dir = mining_dir / selected_run
                    log_path = run_dir / "ui_alpha_miner.log"
                    cmd = [
                        sys.executable,
                        str(AGENT_DIR / "alpha_miner.py"),
                        "resume",
                        "--run-id",
                        selected_run,
                        "--config",
                        str(st.session_state.get("alpha_mining_config") or AGENT_DIR / "alpha_mining_config.yaml"),
                    ]
                    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
                    log_f.write(f"\n\n[{datetime.now().isoformat(timespec='seconds')}] RUN {' '.join(cmd)}\n")
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(ROOT),
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    st.session_state["alpha_mining_proc"] = proc
                    st.session_state["mining_run_dir"] = str(run_dir)
                    _write_mining_ui_state(run_dir, proc.pid)
                    st.success(f"Resume started: {selected_run}")
            if st.button("Refresh mining view", use_container_width=True):
                st.rerun()
        else:
            st.caption("No mining runs found.")
        st.divider()
        if st.button("New conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Must run AFTER sidebar: `pipeline_artifact_dir` is set when "Run pipeline" succeeds.
    # If this block ran earlier in the script, the same rerun would still see stale session state.
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

    mining_key = st.session_state.get("mining_run_dir")
    if mining_key:
        mining_show = Path(mining_key)
        if mining_show.is_dir():
            st.divider()
            _render_mining_run(mining_show)


if __name__ == "__main__":
    main()
