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


def _bootstrap_api_key() -> tuple[bool, str]:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return True, ""
    if not DOTENV.is_file():
        return False, f"Missing {DOTENV} and OPENAI_API_KEY not set in environment."
    try:
        from dotenv import load_dotenv

        load_dotenv(DOTENV)
    except ImportError:
        for line in DOTENV.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return False, "OPENAI_API_KEY empty after loading .env"
    return True, ""


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


def _render_pipeline_artifacts(art: Path) -> None:
    """Show Rank IC table summary, charts, and decile plot images after a successful run."""
    import pandas as pd

    st.subheader("Pipeline results")
    try:
        art_rel = art.relative_to(ROOT)
    except ValueError:
        art_rel = art
    st.caption(f"Output folder: `{art_rel}`")

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


def _stream_assistant(messages: list, model: str, temperature: float):
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
    if "sb_model" not in st.session_state:
        st.session_state.sb_model = "gpt-4.1"
    if "sb_temp" not in st.session_state:
        st.session_state.sb_temp = 0.2

    st.title("Factor Agent")
    st.caption("Chat · generate `compute_factor_df()` · optional Rank IC / decile backtest")

    gen_dir = ROOT / "generated_factors"
    gen_dir.mkdir(parents=True, exist_ok=True)

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
            ):
                collected.append(token)
                text_holder.markdown("".join(collected))
            full = "".join(collected)
            st.session_state.messages.append({"role": "assistant", "content": full})

    last_code = None
    for m in reversed(st.session_state.messages):
        if m["role"] != "assistant":
            continue
        code = extract_python_code(m["content"])
        if "def compute_factor_df" in code:
            last_code = code
            break

    with st.sidebar:
        st.subheader("Model")
        st.selectbox(
            "OpenAI model",
            ["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
            label_visibility="collapsed",
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
        st.slider(
            "Parallel workers (Rank IC + decile)",
            min_value=1,
            max_value=_w_max,
            value=_w_default,
            key="sb_pipeline_workers",
            help="Per-trading-day IC/decile tasks run in a process pool when >1. Factor pickle step stays single-process.",
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


if __name__ == "__main__":
    main()
