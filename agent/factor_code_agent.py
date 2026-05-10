#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor code agent: describe a factor in natural language, get runnable Python code.

Reads OPENAI_API_KEY or ANTHROPIC_API_KEY from .env in the project root (parent of agent/).

Usage:
  cd e:\\data
  python agent/factor_code_agent.py "momentum factor: past 20-day return, winsorize 1% and 99%"
  python agent/factor_code_agent.py --interactive
  python agent/factor_code_agent.py -d "..." --out generated_factors/my_factor.py --model gpt-4.1
  python agent/factor_code_agent.py -d "..." --provider anthropic --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Project root: parent of agent/
ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = ROOT / ".env"
GENERATED_DIR = ROOT / "generated_factors"


# Paths are relative to project root (e:\\data). Fed to the LLM so generated code matches this repo.
DATA_LAYOUT = """
Repository data layout (project root = parent folder of `agent/`):

| Path | Role |
|------|------|
| `agent/` | LLM factor agent: `factor_code_agent.py`, `factor_agent_pipeline.py`, `templates/` (preset factors). Generated code must read market data **only** from `data.pkl`, not from rqdatac. |
| `data.pkl` | Panel for Python Rank IC: columns include `order_book_id`, `date`, `close`, flags `ST`, `limit_up_flag`, `limit_down_flag`, `suspended`. Used by `scripts/analysis/factor_rankic_analysis.py`. |
| `data.npz` | Same market as packaged array (`data`, `columns` keys). Use if you load numpy; user may convert to `data.pkl` separately. |
| `market_data_cpp.npz` | Compact market for C++ pipeline: `stock_ids`, `date_ids`, `return_next`, `close`, `stock_code_list`, `date_values`, etc. |
| `market_data_cpp_arrays/` | Unpacked `.npy` from the same market object. |
| `factors_by_type/` | RQ downloaded factors: per-factor `.pkl`, MultiIndex (`order_book_id`,`date`), one column. |
| `factors_by_type_npy/factors_by_type/<type>/*.npz` | Stock-level factor: keys `values`, `index_order_book_id`, `index_date`. |
| `generated_factors/` | LLM-generated `.py` and saved factor `.pkl` for backtest. |
| `rankic_batch_results/` | Python batch Rank IC CSV outputs. |
| `decile_cpp_batch_results/` | C++ batch decile daily/cum CSV per factor. |
| `decile_plots/` | Decile cumulative plots. |
| `total_income_factor.pkl` | Example custom factor pickle on disk. |

Backtest entrypoint (Python): `scripts/analysis/factor_rankic_analysis.py --factor <.pkl> --data data.pkl [--next-day] [--backtest-decile]`.
Factor file must be DataFrame with MultiIndex (`order_book_id`,`date`) and one numeric column.
"""


SYSTEM_PROMPT = """You are a quantitative research assistant for A-share (China stock) factor coding.

Output requirements (STRICT):
1. Output ONE complete Python file as plain code only. No markdown fences, no explanation before/after the code block.
2. REQUIRED: define a function exactly:
     def compute_factor_df() -> pd.DataFrame:
   It must return the factor panel ready for merge with market data.
3. Factor DataFrame format (choose one):
   - MultiIndex (order_book_id, date) with exactly one column: factor name in ASCII snake_case; dtype float64; NaN allowed; OR
   - Columns: order_book_id, date, <one_factor_column>.
4. DATA SOURCE (MANDATORY — local only): Use ONLY `data.pkl` at the project root. Inside compute_factor_df(), find the project root by walking up from this file until data.pkl is found — this works regardless of where the file is saved:
     import os
     from pathlib import Path
     _p = Path(os.environ.get("FACTOR_DATA_ROOT", str(Path(__file__).resolve().parent)))
     if not (_p / "data.pkl").is_file():
         _p = Path(__file__).resolve().parent
         while not (_p / "data.pkl").is_file() and _p != _p.parent:
             _p = _p.parent
     root = _p
     df = pd.read_pickle(root / "data.pkl")
   Do not use hardcoded absolute paths. Typical columns include: order_book_id, date, open, high, low, close, volume, suspended, ST, limit_up_flag, limit_down_flag, and others present in the file — inspect or select only what you need.
   FORBIDDEN: rqdatac / rq / Ricequant APIs, rq.init(), get_price from cloud, Tushare/eastmoney downloads, or any remote HTTP market-data client. The runtime environment may not have those packages; the user already has the full panel in data.pkl.
5. Vectorized pandas/numpy; avoid per-stock Python loops when possible (wide pivots or groupby are OK when needed).
6. All comments and docstrings in English. Variable names in English.
7. Top-of-file English docstring: factor formula. If unknown, state sensible default.
8. Optional: if __name__ == '__main__': only for quick test; pipeline calls compute_factor_df().

""" + DATA_LAYOUT + """

Stock ids: 000001.XSHE, 600519.XSHG.
"""


def _load_dotenv_file() -> None:
    """Parse .env file into os.environ (minimal, no-dependency parser)."""
    if not DOTENV_PATH.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(DOTENV_PATH)
        return
    except ImportError:
        pass
    for line in DOTENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_env(provider: str = "openai") -> None:
    """Load .env from project root; ensure the correct API key is present."""
    key_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if os.environ.get(key_var, "").strip():
        return
    if not DOTENV_PATH.exists():
        print(f"Missing {DOTENV_PATH} and {key_var} not set.", file=sys.stderr)
        sys.exit(1)
    _load_dotenv_file()
    if not os.environ.get(key_var, "").strip():
        print(f"{key_var} empty after loading .env", file=sys.stderr)
        sys.exit(1)


def extract_python_code(text: str) -> str:
    """If model wrapped code in ```python ... ```, unwrap."""
    text = text.strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text


_SAFE_IMPORTS = frozenset({
    "pandas", "numpy", "pathlib", "os", "os.path", "math", "datetime",
    "collections", "itertools", "functools", "typing", "warnings", "re",
    "scipy", "scipy.stats", "sklearn", "sklearn.preprocessing",
})

def validate_python_syntax(code: str) -> str | None:
    """Return error string if code has a syntax error, else None."""
    import ast
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


def validate_ast_safety(code: str) -> str | None:
    """Return error string if code imports a module outside the safe list, else None."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _SAFE_IMPORTS:
                    return f"Forbidden import: {alias.name!r} (line {node.lineno})"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top not in _SAFE_IMPORTS:
                return f"Forbidden import: from {module!r} (line {node.lineno})"
    return None


def call_openai(user_message: str, model: str) -> str:
    load_env("openai")
    try:
        from openai import OpenAI
    except ImportError:
        print("Install: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip(), timeout=120.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(user_message: str, model: str) -> str:
    load_env("anthropic")
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Install: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip(), timeout=120.0)
    resp = client.messages.create(
        model=model,
        max_tokens=8096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text


def call_llm(user_message: str, model: str, provider: str = "openai") -> str:
    """Dispatch to the appropriate LLM provider (openai or anthropic)."""
    if provider == "anthropic":
        return _call_anthropic(user_message, model)
    return call_openai(user_message, model)


def default_user_prefix(description: str) -> str:
    return (
        "Generate a single Python module for this factor (China A-shares, daily bar).\n\n"
        f"{description.strip()}\n\n"
        "You MUST implement compute_factor_df() -> pd.DataFrame as specified in the system message. "
        "Load ALL inputs from local data.pkl only (Path(__file__).resolve().parents[1] / 'data.pkl'). "
        "Do not use rqdatac or any external data API. "
        "The project will save the result to .pkl and run scripts/analysis/factor_rankic_analysis.py."
    )


def main():
    p = argparse.ArgumentParser(description="LLM agent: natural language -> factor Python code")
    p.add_argument("description", nargs="*", help="Natural language factor description")
    p.add_argument("-d", "--describe", type=str, default="", help="Factor description (alternative to positional)")
    p.add_argument("-o", "--out", type=str, default="", help="Save path (.py); default generated_factors/<timestamp>_.py")
    p.add_argument("--model", type=str, default="gpt-4.1", help="LLM model id (e.g. gpt-4.1 or claude-sonnet-4-6)")
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"],
                   help="LLM provider: openai (default) or anthropic")
    p.add_argument("-i", "--interactive", action="store_true", help="Read description from stdin until EOF")
    args = p.parse_args()

    if args.interactive:
        print("Enter factor description (Ctrl+Z then Enter on Windows to finish):", flush=True)
        description = sys.stdin.read().strip()
    else:
        description = args.describe.strip() or " ".join(args.description).strip()

    if not description:
        print("Provide a description via argv, -d, or --interactive.", file=sys.stderr)
        sys.exit(1)

    print(f"Calling {args.provider.upper()} ({args.model})...", flush=True)
    raw = call_llm(default_user_prefix(description), args.model, provider=args.provider)
    code = extract_python_code(raw)

    out_path = Path(args.out) if args.out else None
    if not out_path:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w\-]+", "_", description[:40]).strip("_") or "factor"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = GENERATED_DIR / f"{ts}_{safe}.py"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Generated by agent/factor_code_agent.py — {datetime.now().isoformat(timespec='seconds')}\n"
        f"# Model: {args.model}\n"
        f"# User request (summary): {description[:200]!r}\n\n"
    )
    syntax_err = validate_python_syntax(code)
    if syntax_err:
        print(f"WARNING: generated code has a syntax error — {syntax_err}", file=sys.stderr)
        print("The file will still be saved so you can inspect and fix it manually.", file=sys.stderr)

    safety_err = validate_ast_safety(code)
    if safety_err:
        print(f"WARNING: generated code contains a disallowed import — {safety_err}", file=sys.stderr)
        print("Review carefully before executing.", file=sys.stderr)

    out_path.write_text(header + code, encoding="utf-8")
    print(f"Saved: {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
