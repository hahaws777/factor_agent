#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Mining System with full harness engineering.

Features
--------
- Config-driven (YAML, all parameters in alpha_mining_config.yaml)
- Checkpoint / resume (save state after every generation, recover from crashes)
- Deduplication (MD5 hash of factor code; never re-evaluate the same code twice)
- Retry harness (exponential backoff on LLM calls)
- Parallel evaluation (subprocess pool with per-factor timeout)
- Population management (survivors feed next generation as few-shot examples)
- Run ledger (CSV log of every factor ever evaluated, across all runs)
- Telegram notifications after each generation
- Safety limit (max_run_hours)

Usage
-----
  cd e:\\data
  python agent/alpha_miner.py start
  python agent/alpha_miner.py start --generations 3 --per-gen 6 --model gpt-4.1
  python agent/alpha_miner.py resume --run-id mining_20260503_161200
  python agent/alpha_miner.py status
  python agent/alpha_miner.py report --run-id mining_20260503_161200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "scripts" / "analysis"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MINING_DIR = ROOT / "agent_runs" / "mining"
CONFIG_PATH = AGENT_DIR / "alpha_mining_config.yaml"

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class MiningConfig:
    # mining
    n_generations: int = 5
    factors_per_generation: int = 8
    top_k_survivors: int = 3
    min_grade: str = "B"
    min_rank_ic: float = 0.015
    diversity_penalty: bool = True
    diversity_corr_threshold: float = 0.90
    # generation
    seed_prompts: list = field(default_factory=lambda: [
        "20-day price momentum: past 20-day cumulative return, cross-sectionally ranked",
        "short-term reversal: negative 5-day return, winsorized 1-99%",
        "volume-price divergence: 10-day close change divided by 10-day average volume change",
        "20-day return volatility: rolling std of daily returns, ranked cross-sectionally",
    ])
    extra_factor_types: list = field(default_factory=lambda: [
        "momentum", "reversal", "volatility", "volume", "value_proxy", "earnings_quality"
    ])
    generation_mode: str = "dsl"  # dsl | python
    allowed_fields: list = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "amount", "vwap", "market_cap", "industry"
    ])
    max_expression_depth: int = 8
    max_expression_nodes: int = 80
    max_lookback_window: int = 252
    # mutation
    mutation_enabled: bool = True
    window_sweeps: list = field(default_factory=lambda: [5, 10, 20, 60])
    transforms: list = field(default_factory=lambda: ["winsorize", "rank", "zscore"])
    max_variants_per_factor: int = 4
    llm_refine: bool = True
    # evaluation
    data_pkl: str = "data.pkl"
    outer_workers: int = 1      # level-1: how many factors to evaluate in parallel
    eval_workers: int = 2       # level-2: IC workers per factor evaluation
    next_day_return: bool = True
    timeout_sec: int = 180
    ic_similarity_threshold: float = 0.90
    diversity_min_overlap: int = 10000
    diversity_sample_size: int = 300000
    train_start: str = ""
    train_end: str = ""
    validation_start: str = ""
    validation_end: str = ""
    test_start: str = ""
    test_end: str = ""
    recent_start: str = ""
    recent_end: str = ""
    transaction_cost_bps: float = 10.0
    max_complexity_score: int = 60
    compute_trade_metrics: bool = False
    # llm
    model: str = "gpt-4.1"
    provider: str = "openai"    # "openai" or "anthropic"
    temperature: float = 0.4
    max_retries: int = 3
    retry_backoff_sec: float = 2.0
    # harness
    checkpoint_every: int = 1
    ledger_file: str = "agent_runs/mining/factor_ledger.csv"
    dedup_on_code_hash: bool = True
    notify_telegram: bool = True
    max_run_hours: float = 4.0

    @classmethod
    def from_yaml(cls, path: Path) -> "MiningConfig":
        try:
            import yaml
        except ImportError:
            log.warning("PyYAML not installed — using default config. Run: pip install pyyaml")
            return cls()
        if not path.is_file():
            log.warning("Config not found at %s — using defaults", path)
            return cls()
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        m = raw.get("mining", {})
        cfg.n_generations = m.get("n_generations", cfg.n_generations)
        cfg.factors_per_generation = m.get("factors_per_generation", cfg.factors_per_generation)
        cfg.top_k_survivors = m.get("top_k_survivors", cfg.top_k_survivors)
        cfg.min_grade = m.get("min_grade", cfg.min_grade)
        cfg.min_rank_ic = m.get("min_rank_ic", cfg.min_rank_ic)
        cfg.diversity_penalty = m.get("diversity_penalty", cfg.diversity_penalty)
        cfg.diversity_corr_threshold = m.get("diversity_corr_threshold", cfg.diversity_corr_threshold)

        g = raw.get("generation", {})
        if g.get("seed_prompts"):
            cfg.seed_prompts = g["seed_prompts"]
        if g.get("extra_factor_types"):
            cfg.extra_factor_types = g["extra_factor_types"]
        cfg.generation_mode = g.get("mode", g.get("generation_mode", cfg.generation_mode))
        cfg.allowed_fields = g.get("allowed_fields", cfg.allowed_fields)
        cfg.max_expression_depth = g.get("max_expression_depth", cfg.max_expression_depth)
        cfg.max_expression_nodes = g.get("max_expression_nodes", cfg.max_expression_nodes)
        cfg.max_lookback_window = g.get("max_lookback_window", cfg.max_lookback_window)

        mut = raw.get("mutation", {})
        cfg.mutation_enabled = mut.get("enabled", cfg.mutation_enabled)
        cfg.window_sweeps = mut.get("window_sweeps", cfg.window_sweeps)
        cfg.transforms = mut.get("transforms", cfg.transforms)
        cfg.max_variants_per_factor = mut.get("max_variants_per_factor", cfg.max_variants_per_factor)
        cfg.llm_refine = mut.get("llm_refine", cfg.llm_refine)

        ev = raw.get("evaluation", {})
        cfg.data_pkl = ev.get("data_pkl", cfg.data_pkl)
        cfg.outer_workers = ev.get("outer_workers", cfg.outer_workers)
        cfg.eval_workers = ev.get("workers", cfg.eval_workers)
        cfg.next_day_return = ev.get("next_day_return", cfg.next_day_return)
        cfg.timeout_sec = ev.get("timeout_sec", cfg.timeout_sec)
        cfg.ic_similarity_threshold = ev.get("ic_similarity_threshold", cfg.ic_similarity_threshold)
        cfg.diversity_min_overlap = ev.get("diversity_min_overlap", cfg.diversity_min_overlap)
        cfg.diversity_sample_size = ev.get("diversity_sample_size", cfg.diversity_sample_size)
        cfg.train_start = ev.get("train_start", cfg.train_start)
        cfg.train_end = ev.get("train_end", cfg.train_end)
        cfg.validation_start = ev.get("validation_start", cfg.validation_start)
        cfg.validation_end = ev.get("validation_end", cfg.validation_end)
        cfg.test_start = ev.get("test_start", cfg.test_start)
        cfg.test_end = ev.get("test_end", cfg.test_end)
        cfg.recent_start = ev.get("recent_start", cfg.recent_start)
        cfg.recent_end = ev.get("recent_end", cfg.recent_end)
        cfg.transaction_cost_bps = ev.get("transaction_cost_bps", cfg.transaction_cost_bps)
        cfg.max_complexity_score = ev.get("max_complexity_score", cfg.max_complexity_score)
        cfg.compute_trade_metrics = ev.get("compute_trade_metrics", cfg.compute_trade_metrics)

        ll = raw.get("llm", {})
        cfg.model = ll.get("model", cfg.model)
        cfg.provider = ll.get("provider", cfg.provider)
        cfg.temperature = ll.get("temperature", cfg.temperature)
        cfg.max_retries = ll.get("max_retries", cfg.max_retries)
        cfg.retry_backoff_sec = ll.get("retry_backoff_sec", cfg.retry_backoff_sec)

        h = raw.get("harness", {})
        cfg.checkpoint_every = h.get("checkpoint_every", cfg.checkpoint_every)
        cfg.ledger_file = h.get("ledger_file", cfg.ledger_file)
        cfg.dedup_on_code_hash = h.get("dedup_on_code_hash", cfg.dedup_on_code_hash)
        cfg.notify_telegram = h.get("notify_telegram", cfg.notify_telegram)
        cfg.max_run_hours = h.get("max_run_hours", cfg.max_run_hours)
        return cfg


# ── Data model ────────────────────────────────────────────────────────────────

GRADE_ORDER = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1, "?": 0}


@dataclass
class FactorCandidate:
    name: str
    code: str
    code_hash: str
    generation: int
    parent: str | None = None
    origin: str = "llm"        # llm | mutation | seed | dsl | dsl_mutation
    family: str = "unknown"
    economic_hypothesis: str = ""
    expression: str = ""
    canonical_expression: str = ""
    expected_sign: str = "unknown"
    required_fields: list[str] = field(default_factory=list)
    lookback_windows: list[int] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    why_not_duplicate: str = ""
    grade: str = "?"
    quality_score: int = 0
    mean_rank_ic: float = 0.0
    rank_ic_ir: float = 0.0
    rank_ic_win_rate: float = 0.0
    ic_t_stat: float = 0.0
    valid_days: int = 0
    coverage: float = 0.0
    avg_cross_sectional_coverage: float = 0.0
    factor_autocorr: float = 0.0
    turnover_estimate: float = 0.0
    long_short_spread_return: float = 0.0
    cost_adjusted_long_short_return: float = 0.0
    max_drawdown_long_short: float = 0.0
    recent_rank_ic: float = 0.0
    train_rank_ic: float = 0.0
    validation_rank_ic: float = 0.0
    test_rank_ic: float = 0.0
    by_year_ic: dict[str, float] = field(default_factory=dict)
    by_regime_ic: dict[str, float] = field(default_factory=dict)
    industry_neutral_ic: float = 0.0
    size_neutral_ic: float = 0.0
    ic_series_max_similarity: float = 0.0
    complexity_score: int = 0
    alpha_direction: str = "unknown"
    recommendation: str = "investigate"
    py_path: str = ""
    pkl_path: str = ""
    ic_csv_path: str = ""
    error: str = ""
    evaluated_at: str = ""
    diversity_rejected: bool = False
    max_similarity: float = 0.0
    most_similar_to: str = ""
    safety_severity: str = ""
    safety_reasons: list[str] = field(default_factory=list)
    suspicious_patterns: list[str] = field(default_factory=list)

    def passes(self, cfg: MiningConfig) -> bool:
        grade_ok = GRADE_ORDER.get(self.grade, 0) >= GRADE_ORDER.get(cfg.min_grade, 0)
        ic_ok = abs(self.mean_rank_ic) >= cfg.min_rank_ic
        stable_ok = True
        if self.validation_rank_ic:
            stable_ok = abs(self.validation_rank_ic) >= cfg.min_rank_ic * 0.5
        if self.test_rank_ic:
            stable_ok = stable_ok and abs(self.test_rank_ic) >= cfg.min_rank_ic * 0.5
        complexity_ok = self.complexity_score <= cfg.max_complexity_score if self.complexity_score else True
        return grade_ok and ic_ok and stable_ok and complexity_ok and not self.error and not self.diversity_rejected

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("code")   # keep ledger compact
        return d


@dataclass
class MiningState:
    run_id: str
    config: dict
    started_at: str
    generations_done: int = 0
    all_candidates: list[dict] = field(default_factory=list)   # serialised FactorCandidates
    survivors: list[str] = field(default_factory=list)         # names of current survivors
    seen_hashes: list[str] = field(default_factory=list)       # dedup registry
    best_grade: str = "F"
    best_mean_ric: float = 0.0

    def save(self, run_dir: Path) -> None:
        path = run_dir / "checkpoint.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)
        log.info("Checkpoint saved: %s", path)

    @classmethod
    def load(cls, run_dir: Path) -> "MiningState":
        path = run_dir / "checkpoint.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        state = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return state


# ── Harness utilities ─────────────────────────────────────────────────────────

def _code_hash(code: str) -> str:
    return hashlib.md5(code.strip().encode()).hexdigest()[:12]


def _notify(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    try:
        import urllib.request
        body = json.dumps({"chat_id": chat_id, "text": message}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)


def _call_llm_with_retry(user_message: str, model: str, cfg: MiningConfig) -> str:
    """Call LLM with exponential-backoff retries."""
    from factor_code_agent import call_llm
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            return call_llm(user_message, model, provider=cfg.provider)
        except Exception as e:
            last_err = e
            wait = cfg.retry_backoff_sec * (2 ** attempt)
            log.warning("LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt + 1, cfg.max_retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"LLM failed after {cfg.max_retries} attempts: {last_err}")


def _append_ledger(candidate: FactorCandidate, ledger_path: Path) -> None:
    """Append one row to the all-time factor ledger CSV."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ledger_path.is_file()
    row = candidate.to_dict()
    with open(ledger_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── LLM prompt builders ───────────────────────────────────────────────────────

def _build_generation_prompt(
    description: str,
    survivors: list[FactorCandidate],
    factor_type_hint: str = "",
) -> str:
    base = (
        "Generate a single Python module for this factor (China A-shares, daily bar).\n\n"
        f"{description.strip()}\n"
    )
    if factor_type_hint:
        base += f"Factor category hint: {factor_type_hint}\n"

    if survivors:
        base += "\nHere are EXAMPLES of high-quality factors from this project (learn the pattern but generate something NEW and DIFFERENT):\n\n"
        for s in survivors[:3]:
            base += f"# Example — {s.name} (grade={s.grade}, mean_rank_ic={s.mean_rank_ic:.4f})\n"
            base += s.code[:800] + "\n\n"
        base += "Generate a factor that is meaningfully DIFFERENT from the examples above.\n"

    base += (
        "\nYou MUST implement compute_factor_df() -> pd.DataFrame as specified. "
        "Load ALL inputs from local data.pkl only. Find root by walking up: "
        "_p=Path(__file__).resolve().parent; "
        "[_p:=_p.parent for _ in range(6) if not (_p/'data.pkl').is_file()]; root=_p. "
        "Do not use rqdatac or any external data API."
    )
    return base


def _build_dsl_generation_prompt(
    description: str,
    survivors: list[FactorCandidate],
    factor_type_hint: str,
    cfg: MiningConfig,
) -> str:
    families = " | ".join(cfg.extra_factor_types + ["liquidity", "quality_proxy", "composite"])
    fields = ", ".join(cfg.allowed_fields)
    operators = (
        "delay(x,n), delta(x,n), ts_return(x,n), ts_mean(x,n), ts_std(x,n), "
        "ts_rank(x,n), ts_zscore(x,n), ts_corr(x,y,n), rank(x), zscore(x), "
        "winsorize(x,lower=0.01,upper=0.99), signed_power(x,p), log1p_abs(x), "
        "neutralize_industry(x), neutralize_size(x)"
    )
    base = f"""Generate ONE alpha factor specification as strict JSON only.

Description: {description.strip()}
Preferred family: {factor_type_hint}

Allowed families: {families}
Allowed raw fields: {fields}
Allowed operators: {operators}
Rules:
- Do not output Python code.
- Do not use future returns, labels, target columns, negative shifts, centered rolling, or full-sample normalization.
- Window sizes must be positive integers <= {cfg.max_lookback_window}.
- The expression must be a single DSL expression using only allowed fields/operators and +, -, *, /.
- Use ts_return(close, n) for trailing returns.
- Explain why the factor is economically different from prior survivors.

Required JSON schema:
{{
  "name": "ascii_snake_case_name",
  "family": "momentum | reversal | volatility | volume | liquidity | value_proxy | quality_proxy | composite",
  "economic_hypothesis": "...",
  "expression": "...",
  "expected_sign": "positive | negative | unknown",
  "required_fields": ["close", "volume"],
  "lookback_windows": [5, 20],
  "risk_notes": ["..."],
  "why_not_duplicate": "..."
}}
"""
    if survivors:
        base += "\nExisting accepted factors to avoid duplicating:\n"
        for s in survivors[:8]:
            detail = s.expression or s.economic_hypothesis or s.name
            base += f"- {s.name}: family={s.family}, sign={s.expected_sign}, expr={detail[:240]}\n"
    return base


# ── Factor evaluation ─────────────────────────────────────────────────────────

def _patch_data_root(code: str, root: Path) -> str:
    """Rewrite any `parents[N]`-based root assignment to use the known absolute root.

    Generated code often uses parents[1] which breaks when the .py is saved
    several levels deep (e.g. agent_runs/mining/<run>/factors/<name>.py).
    """
    import re
    abs_root = str(root.resolve()).replace("\\", "/")
    patched = re.sub(
        r'(root\s*=\s*)Path\(__file__\)\.resolve\(\)\.parents\[\d+\]',
        rf'\1Path(r"{abs_root}")',
        code,
    )
    return patched


def _evaluate_candidate(
    candidate: FactorCandidate,
    run_dir: Path,
    cfg: MiningConfig,
) -> FactorCandidate:
    """Run one factor through the full pipeline: pickle → Rank IC → screen."""
    from factor_screener import screen_ic_csv
    from factor_safety import validate_factor_code

    py_path = run_dir / "factors" / f"{candidate.name}.py"
    pkl_path = run_dir / "factors" / f"{candidate.name}.pkl"
    ic_csv = run_dir / "ic" / f"{candidate.name}_rankic.csv"
    py_path.parent.mkdir(parents=True, exist_ok=True)
    ic_csv.parent.mkdir(parents=True, exist_ok=True)

    py_path.write_text(_patch_data_root(candidate.code, ROOT), encoding="utf-8")
    candidate.py_path = str(py_path)
    candidate.pkl_path = str(pkl_path)
    candidate.ic_csv_path = str(ic_csv)

    safety = validate_factor_code(candidate.code)
    candidate.safety_severity = safety.severity
    candidate.safety_reasons = safety.reasons
    candidate.suspicious_patterns = safety.suspicious_patterns
    if not safety.is_safe:
        py_path.write_text(_patch_data_root(candidate.code, ROOT), encoding="utf-8")
        candidate.error = "rejected by safety validator: " + "; ".join(safety.reasons[:5])
        candidate.evaluated_at = datetime.utcnow().isoformat()
        log.warning("Safety rejected %s: %s", candidate.name, candidate.error)
        return candidate

    # Step 1: compute factor and save pkl
    pipe_script = AGENT_DIR / "factor_agent_pipeline.py"
    env = os.environ.copy()
    env["FACTOR_DATA_ROOT"] = str(ROOT.resolve())
    t0 = time.perf_counter()
    try:
        rc = subprocess.call(
            [sys.executable, str(pipe_script),
             "--skip-generate", str(py_path),
             "--data", cfg.data_pkl,
             "--pkl-out", str(pkl_path),
             "--rankic-out", str(ic_csv),
             "--no-backtest", "--no-plot-backtest",
             "--workers", str(cfg.eval_workers),
             *(["--no-next-day"] if not cfg.next_day_return else []),
             ],
            cwd=str(ROOT),
            timeout=cfg.timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        candidate.error = f"timeout after {cfg.timeout_sec}s"
        log.warning("Timeout: %s", candidate.name)
        return candidate
    except Exception as e:
        candidate.error = str(e)
        log.error("Eval error for %s: %s", candidate.name, e)
        return candidate

    elapsed = time.perf_counter() - t0

    if rc != 0:
        candidate.error = f"pipeline exit {rc}"
        return candidate

    # Step 2: screen
    if ic_csv.is_file():
        result = screen_ic_csv(str(ic_csv), factor_name=candidate.name)
        candidate.grade = result.get("grade", "?")
        candidate.quality_score = result.get("score", 0)
        m = result.get("metrics", {})
        candidate.mean_rank_ic = float(m.get("mean_rank_ic") or 0)
        candidate.rank_ic_ir = float(m.get("rank_ic_ir") or 0)
        candidate.rank_ic_win_rate = float(m.get("rank_ic_win_rate") or 0)
        candidate.valid_days = int(m.get("valid_days") or 0)
        _enrich_candidate_metrics(candidate, cfg)
    else:
        candidate.error = "no IC CSV produced"

    candidate.evaluated_at = datetime.utcnow().isoformat()
    log.info("Evaluated %-40s grade=%-2s ric=%.4f  ir=%.3f  %.0fs",
             candidate.name, candidate.grade, candidate.mean_rank_ic,
             candidate.rank_ic_ir, elapsed)
    return candidate


def _candidate_rank_key(c: FactorCandidate) -> tuple[int, float, float, float, int]:
    validation_bonus = abs(c.validation_rank_ic) if c.validation_rank_ic else 0.0
    recent_bonus = abs(c.recent_rank_ic) if c.recent_rank_ic else 0.0
    return (
        GRADE_ORDER.get(c.grade, 0),
        validation_bonus,
        recent_bonus,
        abs(c.mean_rank_ic),
        -c.complexity_score,
    )


def _enrich_candidate_metrics(candidate: FactorCandidate, cfg: MiningConfig) -> None:
    """Add robust but optional metrics while preserving the old Rank IC fields."""
    _enrich_ic_metrics(candidate, cfg)
    _enrich_factor_panel_metrics(candidate, cfg)
    if cfg.compute_trade_metrics:
        _enrich_trade_metrics(candidate, cfg)
    candidate.alpha_direction = "inverse_alpha" if candidate.mean_rank_ic < 0 else "positive_alpha"
    warnings = []
    if candidate.complexity_score > cfg.max_complexity_score:
        warnings.append("high complexity")
    if candidate.validation_rank_ic and abs(candidate.validation_rank_ic) < cfg.min_rank_ic * 0.5:
        warnings.append("weak validation IC")
    if candidate.test_rank_ic and abs(candidate.test_rank_ic) < cfg.min_rank_ic * 0.5:
        warnings.append("weak test IC")
    if candidate.turnover_estimate and candidate.turnover_estimate > 0.8:
        warnings.append("high turnover")
    if candidate.passes(cfg):
        candidate.recommendation = "keep"
    elif warnings:
        candidate.recommendation = "investigate: " + ", ".join(warnings)
    else:
        candidate.recommendation = "reject"


def _enrich_ic_metrics(candidate: FactorCandidate, cfg: MiningConfig) -> None:
    import numpy as np
    import pandas as pd

    try:
        df = pd.read_csv(candidate.ic_csv_path)
    except Exception as e:
        log.warning("Could not read IC CSV for advanced metrics (%s): %s", candidate.name, e)
        return
    if "date" not in df.columns or "rank_ic" not in df.columns:
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rank_ic"] = pd.to_numeric(df["rank_ic"], errors="coerce")
    ric = df["rank_ic"].dropna()
    if len(ric) > 1:
        std = float(ric.std())
        candidate.ic_t_stat = float(ric.mean() / std * np.sqrt(len(ric))) if std > 0 else 0.0
    candidate.by_year_ic = {
        str(int(year)): float(group["rank_ic"].mean())
        for year, group in df.dropna(subset=["date"]).groupby(df["date"].dt.year)
        if group["rank_ic"].notna().any()
    }

    candidate.train_rank_ic = _window_mean_rank_ic(df, cfg.train_start, cfg.train_end)
    candidate.validation_rank_ic = _window_mean_rank_ic(df, cfg.validation_start, cfg.validation_end)
    candidate.test_rank_ic = _window_mean_rank_ic(df, cfg.test_start, cfg.test_end)
    candidate.recent_rank_ic = _window_mean_rank_ic(df, cfg.recent_start, cfg.recent_end)
    if not candidate.recent_rank_ic and len(df) >= 60:
        candidate.recent_rank_ic = float(df.tail(60)["rank_ic"].mean())


def _window_mean_rank_ic(df, start: str, end: str) -> float:
    import pandas as pd

    if not start and not end:
        return 0.0
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["date"] >= pd.Timestamp(start)
    if end:
        mask &= df["date"] <= pd.Timestamp(end)
    s = df.loc[mask, "rank_ic"].dropna()
    return float(s.mean()) if len(s) else 0.0


def _enrich_factor_panel_metrics(candidate: FactorCandidate, cfg: MiningConfig) -> None:
    import pandas as pd

    path = Path(candidate.pkl_path)
    if not path.is_file():
        return
    try:
        fac = pd.read_pickle(path)
    except Exception as e:
        log.warning("Could not read factor pkl for advanced metrics (%s): %s", candidate.name, e)
        return
    if not isinstance(fac, pd.DataFrame) or fac.empty:
        return
    s = pd.to_numeric(fac.iloc[:, 0], errors="coerce")
    candidate.coverage = float(s.notna().mean())
    try:
        by_date = s.rename("factor").reset_index().groupby("date")["factor"].agg(["count", "size"])
        daily_cov = by_date["count"] / by_date["size"].replace(0, pd.NA)
        candidate.avg_cross_sectional_coverage = float(daily_cov.mean())
    except Exception:
        pass
    try:
        wide = s.unstack("order_book_id").sort_index()
        candidate.factor_autocorr = float(wide.corrwith(wide.shift(1), axis=1).mean())
        ranks = wide.rank(axis=1, pct=True)
        candidate.turnover_estimate = float((ranks - ranks.shift(1)).abs().mean(axis=1).mean())
    except Exception as e:
        log.debug("Could not compute factor autocorr/turnover for %s: %s", candidate.name, e)


def _enrich_trade_metrics(candidate: FactorCandidate, cfg: MiningConfig) -> None:
    import numpy as np
    import pandas as pd

    factor_path = Path(candidate.pkl_path)
    data_path = ROOT / cfg.data_pkl
    if not factor_path.is_file() or not data_path.is_file():
        return
    try:
        fac = pd.read_pickle(factor_path)
        data = pd.read_pickle(data_path)
    except Exception as e:
        log.warning("Could not load data for trade metrics (%s): %s", candidate.name, e)
        return
    required = {"order_book_id", "date", "close"}
    if not required.issubset(data.columns) or not isinstance(fac, pd.DataFrame):
        return
    try:
        f = fac.iloc[:, 0].rename("factor").reset_index()
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        f["order_book_id"] = f["order_book_id"].astype(str)
        px = data[["order_book_id", "date", "close"]].copy()
        px["date"] = pd.to_datetime(px["date"]).dt.normalize()
        px["order_book_id"] = px["order_book_id"].astype(str)
        px = px.sort_values(["order_book_id", "date"])
        px["ret_fwd"] = px.groupby("order_book_id")["close"].pct_change().groupby(px["order_book_id"]).shift(-1)
        merged = f.merge(px[["order_book_id", "date", "ret_fwd"]], on=["order_book_id", "date"], how="inner").dropna()
        if merged.empty:
            return
        merged["rank"] = merged.groupby("date")["factor"].rank(pct=True)
        long_mask = merged["rank"] >= 0.9
        short_mask = merged["rank"] <= 0.1
        daily_long = merged.loc[long_mask].groupby("date")["ret_fwd"].mean()
        daily_short = merged.loc[short_mask].groupby("date")["ret_fwd"].mean()
        ls = (daily_long - daily_short).dropna()
        if candidate.mean_rank_ic < 0:
            ls = -ls
        if ls.empty:
            return
        cost = (cfg.transaction_cost_bps / 10000.0) * 2.0 * (candidate.turnover_estimate or 0.0)
        cost_adj = ls - cost
        equity = (1.0 + cost_adj.fillna(0)).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        candidate.long_short_spread_return = float(ls.mean())
        candidate.cost_adjusted_long_short_return = float(cost_adj.mean())
        candidate.max_drawdown_long_short = float(drawdown.min()) if len(drawdown) else 0.0
    except Exception as e:
        log.warning("Could not compute trade metrics for %s: %s", candidate.name, e)


def _load_factor_series(candidate: FactorCandidate, cfg: MiningConfig):
    import pandas as pd

    if not candidate.pkl_path:
        return None
    path = Path(candidate.pkl_path)
    if not path.is_file():
        return None
    try:
        df = pd.read_pickle(path)
    except Exception as e:
        log.warning("Could not read factor pkl for diversity check (%s): %s", candidate.name, e)
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    series = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    if cfg.diversity_sample_size and len(series) > cfg.diversity_sample_size:
        series = series.sample(n=cfg.diversity_sample_size, random_state=7).sort_index()
    return series


def _factor_similarity(left: FactorCandidate, right: FactorCandidate, cfg: MiningConfig, cache: dict[str, Any]) -> float | None:
    import numpy as np

    if left.name not in cache:
        cache[left.name] = _load_factor_series(left, cfg)
    if right.name not in cache:
        cache[right.name] = _load_factor_series(right, cfg)
    left_s = cache[left.name]
    right_s = cache[right.name]
    if left_s is None or right_s is None:
        return None
    aligned = left_s.rename("left").to_frame().join(right_s.rename("right"), how="inner").dropna()
    if len(aligned) < cfg.diversity_min_overlap:
        return None
    corr = aligned["left"].corr(aligned["right"], method="spearman")
    if corr is None or np.isnan(corr):
        return None
    return float(abs(corr))


def _ic_series_similarity(left: FactorCandidate, right: FactorCandidate) -> float | None:
    import numpy as np
    import pandas as pd

    if not left.ic_csv_path or not right.ic_csv_path:
        return None
    lp = Path(left.ic_csv_path)
    rp = Path(right.ic_csv_path)
    if not lp.is_file() or not rp.is_file():
        return None
    try:
        ldf = pd.read_csv(lp, usecols=["date", "rank_ic"])
        rdf = pd.read_csv(rp, usecols=["date", "rank_ic"])
    except Exception:
        return None
    ldf["date"] = pd.to_datetime(ldf["date"], errors="coerce")
    rdf["date"] = pd.to_datetime(rdf["date"], errors="coerce")
    aligned = ldf.merge(rdf, on="date", suffixes=("_l", "_r")).dropna()
    if len(aligned) < 30:
        return None
    corr = aligned["rank_ic_l"].corr(aligned["rank_ic_r"], method="spearman")
    if corr is None or np.isnan(corr):
        return None
    return float(abs(corr))


# ── Main mining loop ──────────────────────────────────────────────────────────

class AlphaMiner:

    def __init__(self, cfg: MiningConfig, run_id: str | None = None):
        self.cfg = cfg
        self.run_id = run_id or f"mining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = MINING_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = ROOT / cfg.ledger_file
        self.start_wall = time.time()
        self._state: MiningState | None = None
        # Copy config into run dir for reproducibility
        if CONFIG_PATH.is_file():
            shutil.copy2(CONFIG_PATH, self.run_dir / "config_used.yaml")

    # ── State ──────────────────────────────────────────────────────────────

    def _init_state(self) -> MiningState:
        return MiningState(
            run_id=self.run_id,
            config=asdict(self.cfg),
            started_at=datetime.utcnow().isoformat(),
        )

    def _checkpoint(self) -> None:
        if self._state:
            self._state.save(self.run_dir)

    def _load_checkpoint(self) -> MiningState:
        return MiningState.load(self.run_dir)

    def _candidates_from_state(self) -> list[FactorCandidate]:
        return [FactorCandidate(**{k: v for k, v in d.items()
                                    if k in FactorCandidate.__dataclass_fields__})
                for d in self._state.all_candidates]

    def _survivors_from_state(self) -> list[FactorCandidate]:
        all_c = {c.name: c for c in self._candidates_from_state()}
        return [all_c[n] for n in self._state.survivors if n in all_c]

    # ── Generation ────────────────────────────────────────────────────────

    def _generate_new_factors(
        self,
        survivors: list[FactorCandidate],
        generation: int,
        n: int,
    ) -> list[FactorCandidate]:
        """Generate n new factor candidates via LLM."""
        from factor_code_agent import extract_python_code, validate_python_syntax
        from factor_dsl import DSLConfig, FactorSpec, compile_expression_to_module, validate_expression

        candidates = []
        factor_types = self.cfg.extra_factor_types
        dsl_cfg = DSLConfig(
            allowed_fields=set(self.cfg.allowed_fields),
            max_window=self.cfg.max_lookback_window,
            max_depth=self.cfg.max_expression_depth,
            max_nodes=self.cfg.max_expression_nodes,
        )

        # Use seed prompts for gen-0; diversified prompts for later generations
        if generation == 0:
            base_prompts = list(self.cfg.seed_prompts)
        else:
            base_prompts = [
                f"Generate a novel {ft} factor for A-shares"
                for ft in factor_types
            ]

        import itertools
        prompt_cycle = itertools.cycle(base_prompts)

        for i in range(n):
            desc = next(prompt_cycle)
            ft_hint = factor_types[i % len(factor_types)]
            if self.cfg.generation_mode == "dsl":
                user_msg = _build_dsl_generation_prompt(desc, survivors, ft_hint, self.cfg)
            else:
                user_msg = _build_generation_prompt(desc, survivors, ft_hint)

            log.info("Generating factor %d/%d (gen %d)", i + 1, n, generation)
            try:
                raw = _call_llm_with_retry(user_msg, self.cfg.model, self.cfg)
            except Exception as e:
                log.error("LLM generation failed: %s", e)
                continue

            spec = None
            validation = None
            if self.cfg.generation_mode == "dsl":
                try:
                    spec = FactorSpec.from_json_text(raw)
                    validation = validate_expression(spec.expression, dsl_cfg)
                    if not validation.is_valid:
                        log.warning("Invalid DSL expression from LLM: %s", "; ".join(validation.errors))
                        continue
                    code = compile_expression_to_module(spec, factor_name=spec.name, cfg=dsl_cfg)
                except Exception as e:
                    log.warning("Could not parse/compile LLM DSL JSON: %s", e)
                    continue
            else:
                code = extract_python_code(raw)
                err = validate_python_syntax(code)
                if err:
                    log.warning("Syntax error in generated code: %s", err)
                    continue

            h = _code_hash(code)
            if self.cfg.dedup_on_code_hash and h in self._state.seen_hashes:
                log.info("Duplicate code hash %s — skipping", h)
                continue

            base_name = spec.name if spec else "llm"
            name = f"gen{generation}_{base_name}_{i:02d}_{h[:6]}"
            c = FactorCandidate(
                name=name, code=code, code_hash=h,
                generation=generation,
                origin="dsl" if spec else "llm",
                family=spec.family if spec else ft_hint,
                economic_hypothesis=spec.economic_hypothesis if spec else "",
                expression=spec.expression if spec else "",
                canonical_expression=validation.canonical_expression if validation else "",
                expected_sign=spec.expected_sign if spec else "unknown",
                required_fields=validation.required_fields if validation else [],
                lookback_windows=validation.lookback_windows if validation else [],
                risk_notes=spec.risk_notes if spec else [],
                why_not_duplicate=spec.why_not_duplicate if spec else "",
                complexity_score=validation.complexity_score if validation else 0,
            )
            candidates.append(c)
            self._state.seen_hashes.append(h)

        return candidates

    def _mutate_survivors(
        self,
        survivors: list[FactorCandidate],
        generation: int,
    ) -> list[FactorCandidate]:
        """Generate mutations of survivors."""
        if not self.cfg.mutation_enabled or not survivors:
            return []

        from factor_dsl import DSLConfig, FactorSpec, compile_expression_to_module, validate_expression
        from factor_mutator import FactorMutator, mutate_dsl_expression

        def _llm_fn(msg, model):
            return _call_llm_with_retry(msg, model, self.cfg)

        mutator = FactorMutator(
            window_sweeps=self.cfg.window_sweeps,
            transforms=self.cfg.transforms,
            max_variants=self.cfg.max_variants_per_factor,
            llm_model=self.cfg.model,
            llm_temperature=self.cfg.temperature,
        )

        results = []
        for s in survivors:
            if self.cfg.generation_mode == "dsl" and s.expression:
                dsl_cfg = DSLConfig(
                    allowed_fields=set(self.cfg.allowed_fields),
                    max_window=self.cfg.max_lookback_window,
                    max_depth=self.cfg.max_expression_depth,
                    max_nodes=self.cfg.max_expression_nodes,
                )
                variants = mutate_dsl_expression(
                    s.expression,
                    windows=self.cfg.window_sweeps,
                    transforms=self.cfg.transforms,
                    include_sign_flip=s.expected_sign == "negative",
                )
                for vname, expr in variants[: self.cfg.max_variants_per_factor]:
                    validation = validate_expression(expr, dsl_cfg)
                    if not validation.is_valid:
                        log.debug("Skipping invalid DSL mutation %s: %s", vname, "; ".join(validation.errors))
                        continue
                    spec = FactorSpec(
                        name=f"{s.name}_{vname}",
                        family=s.family,
                        economic_hypothesis=s.economic_hypothesis,
                        expression=expr,
                        expected_sign=s.expected_sign,
                        required_fields=validation.required_fields,
                        lookback_windows=validation.lookback_windows,
                        risk_notes=list(s.risk_notes),
                        why_not_duplicate=f"DSL mutation {vname} from {s.name}",
                    )
                    try:
                        vcode = compile_expression_to_module(spec, factor_name=spec.name, cfg=dsl_cfg)
                    except Exception as e:
                        log.debug("Could not compile DSL mutation %s: %s", vname, e)
                        continue
                    h = _code_hash(vcode)
                    if self.cfg.dedup_on_code_hash and h in self._state.seen_hashes:
                        continue
                    c = FactorCandidate(
                        name=f"gen{generation}_dsl_mut_{vname[:28]}_{h[:6]}",
                        code=vcode,
                        code_hash=h,
                        generation=generation,
                        parent=s.name,
                        origin="dsl_mutation",
                        family=s.family,
                        economic_hypothesis=s.economic_hypothesis,
                        expression=expr,
                        canonical_expression=validation.canonical_expression,
                        expected_sign=s.expected_sign,
                        required_fields=validation.required_fields,
                        lookback_windows=validation.lookback_windows,
                        risk_notes=list(s.risk_notes),
                        why_not_duplicate=spec.why_not_duplicate,
                        complexity_score=validation.complexity_score,
                    )
                    results.append(c)
                    self._state.seen_hashes.append(h)
                continue

            variants = mutator.generate_all(
                code=s.code,
                base_name=s.name,
                grade=s.grade,
                metrics={"mean_rank_ic": s.mean_rank_ic,
                         "rank_ic_ir": s.rank_ic_ir,
                         "rank_ic_win_rate": s.rank_ic_win_rate},
                call_llm_fn=_llm_fn if self.cfg.llm_refine else None,
            )
            for vname, vcode in variants:
                h = _code_hash(vcode)
                if self.cfg.dedup_on_code_hash and h in self._state.seen_hashes:
                    continue
                c = FactorCandidate(
                    name=f"gen{generation}_mut_{vname[:40]}_{h[:6]}",
                    code=vcode, code_hash=h,
                    generation=generation, parent=s.name, origin="mutation",
                )
                results.append(c)
                self._state.seen_hashes.append(h)
        log.info("Mutation produced %d new candidates from %d survivors", len(results), len(survivors))
        return results

    def _evaluate_batch(self, candidates: list[FactorCandidate]) -> list[FactorCandidate]:
        """Evaluate candidates with optional outer parallelism (outer_workers)."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_parallel = max(1, min(len(candidates), self.cfg.outer_workers))
        results: list[FactorCandidate] = []
        ledger_lock = threading.Lock()
        stop_early = threading.Event()

        def _run_one(i: int, c: FactorCandidate) -> FactorCandidate:
            if stop_early.is_set():
                c.error = "cancelled: wall-clock limit"
                return c
            log.info("Evaluating %d/%d: %s", i + 1, len(candidates), c.name)
            return _evaluate_candidate(c, self.run_dir, self.cfg)

        if n_parallel == 1:
            for i, c in enumerate(candidates):
                result = _run_one(i, c)
                results.append(result)
                _append_ledger(result, self.ledger_path)
                if (time.time() - self.start_wall) / 3600 > self.cfg.max_run_hours:
                    log.warning("Max run hours (%.1f) reached — stopping evaluation early",
                                self.cfg.max_run_hours)
                    stop_early.set()
                    break
        else:
            log.info("Parallel evaluation: %d candidates × %d outer workers", len(candidates), n_parallel)
            with ThreadPoolExecutor(max_workers=n_parallel) as pool:
                future_map = {pool.submit(_run_one, i, c): c for i, c in enumerate(candidates)}
                for future in as_completed(future_map):
                    result = future.result()
                    results.append(result)
                    with ledger_lock:
                        _append_ledger(result, self.ledger_path)
                    if (time.time() - self.start_wall) / 3600 > self.cfg.max_run_hours:
                        log.warning("Max run hours (%.1f) reached — cancelling remaining",
                                    self.cfg.max_run_hours)
                        stop_early.set()
                        for f in future_map:
                            f.cancel()
                        break

        return results

    def _apply_diversity_filter(
        self,
        candidates: list[FactorCandidate],
        reference: list[FactorCandidate],
    ) -> list[FactorCandidate]:
        """Reject duplicate variants by expression, factor values, family pressure, and IC series."""
        if not self.cfg.diversity_penalty:
            return candidates

        candidates = sorted(candidates, key=_candidate_rank_key, reverse=True)
        accepted = [c for c in reference if c.passes(self.cfg)]
        accepted_expr = {re.sub(r"\s+", "", c.canonical_expression or c.expression) for c in accepted if c.expression}
        cache: dict[str, Any] = {}

        for c in candidates:
            if c.error or c.diversity_rejected:
                continue

            compact_expr = re.sub(r"\s+", "", c.canonical_expression or c.expression)
            if compact_expr and compact_expr in accepted_expr:
                c.diversity_rejected = True
                c.most_similar_to = "same_expression"
                c.max_similarity = 1.0
                log.info("Diversity reject %-40s duplicate DSL expression", c.name)
                continue

            best_name = ""
            best_corr = 0.0
            best_ic_corr = 0.0
            for other in accepted:
                value_corr = _factor_similarity(c, other, self.cfg, cache)
                if value_corr is not None and value_corr > best_corr:
                    best_corr = value_corr
                    best_name = other.name
                ic_corr = _ic_series_similarity(c, other)
                if ic_corr is not None and ic_corr > best_ic_corr:
                    best_ic_corr = ic_corr

            c.max_similarity = best_corr
            c.ic_series_max_similarity = best_ic_corr
            c.most_similar_to = best_name
            if best_corr >= self.cfg.diversity_corr_threshold:
                c.diversity_rejected = True
                log.info("Diversity reject %-40s value_corr=%.3f vs %s", c.name, best_corr, best_name)
                continue
            if best_ic_corr >= self.cfg.ic_similarity_threshold:
                c.diversity_rejected = True
                c.most_similar_to = best_name or "similar_ic_series"
                log.info("Diversity reject %-40s ic_series_corr=%.3f", c.name, best_ic_corr)
                continue

            if c.passes(self.cfg):
                accepted.append(c)
                if compact_expr:
                    accepted_expr.add(compact_expr)

        return candidates

    def _select_survivors(
        self,
        candidates: list[FactorCandidate],
        prev_survivors: list[FactorCandidate],
    ) -> list[FactorCandidate]:
        """Family-aware survivor selection from new candidates and previous survivors."""
        pool = candidates + prev_survivors
        passing = [c for c in pool if c.passes(self.cfg)]

        if not passing:
            log.warning("No factors passed quality gates this generation — keeping best available")
            return sorted(pool, key=_candidate_rank_key, reverse=True)[: self.cfg.top_k_survivors]

        ranked = sorted(passing, key=_candidate_rank_key, reverse=True)
        selected: list[FactorCandidate] = []

        def add(c: FactorCandidate | None) -> None:
            if c and c.name not in {s.name for s in selected} and len(selected) < self.cfg.top_k_survivors:
                selected.append(c)

        add(ranked[0])  # best overall

        best_by_family: dict[str, FactorCandidate] = {}
        for c in ranked:
            fam = c.family or "unknown"
            if fam not in best_by_family:
                best_by_family[fam] = c
        for c in sorted(best_by_family.values(), key=_candidate_rank_key, reverse=True):
            add(c)

        low_corr = sorted(
            passing,
            key=lambda c: (c.max_similarity or 0.0, -abs(c.mean_rank_ic), c.complexity_score),
        )
        add(low_corr[0] if low_corr else None)

        recent = sorted(passing, key=lambda c: abs(c.recent_rank_ic or c.mean_rank_ic), reverse=True)
        add(recent[0] if recent else None)

        exploratory = [
            c for c in pool
            if not c.error and not c.diversity_rejected and c.name not in {s.name for s in selected}
        ]
        exploratory.sort(key=lambda c: (abs(c.recent_rank_ic), -c.complexity_score, abs(c.mean_rank_ic)), reverse=True)
        add(exploratory[0] if exploratory else None)

        for c in ranked:
            add(c)

        return selected[: self.cfg.top_k_survivors]

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self, resume: bool = False) -> None:
        os.chdir(ROOT)

        if resume:
            log.info("Resuming run: %s", self.run_id)
            self._state = self._load_checkpoint()
            start_gen = self._state.generations_done
            survivors = self._survivors_from_state()
        else:
            self._state = self._init_state()
            start_gen = 0
            survivors = []
            log.info("Starting new run: %s", self.run_id)
            log.info("Config: %d generations × %d factors, top-%d survivors",
                     self.cfg.n_generations, self.cfg.factors_per_generation, self.cfg.top_k_survivors)

        _notify(f"Alpha mining started: {self.run_id}\n"
                f"{self.cfg.n_generations} generations × {self.cfg.factors_per_generation} factors each")

        for gen in range(start_gen, self.cfg.n_generations):
            if (time.time() - self.start_wall) / 3600 > self.cfg.max_run_hours:
                log.warning("Max run hours reached before generation %d — stopping", gen)
                break

            log.info("=" * 60)
            log.info("GENERATION %d / %d", gen, self.cfg.n_generations - 1)
            log.info("=" * 60)

            # Generate
            n_llm = max(1, self.cfg.factors_per_generation - len(survivors) * self.cfg.max_variants_per_factor)
            new_candidates = self._generate_new_factors(survivors, gen, n_llm)
            if self.cfg.mutation_enabled and survivors:
                new_candidates += self._mutate_survivors(survivors, gen)

            if not new_candidates:
                log.warning("Generation %d: no new candidates generated — skipping", gen)
                self._state.generations_done = gen + 1
                continue

            log.info("Generation %d: %d candidates to evaluate", gen, len(new_candidates))

            # Evaluate
            evaluated = self._evaluate_batch(new_candidates)
            evaluated = self._apply_diversity_filter(evaluated, self._candidates_from_state())

            # Update state
            for c in evaluated:
                self._state.all_candidates.append(asdict(c))

            # Select survivors
            survivors = self._select_survivors(evaluated, survivors)
            self._state.survivors = [s.name for s in survivors]
            self._state.generations_done = gen + 1

            # Update best
            if survivors:
                best = survivors[0]
                self._state.best_grade = best.grade
                self._state.best_mean_ric = best.mean_rank_ic

            # Checkpoint
            if gen % self.cfg.checkpoint_every == 0:
                self._checkpoint()

            # Generation summary
            passing = [c for c in evaluated if c.passes(self.cfg)]
            rejected = [c for c in evaluated if c.diversity_rejected]
            unsafe = [c for c in evaluated if c.error.startswith("rejected by safety validator")]
            summary = (
                f"Gen {gen} complete — {len(evaluated)} evaluated, {len(passing)} passed, "
                f"{len(rejected)} diversity-rejected, {len(unsafe)} unsafe\n"
                f"Survivors: "
                + ", ".join(
                    f"{s.name}(family={s.family}, grade={s.grade}, ric={s.mean_rank_ic:.4f}, dir={s.alpha_direction})"
                    for s in survivors
                )
            )
            log.info(summary)
            if self.cfg.notify_telegram:
                _notify(f"[{self.run_id}] {summary}")

        # Final checkpoint
        self._checkpoint()
        report = self._write_report()
        log.info("Mining complete. Report: %s", report)
        _notify(f"Mining complete: {self.run_id}\nBest grade: {self._state.best_grade}  "
                f"Best Rank IC: {self._state.best_mean_ric:.4f}\nReport: {report}")

    # ── Reporting ─────────────────────────────────────────────────────────

    def _write_report(self) -> Path:
        """Write a markdown report summarising the full mining run."""
        if not self._state:
            return self.run_dir / "report.md"

        all_c = self._candidates_from_state()
        evaluated = [c for c in all_c if not c.error]
        unsafe = [c for c in all_c if c.error.startswith("rejected by safety validator")]
        diversity_rejected = [c for c in evaluated if c.diversity_rejected]
        passing = [c for c in evaluated if c.passes(self.cfg)]
        passing_sorted = sorted(passing, key=_candidate_rank_key, reverse=True)

        # Grade distribution
        from collections import Counter
        grade_dist = Counter(c.grade for c in evaluated)
        family_dist = Counter(c.family or "unknown" for c in passing)

        lines = [
            f"# Alpha Mining Report — {self.run_id}",
            f"",
            f"**Started:** {self._state.started_at}  ",
            f"**Generations completed:** {self._state.generations_done}/{self.cfg.n_generations}  ",
            f"**Total evaluated:** {len(evaluated)}  |  **Passed:** {len(passing)}  ",
            f"**Unsafe rejected:** {len(unsafe)}  |  **Diversity rejected:** {len(diversity_rejected)}  ",
            f"**Best grade:** {self._state.best_grade}  |  **Best Rank IC:** {self._state.best_mean_ric:.4f}",
            f"",
            f"## Grade Distribution",
            f"",
            "| Grade | Count |",
            "|-------|-------|",
        ]
        for g in ["A", "B", "C", "D", "F"]:
            lines.append(f"| {g} | {grade_dist.get(g, 0)} |")

        lines += ["", "## Family Distribution", "", "| Family | Passing Count |", "|--------|---------------|"]
        for family, count in family_dist.most_common():
            lines.append(f"| {family} | {count} |")

        lines += ["", "## Top Factors", "",
                  "| Name | Family | Direction | Grade | Mean RIC | Val RIC | Test RIC | Recent RIC | Turnover | Max Corr | Rec |",
                  "|------|--------|-----------|-------|----------|---------|----------|------------|----------|----------|-----|"]
        for c in passing_sorted[:20]:
            lines.append(
                f"| {c.name} | {c.family} | {c.alpha_direction} | {c.grade} | "
                f"{c.mean_rank_ic:.4f} | {c.validation_rank_ic:.4f} | {c.test_rank_ic:.4f} | "
                f"{c.recent_rank_ic:.4f} | {c.turnover_estimate:.3f} | {c.max_similarity:.3f} | {c.recommendation} |"
            )

        if unsafe:
            lines += ["", "## Unsafe Rejections", "",
                      "| Name | Origin | Safety Severity | Reasons |",
                      "|------|--------|-----------------|---------|"]
            for c in unsafe[:30]:
                lines.append(f"| {c.name} | {c.origin} | {c.safety_severity} | {'; '.join(c.safety_reasons[:3])} |")

        if diversity_rejected:
            lines += ["", "## Duplicate And Diversity Rejections", "",
                      "| Name | Family | Value Corr | IC-Series Corr | Most Similar To |",
                      "|------|--------|------------|----------------|-----------------|"]
            for c in sorted(diversity_rejected, key=lambda x: max(x.max_similarity, x.ic_series_max_similarity), reverse=True)[:30]:
                lines.append(
                    f"| {c.name} | {c.family} | {c.max_similarity:.3f} | "
                    f"{c.ic_series_max_similarity:.3f} | {c.most_similar_to} |"
                )

        lines += ["", "## Generation Evolution", ""]
        for gen_i in range(self._state.generations_done):
            gen_c = [c for c in evaluated if c.generation == gen_i]
            if not gen_c:
                continue
            best_gen = max(gen_c, key=lambda c: abs(c.mean_rank_ic))
            gen_pass = [c for c in gen_c if c.passes(self.cfg)]
            lines.append(f"**Gen {gen_i}:** {len(gen_c)} evaluated, {len(gen_pass)} passed  "
                         f"| best: {best_gen.name} ric={best_gen.mean_rank_ic:.4f} grade={best_gen.grade}")

        lines += ["", "## Config Used", "", "```yaml"]
        for k, v in asdict(self.cfg).items():
            lines.append(f"{k}: {v}")
        lines += ["```", ""]

        report_path = self.run_dir / "mining_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        # Also save top factors as CSV
        if passing_sorted:
            import csv as _csv
            top_csv = self.run_dir / "top_factors.csv"
            with open(top_csv, "w", newline="", encoding="utf-8-sig") as f:
                fields = ["name", "generation", "origin", "parent", "grade", "quality_score",
                          "family", "expected_sign", "alpha_direction", "mean_rank_ic", "rank_ic_ir",
                          "rank_ic_win_rate", "ic_t_stat", "valid_days", "coverage",
                          "avg_cross_sectional_coverage", "factor_autocorr", "turnover_estimate",
                          "recent_rank_ic", "train_rank_ic", "validation_rank_ic", "test_rank_ic",
                          "max_similarity", "ic_series_max_similarity", "complexity_score",
                          "recommendation", "expression", "ic_csv_path"]
                w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(asdict(c) for c in passing_sorted[:50])
            log.info("Top factors CSV: %s", top_csv)

        return report_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="alpha_miner", description="Alpha Mining System")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # start
    p_start = sub.add_parser("start", help="Start a new mining run")
    p_start.add_argument("--config", default=str(CONFIG_PATH), help="YAML config path")
    p_start.add_argument("--generations", type=int)
    p_start.add_argument("--per-gen", type=int)
    p_start.add_argument("--model", type=str)
    p_start.add_argument("--provider", type=str, choices=["openai", "anthropic"],
                         help="LLM provider (overrides config)")
    p_start.add_argument("--outer-workers", type=int, help="Parallel factor evaluations (overrides config)")
    p_start.add_argument("--data", type=str)
    p_start.add_argument("--run-id", type=str)

    # resume
    p_resume = sub.add_parser("resume", help="Resume an interrupted run")
    p_resume.add_argument("--run-id", required=True)
    p_resume.add_argument("--config", default=str(CONFIG_PATH))

    # status
    sub.add_parser("status", help="List all mining runs")

    # report
    p_report = sub.add_parser("report", help="Generate/view report for a run")
    p_report.add_argument("--run-id", default="", help="Run ID (latest if omitted)")

    args = parser.parse_args()

    if args.cmd == "start":
        cfg = MiningConfig.from_yaml(Path(args.config))
        if args.generations:
            cfg.n_generations = args.generations
        if args.per_gen:
            cfg.factors_per_generation = args.per_gen
        if args.model:
            cfg.model = args.model
        if getattr(args, "provider", None):
            cfg.provider = args.provider
        if getattr(args, "outer_workers", None):
            cfg.outer_workers = args.outer_workers
        if args.data:
            cfg.data_pkl = args.data
        miner = AlphaMiner(cfg, run_id=args.run_id)
        miner.run(resume=False)

    elif args.cmd == "resume":
        cfg = MiningConfig.from_yaml(Path(args.config))
        miner = AlphaMiner(cfg, run_id=args.run_id)
        miner.run(resume=True)

    elif args.cmd == "status":
        if not MINING_DIR.is_dir():
            print("No mining runs found.")
            return
        runs = sorted(MINING_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"\n{'Run ID':<35}  {'Gens':>5}  {'Best Grade':>10}  {'Best RIC':>9}")
        print("-" * 70)
        for r in runs:
            cp = r / "checkpoint.json"
            if not cp.is_file():
                continue
            try:
                s = json.loads(cp.read_text(encoding="utf-8"))
                print(f"{r.name:<35}  {s.get('generations_done', 0):>5}  "
                      f"{s.get('best_grade', '?'):>10}  {s.get('best_mean_ric', 0):>9.4f}")
            except Exception:
                print(f"{r.name:<35}  (unreadable checkpoint)")

    elif args.cmd == "report":
        run_id = args.run_id
        if not run_id:
            runs = sorted(MINING_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            run_dirs = [r for r in runs if (r / "checkpoint.json").is_file()]
            if not run_dirs:
                print("No runs found.")
                return
            run_dir = run_dirs[0]
        else:
            run_dir = MINING_DIR / run_id

        cfg = MiningConfig.from_yaml(CONFIG_PATH)
        miner = AlphaMiner(cfg, run_id=run_dir.name)
        miner._state = miner._load_checkpoint()
        report = miner._write_report()
        print(f"Report written: {report}")
        print(report.read_text(encoding="utf-8")[:3000])


if __name__ == "__main__":
    main()
