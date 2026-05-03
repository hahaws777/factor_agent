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
    # mutation
    mutation_enabled: bool = True
    window_sweeps: list = field(default_factory=lambda: [5, 10, 20, 60])
    transforms: list = field(default_factory=lambda: ["winsorize", "rank", "zscore"])
    max_variants_per_factor: int = 4
    llm_refine: bool = True
    # evaluation
    data_pkl: str = "data.pkl"
    eval_workers: int = 2
    next_day_return: bool = True
    timeout_sec: int = 180
    # llm
    model: str = "gpt-4.1"
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

        g = raw.get("generation", {})
        if g.get("seed_prompts"):
            cfg.seed_prompts = g["seed_prompts"]
        if g.get("extra_factor_types"):
            cfg.extra_factor_types = g["extra_factor_types"]

        mut = raw.get("mutation", {})
        cfg.mutation_enabled = mut.get("enabled", cfg.mutation_enabled)
        cfg.window_sweeps = mut.get("window_sweeps", cfg.window_sweeps)
        cfg.transforms = mut.get("transforms", cfg.transforms)
        cfg.max_variants_per_factor = mut.get("max_variants_per_factor", cfg.max_variants_per_factor)
        cfg.llm_refine = mut.get("llm_refine", cfg.llm_refine)

        ev = raw.get("evaluation", {})
        cfg.data_pkl = ev.get("data_pkl", cfg.data_pkl)
        cfg.eval_workers = ev.get("workers", cfg.eval_workers)
        cfg.next_day_return = ev.get("next_day_return", cfg.next_day_return)
        cfg.timeout_sec = ev.get("timeout_sec", cfg.timeout_sec)

        ll = raw.get("llm", {})
        cfg.model = ll.get("model", cfg.model)
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
    origin: str = "llm"        # llm | mutation | seed
    grade: str = "?"
    quality_score: int = 0
    mean_rank_ic: float = 0.0
    rank_ic_ir: float = 0.0
    rank_ic_win_rate: float = 0.0
    valid_days: int = 0
    py_path: str = ""
    pkl_path: str = ""
    ic_csv_path: str = ""
    error: str = ""
    evaluated_at: str = ""

    def passes(self, cfg: MiningConfig) -> bool:
        grade_ok = GRADE_ORDER.get(self.grade, 0) >= GRADE_ORDER.get(cfg.min_grade, 0)
        ic_ok = abs(self.mean_rank_ic) >= cfg.min_rank_ic
        return grade_ok and ic_ok and not self.error

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
    from factor_code_agent import call_openai
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            return call_openai(user_message, model)
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

_SEED_PROMPTS = [
    "20-day price momentum: past 20-day cumulative return, cross-sectionally ranked",
    "short-term reversal: negative 5-day return, winsorized 1-99%",
    "volume-price divergence: 10-day close change divided by 10-day average volume change",
    "20-day return volatility: rolling std of daily returns, ranked cross-sectionally",
]

_FACTOR_TYPES = ["momentum", "reversal", "volatility", "volume", "value_proxy", "earnings_quality"]


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
        "Load ALL inputs from local data.pkl only (Path(__file__).resolve().parents[1] / 'data.pkl'). "
        "Do not use rqdatac or any external data API."
    )
    return base


# ── Factor evaluation ─────────────────────────────────────────────────────────

def _evaluate_candidate(
    candidate: FactorCandidate,
    run_dir: Path,
    cfg: MiningConfig,
) -> FactorCandidate:
    """Run one factor through the full pipeline: pickle → Rank IC → screen."""
    from factor_screener import screen_ic_csv

    py_path = run_dir / "factors" / f"{candidate.name}.py"
    pkl_path = run_dir / "factors" / f"{candidate.name}.pkl"
    ic_csv = run_dir / "ic" / f"{candidate.name}_rankic.csv"
    py_path.parent.mkdir(parents=True, exist_ok=True)
    ic_csv.parent.mkdir(parents=True, exist_ok=True)

    py_path.write_text(candidate.code, encoding="utf-8")
    candidate.py_path = str(py_path)
    candidate.pkl_path = str(pkl_path)
    candidate.ic_csv_path = str(ic_csv)

    # Step 1: compute factor and save pkl
    pipe_script = AGENT_DIR / "factor_agent_pipeline.py"
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
    else:
        candidate.error = "no IC CSV produced"

    candidate.evaluated_at = datetime.utcnow().isoformat()
    log.info("Evaluated %-40s grade=%-2s ric=%.4f  ir=%.3f  %.0fs",
             candidate.name, candidate.grade, candidate.mean_rank_ic,
             candidate.rank_ic_ir, elapsed)
    return candidate


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

        candidates = []
        factor_types = self.cfg.extra_factor_types or _FACTOR_TYPES

        # Use seed prompts for gen-0; diversified prompts for later generations
        if generation == 0:
            base_prompts = list(self.cfg.seed_prompts) or _SEED_PROMPTS.copy()
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
            user_msg = _build_generation_prompt(desc, survivors, ft_hint)

            log.info("Generating factor %d/%d (gen %d)", i + 1, n, generation)
            try:
                raw = _call_llm_with_retry(user_msg, self.cfg.model, self.cfg)
            except Exception as e:
                log.error("LLM generation failed: %s", e)
                continue

            code = extract_python_code(raw)
            err = validate_python_syntax(code)
            if err:
                log.warning("Syntax error in generated code: %s", err)
                continue

            h = _code_hash(code)
            if self.cfg.dedup_on_code_hash and h in self._state.seen_hashes:
                log.info("Duplicate code hash %s — skipping", h)
                continue

            name = f"gen{generation}_llm_{i:02d}_{h[:6]}"
            c = FactorCandidate(
                name=name, code=code, code_hash=h,
                generation=generation, origin="llm",
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

        from factor_mutator import FactorMutator

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
        """Evaluate candidates sequentially (subprocess already parallel inside)."""
        results = []
        for i, c in enumerate(candidates):
            log.info("Evaluating %d/%d: %s", i + 1, len(candidates), c.name)
            result = _evaluate_candidate(c, self.run_dir, self.cfg)
            results.append(result)
            _append_ledger(result, self.ledger_path)
            # Check wall-clock limit
            if (time.time() - self.start_wall) / 3600 > self.cfg.max_run_hours:
                log.warning("Max run hours (%.1f) reached — stopping evaluation early", self.cfg.max_run_hours)
                break
        return results

    def _select_survivors(
        self,
        candidates: list[FactorCandidate],
        prev_survivors: list[FactorCandidate],
    ) -> list[FactorCandidate]:
        """Pick top-k from (new candidates ∪ prev survivors)."""
        pool = candidates + prev_survivors
        passing = [c for c in pool if c.passes(self.cfg)]

        if not passing:
            log.warning("No factors passed quality gates this generation — keeping best available")
            passing = sorted(pool, key=lambda c: abs(c.mean_rank_ic), reverse=True)[:self.cfg.top_k_survivors]
        else:
            passing.sort(key=lambda c: (GRADE_ORDER.get(c.grade, 0), abs(c.mean_rank_ic)), reverse=True)

        # Diversity penalty: if two factors are near-identical, prefer the one with higher IC
        if self.cfg.diversity_penalty:
            unique_survivors: list[FactorCandidate] = []
            seen_ric: list[float] = []
            for c in passing:
                if all(abs(c.mean_rank_ic - r) > 0.005 for r in seen_ric):
                    unique_survivors.append(c)
                    seen_ric.append(c.mean_rank_ic)
                if len(unique_survivors) >= self.cfg.top_k_survivors:
                    break
            passing = unique_survivors or passing

        return passing[: self.cfg.top_k_survivors]

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
            summary = (
                f"Gen {gen} complete — {len(evaluated)} evaluated, {len(passing)} passed\n"
                f"Survivors: " + ", ".join(f"{s.name}(grade={s.grade}, ric={s.mean_rank_ic:.4f})" for s in survivors)
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
        passing = [c for c in evaluated if c.passes(self.cfg)]
        passing_sorted = sorted(passing, key=lambda c: abs(c.mean_rank_ic), reverse=True)

        # Grade distribution
        from collections import Counter
        grade_dist = Counter(c.grade for c in evaluated)

        lines = [
            f"# Alpha Mining Report — {self.run_id}",
            f"",
            f"**Started:** {self._state.started_at}  ",
            f"**Generations completed:** {self._state.generations_done}/{self.cfg.n_generations}  ",
            f"**Total evaluated:** {len(evaluated)}  |  **Passed:** {len(passing)}  ",
            f"**Best grade:** {self._state.best_grade}  |  **Best Rank IC:** {self._state.best_mean_ric:.4f}",
            f"",
            f"## Grade Distribution",
            f"",
            "| Grade | Count |",
            "|-------|-------|",
        ]
        for g in ["A", "B", "C", "D", "F"]:
            lines.append(f"| {g} | {grade_dist.get(g, 0)} |")

        lines += ["", "## Top Factors", "",
                  "| Name | Gen | Origin | Grade | Mean Rank IC | IC IR | Win Rate |",
                  "|------|-----|--------|-------|-------------|-------|----------|"]
        for c in passing_sorted[:20]:
            lines.append(
                f"| {c.name} | {c.generation} | {c.origin} | {c.grade} | "
                f"{c.mean_rank_ic:.4f} | {c.rank_ic_ir:.3f} | {c.rank_ic_win_rate:.2%} |"
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
                          "mean_rank_ic", "rank_ic_ir", "rank_ic_win_rate", "valid_days", "ic_csv_path"]
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
