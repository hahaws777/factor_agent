#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Mutation Engine.

Takes a working factor's Python code and generates variants by:
  1. window_sweep   — replace detected integer windows with alternatives
  2. add_transform  — prepend a cross-sectional transform (rank/winsorize/zscore)
  3. llm_refine     — ask the LLM to intelligently improve the factor
  4. generate_all   — combine all methods up to max_variants cap
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

# ── Transform code snippets injected at the end of compute_factor_df ─────────

_TRANSFORM_SNIPPETS: dict[str, str] = {
    "winsorize": '''
    # Winsorize cross-sectionally at 1% / 99%
    def _winsorize(s, lo=0.01, hi=0.99):
        low, high = s.quantile(lo), s.quantile(hi)
        return s.clip(lower=low, upper=high)
    factor_col = fac.columns[0]
    fac[factor_col] = fac.reset_index().groupby("date")[factor_col].transform(_winsorize).values
''',
    "rank": '''
    # Cross-sectional rank normalization (0..1)
    factor_col = fac.columns[0]
    fac[factor_col] = fac.reset_index().groupby("date")[factor_col].rank(pct=True).values
''',
    "zscore": '''
    # Cross-sectional z-score
    def _zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 1e-9 else s * 0
    factor_col = fac.columns[0]
    fac[factor_col] = fac.reset_index().groupby("date")[factor_col].transform(_zscore).values
''',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_windows(code: str) -> list[int]:
    """Return unique integers that look like rolling window sizes (5–252)."""
    candidates = [int(m) for m in re.findall(r"\b(\d+)\b", code)]
    return sorted({c for c in candidates if 3 <= c <= 252})


def _replace_window(code: str, old: int, new: int) -> str:
    return re.sub(rf"\b{old}\b", str(new), code)


def _inject_transform(code: str, transform: str, factor_name_suffix: str) -> str | None:
    """Inject a transform snippet before the final return statement."""
    snippet = _TRANSFORM_SNIPPETS.get(transform)
    if not snippet:
        return None
    # Find last assignment to `fac` before the return
    pattern = r"(return\s+fac(?:\[[^\]]+\])?(?:\[\[?[^\]]+\]?\])?.*)"
    m = re.search(pattern, code)
    if not m:
        # Fallback: append before last line that starts with 'return'
        lines = code.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("return"):
                lines.insert(i, snippet)
                break
        else:
            return None
        return "\n".join(lines)
    insert_pos = m.start()
    return code[:insert_pos] + snippet + "\n    " + code[insert_pos:]


def _rename_factor(code: str, old_name: str, new_name: str) -> str:
    """Rename the factor column in the code."""
    return code.replace(f'"{old_name}"', f'"{new_name}"').replace(f"'{old_name}'", f"'{new_name}'")


# ── Main mutation class ───────────────────────────────────────────────────────

class FactorMutator:
    """Generate systematic variants of a factor's Python code."""

    def __init__(
        self,
        window_sweeps: list[int] | None = None,
        transforms: list[str] | None = None,
        max_variants: int = 4,
        llm_model: str = "gpt-4.1",
        llm_temperature: float = 0.3,
    ):
        self.window_sweeps = window_sweeps or [5, 10, 20, 60]
        self.transforms = transforms or ["winsorize", "rank", "zscore"]
        self.max_variants = max_variants
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    def window_sweep(self, code: str, base_name: str) -> list[tuple[str, str]]:
        """Replace detected window sizes with alternatives. Returns (new_name, new_code) pairs."""
        detected = _detect_windows(code)
        if not detected:
            log.debug("No window integers detected in %s", base_name)
            return []
        primary = detected[0]   # assume first detected window is the main one
        variants = []
        for w in self.window_sweeps:
            if w == primary:
                continue
            new_code = _replace_window(code, primary, w)
            new_name = re.sub(r"\d+d?$", f"{w}d", base_name) or f"{base_name}_{w}d"
            new_code = _rename_factor(new_code, base_name, new_name)
            variants.append((new_name, new_code))
        log.info("Window sweep on %s: primary=%d → %d variants", base_name, primary, len(variants))
        return variants

    def add_transform(self, code: str, base_name: str) -> list[tuple[str, str]]:
        """Inject cross-sectional transforms. Returns (new_name, new_code) pairs."""
        variants = []
        for t in self.transforms:
            mutated = _inject_transform(code, t, base_name)
            if mutated is None:
                log.debug("Could not inject %s into %s", t, base_name)
                continue
            new_name = f"{base_name}_{t}"
            mutated = _rename_factor(mutated, base_name, new_name)
            variants.append((new_name, mutated))
        log.info("Transform mutations on %s: %d variants", base_name, len(variants))
        return variants

    def llm_refine(
        self,
        code: str,
        base_name: str,
        grade: str,
        metrics: dict,
        call_llm_fn,   # callable(user_message, model) -> str
    ) -> list[tuple[str, str]]:
        """
        Ask the LLM to produce an improved variant given the factor's grade and metrics.
        Returns up to 2 (name, code) pairs.
        """
        mean_ric = metrics.get("mean_rank_ic", 0) or 0
        ir = metrics.get("rank_ic_ir", 0) or 0
        win = metrics.get("rank_ic_win_rate", 0) or 0

        prompt = (
            f"The following Python factor (for China A-shares) achieved:\n"
            f"  Grade: {grade}  Mean Rank IC: {mean_ric:.4f}  IR: {ir:.3f}  Win rate: {win:.2%}\n\n"
            f"```python\n{code}\n```\n\n"
            "Generate TWO improved variants. For each:\n"
            "1. Change the factor formula meaningfully (not just rename).\n"
            "2. The function must still be named compute_factor_df() and return a MultiIndex DataFrame.\n"
            "3. Separate the two variants with exactly the line: # ---VARIANT 2---\n"
            "Output only Python code, no explanations."
        )
        try:
            raw = call_llm_fn(prompt, self.llm_model)
        except Exception as e:
            log.warning("LLM refinement failed for %s: %s", base_name, e)
            return []

        from factor_code_agent import extract_python_code, validate_python_syntax
        parts = re.split(r"#\s*-+VARIANT 2-+", raw)
        results = []
        for i, part in enumerate(parts[:2], start=1):
            code_v = extract_python_code(part.strip())
            err = validate_python_syntax(code_v)
            if err:
                log.warning("LLM variant %d for %s has syntax error: %s", i, base_name, err)
                continue
            new_name = f"{base_name}_llm_v{i}"
            results.append((new_name, code_v))
        log.info("LLM refinement of %s: %d valid variants", base_name, len(results))
        return results

    def generate_all(
        self,
        code: str,
        base_name: str,
        grade: str = "C",
        metrics: dict | None = None,
        call_llm_fn=None,
    ) -> list[tuple[str, str]]:
        """
        Generate all variants up to self.max_variants.
        Priority: window_sweep → add_transform → llm_refine.
        """
        all_variants: list[tuple[str, str]] = []

        all_variants.extend(self.window_sweep(code, base_name))
        if len(all_variants) < self.max_variants:
            all_variants.extend(self.add_transform(code, base_name))
        if len(all_variants) < self.max_variants and call_llm_fn and grade in ("A", "B", "C"):
            all_variants.extend(
                self.llm_refine(code, base_name, grade, metrics or {}, call_llm_fn)
            )

        # Deduplicate by code content
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for name, c in all_variants:
            h = hash(c.strip())
            if h not in seen:
                seen.add(h)
                unique.append((name, c))

        result = unique[: self.max_variants]
        log.info("Generated %d/%d unique variants for %s", len(result), len(all_variants), base_name)
        return result
