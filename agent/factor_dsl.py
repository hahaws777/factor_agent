#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constrained factor expression DSL and compiler."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any


DEFAULT_ALLOWED_FIELDS = {
    "open", "high", "low", "close", "volume", "amount", "vwap", "market_cap", "industry",
}

ALLOWED_OPERATORS = {
    "delay", "delta", "ts_return", "ret", "ts_mean", "ts_std", "ts_rank", "ts_zscore", "ts_corr",
    "rank", "zscore", "winsorize", "signed_power", "log1p_abs",
    "neutralize_industry", "neutralize_size",
}


@dataclass
class FactorSpec:
    name: str
    family: str
    economic_hypothesis: str
    expression: str
    expected_sign: str = "unknown"
    required_fields: list[str] = field(default_factory=list)
    lookback_windows: list[int] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    why_not_duplicate: str = ""

    @classmethod
    def from_json_text(cls, text: str) -> "FactorSpec":
        raw = _extract_json(text)
        data = json.loads(raw)
        return cls(
            name=_safe_name(str(data.get("name") or "dsl_factor")),
            family=str(data.get("family") or "composite"),
            economic_hypothesis=str(data.get("economic_hypothesis") or ""),
            expression=str(data.get("expression") or ""),
            expected_sign=str(data.get("expected_sign") or "unknown"),
            required_fields=[str(x) for x in data.get("required_fields", [])],
            lookback_windows=[int(x) for x in data.get("lookback_windows", []) if _is_int_like(x)],
            risk_notes=[str(x) for x in data.get("risk_notes", [])],
            why_not_duplicate=str(data.get("why_not_duplicate") or ""),
        )


@dataclass
class DSLValidationResult:
    is_valid: bool
    expression: str
    canonical_expression: str = ""
    required_fields: list[str] = field(default_factory=list)
    lookback_windows: list[int] = field(default_factory=list)
    complexity_score: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DSLConfig:
    allowed_fields: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_FIELDS))
    max_window: int = 252
    min_window: int = 1
    max_depth: int = 8
    max_nodes: int = 80


def validate_expression(expression: str, cfg: DSLConfig | None = None) -> DSLValidationResult:
    cfg = cfg or DSLConfig()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return DSLValidationResult(False, expression, errors=[f"syntax error: {e.msg}"])

    validator = _DSLValidator(cfg)
    validator.visit(tree)
    if validator.max_depth > cfg.max_depth:
        validator.errors.append(f"expression depth {validator.max_depth} exceeds max_depth={cfg.max_depth}")
    if validator.nodes > cfg.max_nodes:
        validator.errors.append(f"expression node count {validator.nodes} exceeds max_nodes={cfg.max_nodes}")

    canonical = _canonical(tree.body)
    return DSLValidationResult(
        is_valid=not validator.errors,
        expression=expression,
        canonical_expression=canonical,
        required_fields=sorted(validator.fields),
        lookback_windows=sorted(validator.windows),
        complexity_score=validator.nodes + 2 * len(validator.windows),
        errors=validator.errors,
        warnings=validator.warnings,
    )


def compile_expression_to_module(
    spec: FactorSpec,
    factor_name: str | None = None,
    cfg: DSLConfig | None = None,
) -> str:
    cfg = cfg or DSLConfig()
    validation = validate_expression(spec.expression, cfg)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.errors))

    name = _safe_name(factor_name or spec.name)
    expr = validation.canonical_expression
    allowed_fields = sorted(cfg.allowed_fields)
    hypothesis = spec.economic_hypothesis.replace('"""', "'''")

    return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generated DSL factor: {name}
Family: {spec.family}
Expected sign: {spec.expected_sign}
Hypothesis: {hypothesis}
Expression: {expr}
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd

ALLOWED_FIELDS = {allowed_fields!r}
FACTOR_EXPRESSION = {expr!r}


def _load_panel() -> pd.DataFrame:
    root = Path(os.environ.get("FACTOR_DATA_ROOT", str(Path(__file__).resolve().parent)))
    if not (root / "data.pkl").is_file():
        root = Path(__file__).resolve().parent
        while not (root / "data.pkl").is_file() and root != root.parent:
            root = root.parent
    df = pd.read_pickle(root / "data.pkl")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["order_book_id"] = df["order_book_id"].astype(str)
    return df.sort_values(["date", "order_book_id"])


def _wide(df: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in df.columns:
        raise KeyError(f"Required field {{field!r}} missing from data.pkl")
    return df.pivot(index="date", columns="order_book_id", values=field).sort_index()


def delay(x, n):
    return x.shift(int(n))


def delta(x, n):
    n = int(n)
    return x - x.shift(n)


def ts_return(x, n):
    n = int(n)
    return x / x.shift(n) - 1.0


def ret(x, n):
    return ts_return(x, n)


def ts_mean(x, n):
    n = int(n)
    return x.rolling(n, min_periods=max(2, n // 2)).mean()


def ts_std(x, n):
    n = int(n)
    return x.rolling(n, min_periods=max(2, n // 2)).std()


def ts_rank(x, n):
    n = int(n)
    return x.rolling(n, min_periods=max(2, n // 2)).apply(
        lambda a: pd.Series(a).rank(pct=True).iloc[-1], raw=False
    )


def ts_zscore(x, n):
    mean = ts_mean(x, n)
    std = ts_std(x, n).replace(0, np.nan)
    return (x - mean) / std


def ts_corr(x, y, n):
    n = int(n)
    return x.rolling(n, min_periods=max(2, n // 2)).corr(y)


def rank(x):
    return x.rank(axis=1, pct=True)


def zscore(x):
    std = x.std(axis=1).replace(0, np.nan)
    return x.sub(x.mean(axis=1), axis=0).div(std, axis=0)


def winsorize(x, lower=0.01, upper=0.99):
    lo = x.quantile(float(lower), axis=1)
    hi = x.quantile(float(upper), axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def signed_power(x, p):
    return np.sign(x) * (np.abs(x) ** float(p))


def log1p_abs(x):
    return np.sign(x) * np.log1p(np.abs(x))


def neutralize_industry(x):
    if "industry" not in _FIELDS:
        return x
    stacked = x.stack(dropna=False).rename("x").to_frame()
    ind = _FIELDS["industry"].stack(dropna=False).rename("industry")
    joined = stacked.join(ind).dropna(subset=["industry"])
    demeaned = joined["x"] - joined.groupby([joined.index.get_level_values(0), "industry"])["x"].transform("mean")
    out = x.copy()
    out.loc[:, :] = np.nan
    out.update(demeaned.unstack())
    return out


def neutralize_size(x):
    if "market_cap" not in _FIELDS:
        return x
    size = np.log1p(pd.to_numeric(_FIELDS["market_cap"].stack(dropna=False), errors="coerce")).rename("size")
    vals = x.stack(dropna=False).rename("x")
    joined = pd.concat([vals, size], axis=1).dropna()

    def _resid(g):
        sx = g["size"]
        sy = g["x"]
        var = sx.var()
        if var is None or var <= 1e-12:
            return sy - sy.mean()
        beta = sy.cov(sx) / var
        return sy - (beta * sx + (sy.mean() - beta * sx.mean()))

    resid = joined.groupby(level=0, group_keys=False).apply(_resid)
    out = x.copy()
    out.loc[:, :] = np.nan
    out.update(resid.unstack())
    return out


def compute_factor_df() -> pd.DataFrame:
    global _FIELDS
    df = _load_panel()
    _FIELDS = {{field: _wide(df, field) for field in ALLOWED_FIELDS if field in df.columns}}
    { _field_assignments(validation.required_fields) }
    factor = {expr}
    factor = factor.astype(float)
    out = factor.stack(dropna=False).rename("{name}").to_frame()
    out.index.names = ["date", "order_book_id"]
    out = out.reorder_levels(["order_book_id", "date"]).sort_index()
    return out
'''


class _DSLValidator(ast.NodeVisitor):
    def __init__(self, cfg: DSLConfig) -> None:
        self.cfg = cfg
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.fields: set[str] = set()
        self.windows: set[int] = set()
        self.nodes = 0
        self.max_depth = 0
        self._depth = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.nodes += 1
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        super().generic_visit(node)
        self._depth -= 1

    def visit_Name(self, node: ast.Name) -> None:
        self.nodes += 1
        if node.id not in self.cfg.allowed_fields:
            self.errors.append(f"unknown field or bare identifier: {node.id}")
        else:
            self.fields.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        self.nodes += 1
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        if not isinstance(node.func, ast.Name):
            self.errors.append("only direct function calls are allowed")
        else:
            fn = node.func.id
            if fn not in ALLOWED_OPERATORS:
                self.errors.append(f"operator not allowed: {fn}")
            self._validate_call(fn, node)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg not in {"lower", "upper"}:
                self.errors.append(f"keyword not allowed: {kw.arg}")
            self.visit(kw.value)
        self._depth -= 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.errors.append("attribute access is not allowed")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.errors.append("subscript access is not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.errors.append("lambda is not allowed")

    def visit_Compare(self, node: ast.Compare) -> None:
        self.errors.append("comparisons are not allowed in factor DSL")

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.errors.append("boolean operations are not allowed in factor DSL")

    def visit_Constant(self, node: ast.Constant) -> None:
        self.nodes += 1
        if not isinstance(node.value, (int, float)):
            self.errors.append(f"constant type not allowed: {type(node.value).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.nodes += 1
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self.errors.append("only +, -, *, / binary operators are allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.nodes += 1
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            self.errors.append("only unary +/- are allowed")
        self.visit(node.operand)

    def _validate_call(self, fn: str, node: ast.Call) -> None:
        window_fns = {"delay", "delta", "ts_return", "ret", "ts_mean", "ts_std", "ts_rank", "ts_zscore"}
        if fn in window_fns:
            self._window_arg(node, 1, fn)
        elif fn == "ts_corr":
            self._window_arg(node, 2, fn)

    def _window_arg(self, node: ast.Call, pos: int, fn: str) -> None:
        if len(node.args) <= pos:
            self.errors.append(f"{fn} requires a window argument")
            return
        raw = node.args[pos]
        if not isinstance(raw, ast.Constant) or not isinstance(raw.value, int):
            self.errors.append(f"{fn} window must be an integer literal")
            return
        window = int(raw.value)
        if window < self.cfg.min_window or window > self.cfg.max_window:
            self.errors.append(f"{fn} window {window} outside [{self.cfg.min_window}, {self.cfg.max_window}]")
        self.windows.add(window)


def _canonical(node: ast.AST) -> str:
    return ast.unparse(node)


def _field_assignments(fields: list[str]) -> str:
    if not fields:
        return "pass"
    return "\n    ".join(f"{field} = _FIELDS[{field!r}]" for field in fields)


def _extract_json(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return text[start:end + 1]


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip().lower()).strip("_")
    if not value:
        value = "dsl_factor"
    if value[0].isdigit():
        value = f"factor_{value}"
    return value[:80]


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False
