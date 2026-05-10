#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static safety checks for generated factor modules."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field


@dataclass
class SafetyReport:
    is_safe: bool
    severity: str = "PASS"  # PASS | WARN | REJECT
    reasons: list[str] = field(default_factory=list)
    suspicious_patterns: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "suspicious_patterns": list(self.suspicious_patterns),
            "recommendations": list(self.recommendations),
        }


_REJECT_REGEX: list[tuple[str, str, str]] = [
    (r"\.shift\s*\(\s*-\d+", "negative shift / future data access", "Use delay(x, n) or shift(+n) only."),
    (r"\bshift\s*\(\s*periods\s*=\s*-\d+", "negative shift / future data access", "Use only non-negative historical shifts."),
    (r"\b(future|next_return|tomorrow_return|forward_return|target|label)\b", "target or future label reference", "Do not use label/forward-return columns in factor computation."),
    (r"\biloc\s*\[[^\]]*(?:\+\s*1|\+\s*\d{1,3})", "future iloc access pattern", "Use rolling/shifted vectorized history only."),
    (r"\bindex\s*\+\s*\d+", "future index access pattern", "Avoid manual future index arithmetic."),
    (r"\brolling\s*\([^)]*center\s*=\s*True", "centered rolling window uses future observations", "Use trailing rolling windows only."),
    (r"\b(rqdatac|yfinance|requests|urllib|akshare|tushare)\b", "external data/API access", "Read only the local data.pkl panel."),
    (r"\b(eval|exec)\s*\(", "dynamic code execution", "Remove eval/exec."),
    (r"\bsubprocess\b|\bos\.system\s*\(", "subprocess or shell execution", "Generated factors must not start processes."),
    (r"\bto_(csv|pickle|parquet|feather|excel)\s*\(", "file write inside factor computation", "The pipeline owns all output writes."),
    (r"\bopen\s*\([^)]*[\"'](?!data\.pkl)[^\"']+[\"']", "file access beyond data.pkl", "Only read data.pkl unless explicitly reviewed."),
    (r"\bread_(pickle|csv|parquet|feather|excel)\s*\([^)]*[\"'](?!data\.pkl)[^\"']+[\"']", "file access beyond data.pkl", "Only read data.pkl unless explicitly reviewed."),
]

_WARN_REGEX: list[tuple[str, str, str]] = [
    (r"\bexpanding\s*\(", "expanding window can leak full-sample normalization if misused", "Prefer trailing rolling windows or explicitly shift expanding statistics."),
    (r"\.mean\s*\(\s*\)|\.std\s*\(\s*\)|\.min\s*\(\s*\)|\.max\s*\(\s*\)", "possible full-sample aggregate", "Ensure aggregates are cross-sectional by date or trailing in time."),
    (r"for\s+.*\s+in\s+.*date.*:\s*[\s\S]{0,500}for\s+.*\s+in\s+.*(stock|order_book|security)", "nested date/security loops", "Prefer vectorized pandas operations."),
]

_DANGEROUS_IMPORTS = {
    "rqdatac", "yfinance", "requests", "urllib", "akshare", "tushare",
    "subprocess", "socket", "http", "ftplib",
}


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.rejects: list[str] = []
        self.warns: list[str] = []
        self.patterns: list[str] = []
        self.has_compute = False
        self.compute_returns_value = False
        self._in_compute = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _DANGEROUS_IMPORTS:
                self._reject(f"dangerous import: {alias.name}", alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in _DANGEROUS_IMPORTS:
            self._reject(f"dangerous import: {node.module}", node.module or "")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "compute_factor_df":
            self.has_compute = True
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value is not None:
                    self.compute_returns_value = True
                    if isinstance(child.value, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Constant)):
                        self._reject("compute_factor_df() returns an obvious non-DataFrame object", "non_dataframe_return")
            old = self._in_compute
            self._in_compute = True
            for stmt in node.body:
                self.visit(stmt)
            self._in_compute = old
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in {"eval", "exec", "os.system", "subprocess.call", "subprocess.run", "subprocess.Popen"}:
            self._reject(f"dangerous call: {name}", name)
        if name.endswith(".shift"):
            for arg in list(node.args) + [kw.value for kw in node.keywords if kw.arg == "periods"]:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    self._reject("negative shift / future data access", "shift(-n)")
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
                    self._reject("negative shift / future data access", f"shift({arg.value})")
        if name.endswith(".rolling"):
            for kw in node.keywords:
                if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self._reject("centered rolling window uses future observations", "rolling(center=True)")
        if name.endswith(".expanding"):
            self._warn("expanding window requires careful shifting", "expanding()")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self._in_compute and node.id in {"future", "next_return", "target", "label", "y", "tomorrow_return", "forward_return"}:
            self._reject(f"target or future label reference: {node.id}", node.id)

    def visit_For(self, node: ast.For) -> None:
        inner_for = any(isinstance(n, ast.For) for n in ast.walk(node) if n is not node)
        if inner_for:
            self._warn("nested loops may be too slow or non-vectorized", "nested for")
        self.generic_visit(node)

    def _reject(self, reason: str, pattern: str) -> None:
        self.rejects.append(reason)
        self.patterns.append(pattern)

    def _warn(self, reason: str, pattern: str) -> None:
        self.warns.append(reason)
        self.patterns.append(pattern)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def _add_regex_findings(code: str, reasons: list[str], patterns: list[str], recommendations: list[str], rules) -> None:
    for regex, reason, recommendation in rules:
        if re.search(regex, code, flags=re.IGNORECASE | re.MULTILINE):
            reasons.append(reason)
            patterns.append(regex)
            recommendations.append(recommendation)


def validate_factor_code(code: str) -> SafetyReport:
    """Return a static safety report for a generated Python factor module."""
    reject_reasons: list[str] = []
    warn_reasons: list[str] = []
    patterns: list[str] = []
    recommendations: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return SafetyReport(
            is_safe=False,
            severity="REJECT",
            reasons=[f"syntax error: line {e.lineno}: {e.msg}"],
            suspicious_patterns=["SyntaxError"],
            recommendations=["Regenerate the factor or fix the syntax before evaluation."],
        )

    visitor = _SafetyVisitor()
    visitor.visit(tree)
    reject_reasons.extend(visitor.rejects)
    warn_reasons.extend(visitor.warns)
    patterns.extend(visitor.patterns)

    code_for_patterns = _strip_strings_and_comments(code)
    _add_regex_findings(code_for_patterns, reject_reasons, patterns, recommendations, _REJECT_REGEX)
    _add_regex_findings(code_for_patterns, warn_reasons, patterns, recommendations, _WARN_REGEX)

    if not visitor.has_compute:
        reject_reasons.append("missing compute_factor_df()")
        recommendations.append("Generated modules must define compute_factor_df() -> pd.DataFrame.")
    elif not visitor.compute_returns_value:
        reject_reasons.append("compute_factor_df() has no return value")
        recommendations.append("Return a pandas DataFrame with one factor column.")

    if "pd.DataFrame" not in code and "pandas.DataFrame" not in code:
        warn_reasons.append("output type is not statically evident as pandas DataFrame")
        recommendations.append("Annotate or construct a pandas DataFrame explicitly.")

    if reject_reasons:
        severity = "REJECT"
        is_safe = False
        reasons = reject_reasons + warn_reasons
    elif warn_reasons:
        severity = "WARN"
        is_safe = True
        reasons = warn_reasons
    else:
        severity = "PASS"
        is_safe = True
        reasons = []

    if not recommendations and severity == "PASS":
        recommendations = ["No static safety issues detected."]

    return SafetyReport(
        is_safe=is_safe,
        severity=severity,
        reasons=_dedupe(reasons),
        suspicious_patterns=_dedupe(patterns),
        recommendations=_dedupe(recommendations),
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _strip_strings_and_comments(code: str) -> str:
    """Keep identifiers/operators for regex checks, but ignore comments/docstrings/string literals."""
    try:
        tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in {tokenize.STRING, tokenize.COMMENT}:
                tokens.append((tok.type, ""))
            else:
                tokens.append((tok.type, tok.string))
        return tokenize.untokenize(tokens)
    except Exception:
        return code
