#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared alpha DSL recipes.

LLM generation should choose from this library instead of inventing common
time-series rank/return/volatility formulas from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from factor_dsl import FactorSpec
except ImportError:  # package import path used by unit tests
    from .factor_dsl import FactorSpec


@dataclass(frozen=True)
class FactorRecipe:
    recipe_id: str
    name: str
    family: str
    expression: str
    expected_sign: str = "unknown"
    economic_hypothesis: str = ""
    required_fields: list[str] = field(default_factory=list)
    lookback_windows: list[int] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)

    def to_spec(
        self,
        *,
        name_suffix: str = "",
        why_not_duplicate: str = "",
        economic_hypothesis: str = "",
    ) -> FactorSpec:
        name = self.name if not name_suffix else f"{self.name}_{name_suffix}"
        return FactorSpec(
            name=name,
            family=self.family,
            economic_hypothesis=economic_hypothesis or self.economic_hypothesis,
            expression=self.expression,
            expected_sign=self.expected_sign,
            required_fields=list(self.required_fields),
            lookback_windows=list(self.lookback_windows),
            risk_notes=list(self.risk_notes),
            why_not_duplicate=why_not_duplicate,
        )


PREPARED_FACTOR_RECIPES: list[FactorRecipe] = [
    FactorRecipe(
        recipe_id="momentum_20d_rank",
        name="momentum_20d_rank",
        family="momentum",
        expression="rank(ts_return(close, 20))",
        expected_sign="positive",
        economic_hypothesis="Medium-term winners may continue due to trend-following and delayed information diffusion.",
        required_fields=["close"],
        lookback_windows=[20],
    ),
    FactorRecipe(
        recipe_id="momentum_60d_rank",
        name="momentum_60d_rank",
        family="momentum",
        expression="rank(ts_return(close, 60))",
        expected_sign="positive",
        economic_hypothesis="Quarterly price momentum captures persistent relative strength.",
        required_fields=["close"],
        lookback_windows=[60],
    ),
    FactorRecipe(
        recipe_id="ts_rank_return_20d",
        name="ts_rank_return_20d",
        family="momentum",
        expression="rank(ts_rank(ts_return(close, 5), 20))",
        expected_sign="positive",
        economic_hypothesis="Recent returns that rank high versus their own short history may indicate strengthening trend.",
        required_fields=["close"],
        lookback_windows=[5, 20],
    ),
    FactorRecipe(
        recipe_id="short_reversal_5d_winsor",
        name="short_reversal_5d_winsor",
        family="reversal",
        expression="winsorize(-ts_return(close, 5), lower=0.01, upper=0.99)",
        expected_sign="positive",
        economic_hypothesis="Short-term losers may rebound after temporary liquidity pressure or overreaction.",
        required_fields=["close"],
        lookback_windows=[5],
        risk_notes=["Can have high turnover and may be sensitive to transaction costs."],
    ),
    FactorRecipe(
        recipe_id="short_reversal_10d_rank",
        name="short_reversal_10d_rank",
        family="reversal",
        expression="rank(-ts_return(close, 10))",
        expected_sign="positive",
        economic_hypothesis="Two-week reversal captures mean reversion after short-term overreaction.",
        required_fields=["close"],
        lookback_windows=[10],
        risk_notes=["Can load on liquidity and small-cap effects."],
    ),
    FactorRecipe(
        recipe_id="volatility_20d_lowvol_rank",
        name="volatility_20d_lowvol_rank",
        family="volatility",
        expression="rank(-ts_std(ts_return(close, 1), 20))",
        expected_sign="positive",
        economic_hypothesis="Lower recent realized volatility may proxy for quality or lower lottery demand.",
        required_fields=["close"],
        lookback_windows=[1, 20],
    ),
    FactorRecipe(
        recipe_id="volatility_zscore_60d",
        name="volatility_zscore_60d",
        family="volatility",
        expression="rank(-ts_zscore(ts_std(ts_return(close, 1), 20), 60))",
        expected_sign="positive",
        economic_hypothesis="Stocks with unusually low volatility versus their own history may have calmer risk profile.",
        required_fields=["close"],
        lookback_windows=[1, 20, 60],
    ),
    FactorRecipe(
        recipe_id="volume_rank_20d",
        name="volume_rank_20d",
        family="volume",
        expression="rank(ts_rank(volume, 20))",
        expected_sign="unknown",
        economic_hypothesis="Unusual recent trading activity can indicate attention, liquidity pressure, or informed trading.",
        required_fields=["volume"],
        lookback_windows=[20],
    ),
    FactorRecipe(
        recipe_id="amount_liquidity_20d",
        name="amount_liquidity_20d",
        family="liquidity",
        expression="rank(ts_mean(amount, 20))",
        expected_sign="unknown",
        economic_hypothesis="Higher traded amount captures liquidity and institutional investability.",
        required_fields=["amount"],
        lookback_windows=[20],
    ),
    FactorRecipe(
        recipe_id="volume_price_divergence_10d",
        name="volume_price_divergence_10d",
        family="volume",
        expression="rank(delta(close, 10) / (ts_mean(volume, 10) + 1))",
        expected_sign="unknown",
        economic_hypothesis="Price movement per unit of volume can capture fragile price moves or supply-demand imbalance.",
        required_fields=["close", "volume"],
        lookback_windows=[10],
    ),
    FactorRecipe(
        recipe_id="small_cap_inverse_mcap",
        name="small_cap_inverse_mcap",
        family="value_proxy",
        expression="rank(1 / market_cap)",
        expected_sign="positive",
        economic_hypothesis="Smaller capitalization stocks may earn a size premium, subject to liquidity and trading constraints.",
        required_fields=["market_cap"],
        lookback_windows=[],
        risk_notes=["Hidden liquidity exposure; capacity may be limited."],
    ),
    FactorRecipe(
        recipe_id="vwap_close_reversal_5d",
        name="vwap_close_reversal_5d",
        family="reversal",
        expression="rank((vwap - close) / close)",
        expected_sign="positive",
        economic_hypothesis="Close below VWAP can indicate intraday selling pressure that may partially revert.",
        required_fields=["vwap", "close"],
        lookback_windows=[],
    ),
    FactorRecipe(
        recipe_id="industry_neutral_momentum_20d",
        name="industry_neutral_momentum_20d",
        family="momentum",
        expression="neutralize_industry(rank(ts_return(close, 20)))",
        expected_sign="positive",
        economic_hypothesis="Industry-neutral momentum seeks stock-specific trend while reducing sector exposure.",
        required_fields=["close", "industry"],
        lookback_windows=[20],
    ),
    FactorRecipe(
        recipe_id="size_neutral_reversal_5d",
        name="size_neutral_reversal_5d",
        family="reversal",
        expression="neutralize_size(winsorize(-ts_return(close, 5), lower=0.01, upper=0.99))",
        expected_sign="positive",
        economic_hypothesis="Size-neutral short-term reversal reduces hidden small-cap exposure.",
        required_fields=["close", "market_cap"],
        lookback_windows=[5],
    ),
]


def recipe_by_id(recipe_id: str) -> FactorRecipe | None:
    recipe_id = recipe_id.strip()
    return next((r for r in PREPARED_FACTOR_RECIPES if r.recipe_id == recipe_id), None)


def recipes_for_family(family: str, *, include_composite: bool = True) -> list[FactorRecipe]:
    family = (family or "").strip()
    recipes = [r for r in PREPARED_FACTOR_RECIPES if not family or r.family == family]
    if include_composite and family and len(recipes) < 4:
        recipes.extend(r for r in PREPARED_FACTOR_RECIPES if r.family != family)
    return recipes


def select_recipe(
    family_hint: str,
    index: int,
    used_expressions: set[str] | None = None,
) -> FactorRecipe:
    used_expressions = used_expressions or set()
    candidates = recipes_for_family(family_hint)
    fresh = [r for r in candidates if r.expression.replace(" ", "") not in used_expressions]
    candidates = fresh or candidates or PREPARED_FACTOR_RECIPES
    return candidates[index % len(candidates)]


def recipes_for_prompt(family_hint: str, limit: int = 10) -> str:
    rows = []
    for r in recipes_for_family(family_hint)[:limit]:
        rows.append(
            f"- recipe_id={r.recipe_id}; family={r.family}; name={r.name}; "
            f"expression={r.expression}; expected_sign={r.expected_sign}; "
            f"hypothesis={r.economic_hypothesis}"
        )
    return "\n".join(rows)
