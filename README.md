# Factor Research Workspace

Evolutionary alpha mining system for China A-shares: LLM-generated factors, Rank IC evaluation, factor screening, and a Streamlit chat UI.

- `agent/` — LLM factor generation, evolutionary miner, pipeline runner, Streamlit UI
- `scripts/` — analysis, batch tools, plotting, data utilities, download helpers

---

## Quick Start

### 1) Environment

```bash
python3.12 -m venv venv312
source venv312/bin/activate        # Windows: venv312\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.10+**. The `requirements.txt` covers all dependencies including `akshare`, `openai`, `anthropic`, `pandas`, `scipy`, `streamlit`, and `pyyaml`.

### 2) API Keys

Create `.env` in the project root. At least one LLM provider key is required:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Optional Telegram notifications:

```
TELEGRAM_BOT_TOKEN=<token>
TELEGRAM_CHAT_ID=<chat_id>
```

### 3) Required Local Data

Place market panel data at project root:

```
data.pkl        — individual A-share daily bars (order_book_id, date, close, ST, limit_up_flag, ...)
index_data.pkl  — benchmark index daily bars (download via script below)
```

Download index data (CSI 300, CSI 500, CSI 1000, SSE 50, Shanghai Composite, etc.):

```bash
python scripts/download/download_index_data.py
# optionally specify date range:
python scripts/download/download_index_data.py --start 20100101 --end 20251231
```

This fetches 7 major indices via `akshare` and saves them with RiceQuant-format codes (`000300.XSHG`, `000905.XSHG`, etc.) to `index_data.pkl`.

### 4) Run Preset Pipeline (No LLM Required)

```bash
python agent/factor_agent_pipeline.py --preset volatility_20d --artifact-dir agent_runs/smoke --data data.pkl
```

### 5) Generate Factor From Natural Language

```bash
# OpenAI (default)
python agent/factor_code_agent.py -d "Your factor description"

# Anthropic / Claude
python agent/factor_code_agent.py -d "Your factor description" --provider anthropic --model claude-sonnet-4-6

# Full pipeline (generate → pickle → Rank IC → screen)
python agent/factor_agent_pipeline.py -d "20-day momentum" --data data.pkl --artifact-dir agent_runs/my_run
python agent/factor_agent_pipeline.py -d "..." --provider anthropic --model claude-sonnet-4-6 --artifact-dir agent_runs/my_run
```

### 6) Run Chat UI

```bash
streamlit run agent/ui/streamlit_app.py
```

The UI supports:
- **Provider selection** — switch between OpenAI and Anthropic models in the sidebar
- **Multi-level parallel controls** — Level-2 per-day IC workers for single runs; Level-1 factor workers + Level-2 IC workers for batch
- **Backend/device selection** — `pandas` or `torch` with `cuda/cpu/auto`
- **Alpha Miner viewer** — browse mining run status and top factors directly from the UI

---

## Analysis Scripts

Single-factor Rank IC analysis:

```bash
python scripts/analysis/factor_rankic_analysis.py --factor path/to/factor.pkl
```

Rank IC defaults to next-day forward return. Add `--same-day` only for diagnostic same-day analysis.

Batch analysis:

```bash
python scripts/analysis/batch_factor_analysis.py factors_by_type/alpha101
python scripts/analysis/batch_factor_analysis.py factors_by_type/alpha101 \
    --output-dir rankic_batch_results --factor-workers 4 --ic-workers 8 --backend pandas
```

> **IC filter correctness**: `limit_up_flag` / `limit_down_flag` are `object` columns containing `True`, `False`, and `NaN`. The analyzer uses `!= True` comparisons (not `.astype(bool)`) to avoid treating `NaN` as limit-up. When using next-day returns, stocks that hit the limit on the **return day T+1** are also excluded, since that return is capped and not a clean alpha signal.

---

## Run Metadata

Core workflows write `run_metadata.json` automatically:

- `agent/factor_agent_pipeline.py` → `<artifact_dir>/run_metadata.json`
- `scripts/analysis/batch_factor_analysis.py` → `<output_dir>/run_metadata.json`
- `scripts/analysis/factor_compare_dashboard.py` → `<out_dir>/run_metadata.json`

---

## Alpha Mining Agent Architecture

An **evolutionary factor discovery loop** driven by an LLM, with safer constrained factor generation and full harness engineering (checkpointing, deduplication, Telegram alerts, run ledger).

By default, `alpha_miner.py` no longer asks the LLM to write arbitrary Python factor modules. It asks for a structured JSON factor spec, validates the contained DSL expression, compiles it internally into vectorized pandas code, runs a static safety validator, then evaluates the factor through the existing Rank IC pipeline. Legacy full-Python generation is still available with `generation.mode: "python"`.

### Top-Level Flow

```
alpha_mining_config.yaml
        │ MiningConfig.from_yaml()
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         alpha_miner.py — AlphaMiner                         │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║               Evolutionary Loop  (generation 0 → N)                   ║  │
│  ║                                                                       ║  │
│  ║  ┌─────────────┐     ┌────────────────────┐     ┌─────────────────┐  ║  │
│  ║  │  GENERATE   │     │     EVALUATE        │     │     SELECT      │  ║  │
│  ║  │             │     │                    │     │                 │  ║  │
│  ║  │ LLM JSON    │────▶│ DSL compile +      │────▶│ factor_agent_   │  ║  │
│  ║  │ spec        │     │ safety validate    │     │ pipeline.py     │  ║  │
│  ║  │ (few-shot   │     │ factor_dsl.py      │     │ compute + IC    │  ║  │
│  ║  │  survivors) │     │ factor_safety.py   │     │ metrics         │  ║  │
│  ║  └─────────────┘     └────────────────────┘     └────────┬────────┘  ║  │
│  ║         ▲  LLM                                  family-   │           ║  │
│  ║         │  few-shot                             aware     │           ║  │
│  ║         │                                      survivors  │           ║  │
│  ║  ┌──────┴──────┐ ◀──────────────────────────────────────┘            ║  │
│  ║  │   MUTATE    │                                                      ║  │
│  ║  │ DSL mutate  │  window / transform / operator / neutralize / sign    ║  │
│  ║  │ legacy mut. │  legacy Python mutation remains available             ║  │
│  ║  └─────────────┘                                                      ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
│  Harness                                                                    │
│  ├── checkpoint.json        resume interrupted runs                         │
│  ├── factor_ledger.csv      all-time log of every factor evaluated          │
│  ├── MD5 dedup              never re-evaluate identical code                │
│  ├── expression dedup       reject duplicate DSL expressions                │
│  ├── factor/IC correlation  reject duplicated realised factors / IC series  │
│  ├── Telegram notify        progress messages after each generation         │
│  └── max_run_hours          safety wall-clock limit                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

```
GENERATE / CONSTRAIN              EVALUATE                         SCREEN / ANALYSE
─────────────────────             ────────────────────────         ───────────────────────────
factor_code_agent.py              factor_agent_pipeline.py         factor_screener.py
 call_llm()             JSON       compute_factor_df()     .pkl     grade_metrics()
  ├─ call_openai()     ─────────▶  ensure_multiindex()   ────────▶ screen_ic_csv()
  └─ _call_anthropic()  spec       save .pkl                        grade: A / B / C / D / F

factor_dsl.py                     factor_rankic_analysis.py         alpha_miner.py
 FactorSpec.from_json_text()       FactorRankICAnalyzer              walk-forward IC windows
 validate_expression()             .load_market_data()               recent IC / IC t-stat
 compile_expression_to_module()    .calculate_rank_ic()              coverage / autocorr / turnover
 safe pandas vectorization          Rank IC / IC IR / win rate        inverse-alpha labeling

factor_safety.py                  factor_mutator.py                 diversity gates
 validate_factor_code()            mutate_dsl_expression()           expression duplicate
 reject lookahead / leakage         window / transform / operator     factor-value correlation
 reject external APIs / eval        neutralization / sign flip        IC-series correlation

factor_neutralizer.py
 cross_demean
 mktcap OLS residual
 industry group mean
```

### Safe DSL Generation

Default miner generation mode:

```yaml
generation:
  mode: "dsl"
  allowed_fields: [open, high, low, close, volume, amount, vwap, market_cap, industry]
  max_expression_depth: 8
  max_expression_nodes: 80
  max_lookback_window: 252
```

The LLM must return JSON like:

```json
{
  "name": "ranked_20d_momentum",
  "family": "momentum",
  "economic_hypothesis": "Stocks with stronger trailing returns may continue to outperform.",
  "expression": "rank(ts_return(close, 20))",
  "expected_sign": "positive",
  "required_fields": ["close"],
  "lookback_windows": [20],
  "risk_notes": ["May crowd into momentum regimes."],
  "why_not_duplicate": "Uses trailing return only, not volatility or volume divergence."
}
```

Allowed DSL operators include `delay`, `delta`, `ts_return`, `ts_mean`, `ts_std`, `ts_rank`, `ts_zscore`, `ts_corr`, `rank`, `zscore`, `winsorize`, `signed_power`, `log1p_abs`, `neutralize_industry`, and `neutralize_size`. The compiler rejects unknown fields, unknown operators, future access, oversized windows, deep expressions, attribute access, subscript access, and arbitrary Python execution.

### Safety And Robustness Gates

Before any candidate is evaluated, `factor_safety.validate_factor_code()` checks the generated module. Rejected candidates are written to the run ledger with `error="rejected by safety validator: ..."` and are not run through the pipeline.

The validator rejects or flags common alpha-mining hazards:

- Negative shifts such as `shift(-1)`, future labels, `target`, `label`, `y`, `next_return`, `forward_return`
- Future index access patterns, centered rolling windows, unsafe expanding-window usage
- External data/API access (`rqdatac`, `yfinance`, `requests`, `akshare`, `tushare`, `urllib`)
- `eval`, `exec`, subprocesses, shell calls, or file writes inside `compute_factor_df()`
- Obvious non-DataFrame returns and suspicious nested loops

After Rank IC evaluation, the miner adds incremental robustness metrics while preserving the old fields: IC t-stat, coverage, average cross-sectional coverage, factor autocorrelation, turnover estimate, recent-period IC, train/validation/test IC windows, by-year IC, factor-value correlation, IC-series correlation, complexity score, alpha direction (`positive_alpha` vs `inverse_alpha`), and a `keep / investigate / reject` recommendation. Optional long-short, cost-adjusted return, and drawdown estimates can be enabled with:

```yaml
evaluation:
  compute_trade_metrics: true
  transaction_cost_bps: 10.0
```

This option reads `data.pkl` again per factor, so it is disabled by default for speed.

### Walk-Forward Validation

Use date windows to avoid accepting factors that only work in the full sample:

```yaml
evaluation:
  train_start: "2010-01-01"
  train_end: "2018-12-31"
  validation_start: "2019-01-01"
  validation_end: "2022-12-31"
  test_start: "2023-01-01"
  test_end: ""
  recent_start: "2024-01-01"
  recent_end: ""
```

`passes()` now requires the old grade/Rank IC gates plus basic validation/test stability when those windows are configured. Strong negative IC is explicitly labeled as `inverse_alpha` rather than being silently treated the same as positive alpha.

### Diversity And Survivor Selection

The miner now uses multiple duplication checks:

- Code hash deduplication
- DSL canonical-expression deduplication
- Realized factor-value Spearman correlation
- Daily IC-series correlation

Survivor selection is family-aware. It keeps the best overall factor, best factors from different families, a low-correlation factor, a strong recent-period factor, and one exploratory candidate when available. This prevents every survivor from collapsing into the same momentum/reversal variant after a few generations.

### LLM Providers

Both OpenAI and Anthropic are supported everywhere. Set `provider` in `alpha_mining_config.yaml` or pass `--provider` on the CLI:

```bash
# OpenAI (default)
python agent/alpha_miner.py start --model gpt-4.1

# Anthropic / Claude
python agent/alpha_miner.py start --provider anthropic --model claude-sonnet-4-6
```

### Parallelism — Two Levels

| Level | Config key | CLI flag | What it parallelises |
|-------|-----------|----------|----------------------|
| **Level-1** | `evaluation.outer_workers` | `--outer-workers N` | Factor evaluations (each is a subprocess) |
| **Level-2** | `evaluation.workers` | `--workers N` | Per-day IC/decile tasks inside one evaluation |

```bash
# Evaluate 3 factors at once, each using 2 IC workers
python agent/alpha_miner.py start --outer-workers 3 --generations 5 --per-gen 8
```

### Queue Pipeline Mode

The miner can overlap LLM generation with factor evaluation inside each generation:

```yaml
evaluation:
  pipeline_queue_enabled: true
  pipeline_queue_size: 0   # 0 = auto, max(2, 2 * outer_workers)
```

With queue mode on, a candidate is submitted to the evaluation pool as soon as it is generated. The LLM can continue producing later candidates while earlier candidates are already running through `factor_agent_pipeline.py`. This preserves the generation-level checkpoint/report flow, but reduces idle time when LLM calls and factor evaluations are both slow.

Set `pipeline_queue_enabled: false` to use the old deterministic batch path: generate all candidates first, then evaluate them as a batch.

### Running the Alpha Miner

```bash
# Start a new run (reads alpha_mining_config.yaml)
python agent/alpha_miner.py start

# Override key params from CLI
python agent/alpha_miner.py start --generations 3 --per-gen 6 --model gpt-4.1
python agent/alpha_miner.py start --provider anthropic --model claude-sonnet-4-6 --outer-workers 2

# Resume after a crash
python agent/alpha_miner.py resume --run-id mining_20260503_161200

# Check all runs
python agent/alpha_miner.py status

# Regenerate report for a run
python agent/alpha_miner.py report --run-id mining_20260503_161200
```

### Single-Factor & Batch Commands

```
factor_orchestrator.py              batch_factor_analysis.py
 run         → full pipeline         batch_analyze_factors()
 screen-dir  → bulk grade ─────────▶  factor-workers  (level-1 parallelism)
 decay       → decay curve            IC workers      (level-2 parallelism)
 leaderboard → rank all               skip_existing   (resume interrupted)
 compare     → side-by-side
 ├── factor_registry.json  (persistent leaderboard across runs)
 ├── Telegram notifications
 └── Markdown reports per factor
```

### File Map

```
E:/data/
├── data.pkl                  individual A-share daily bars
├── index_data.pkl            benchmark indices (CSI300, CSI500, CSI1000, SSE50, ...)
├── requirements.txt          all Python dependencies
├── agent/
│   ├── alpha_miner.py            main evolutionary loop + harness
│   ├── alpha_mining_config.yaml  all tuning parameters (provider, outer_workers, ...)
│   ├── factor_code_agent.py      LLM calls (OpenAI + Anthropic)
│   ├── factor_dsl.py             constrained JSON/DSL expression compiler
│   ├── factor_safety.py          static safety validator for generated modules
│   ├── factor_mutator.py         DSL + legacy systematic factor variants
│   ├── factor_agent_pipeline.py  compute → pickle → rank IC
│   ├── factor_screener.py        grading (A/B/C/D/F) + thresholds
│   ├── factor_neutralizer.py     mktcap / industry neutralization
│   ├── factor_orchestrator.py    single-factor + batch CLI
│   └── ui/streamlit_app.py       chat UI + pipeline + batch + miner viewer
├── scripts/
│   ├── analysis/
│   │   ├── factor_rankic_analysis.py   Rank IC engine (limit-up NaN-safe)
│   │   ├── batch_factor_analysis.py    bulk Rank IC with parallelism
│   │   └── ic_decay_analysis.py        multi-horizon IC decay curves
│   └── download/
│       └── download_index_data.py      akshare → index_data.pkl
└── agent_runs/
    ├── mining/<run_id>/
    │   ├── checkpoint.json      resume state
    │   ├── factors/             generated .py + .pkl files
    │   ├── ic/                  per-factor Rank IC CSVs
    │   ├── mining_report.md     final summary report
    │   └── top_factors.csv      ranked survivors
    └── mining/factor_ledger.csv all-time evaluated factor log
```

---

## Index Data

`index_data.pkl` is a DataFrame with columns:

| Column | Type | Notes |
|--------|------|-------|
| `date` | datetime64 | trading day |
| `order_book_id` | str | RiceQuant format: `000300.XSHG`, `000905.XSHG`, etc. |
| `open` / `high` / `low` / `close` | float64 | price |
| `volume` | float64 | shares traded |
| `total_turnover` | float64 | CNY value |
| `pct_change` | float64 | daily return % |

**Indices included:**

| `order_book_id` | Name |
|----------------|------|
| `000001.XSHG` | Shanghai Composite |
| `000016.XSHG` | SSE 50 |
| `000300.XSHG` | CSI 300 |
| `000905.XSHG` | CSI 500 |
| `000852.XSHG` | CSI 1000 |
| `399001.XSHE` | Shenzhen Component |
| `399006.XSHE` | ChiNext |

**Code conversion** (akshare → RiceQuant): akshare uses plain numeric codes (`000300`). Codes starting with `39`/`40` get `.XSHE`; all others get `.XSHG`.

---

## Repository Structure

- `agent/` — factor code generation, evolutionary miner, pipeline, Streamlit UI
- `scripts/analysis/` — Rank IC, batch analysis, IC decay
- `scripts/plotting/` — result plotting helpers
- `scripts/download/` — factor and index download scripts
- `scripts/tools/` — data preparation and overview tools

## Notes

- Keep API keys in `.env` only (never commit secrets).
- Keep large datasets/results local; do not upload data artifacts.
- `limit_up_flag` / `limit_down_flag` in `data.pkl` are `object` dtype with `NaN` — always use `!= True` not `.astype(bool)` when filtering.
