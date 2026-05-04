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
python scripts/analysis/factor_rankic_analysis.py --factor path/to/factor.pkl --next-day
```

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

An **evolutionary factor discovery loop** driven by an LLM, with full harness engineering (checkpointing, deduplication, Telegram alerts, run ledger).

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
│  ║  │ LLM prompt  │────▶│ factor_agent_      │────▶│ factor_screener │  ║  │
│  ║  │ (few-shot   │     │ pipeline.py        │     │ grade: A–F      │  ║  │
│  ║  │  survivors) │     │ compute + rank IC  │     │ diversity pen.  │  ║  │
│  ║  └─────────────┘     └────────────────────┘     └────────┬────────┘  ║  │
│  ║         ▲  LLM                                    top-k   │           ║  │
│  ║         │  few-shot                             survivors │           ║  │
│  ║  ┌──────┴──────┐ ◀──────────────────────────────────────┘            ║  │
│  ║  │   MUTATE    │                                                      ║  │
│  ║  │ window_sweep│  factor_mutator.py                                   ║  │
│  ║  │ add_transform│  (winsorize / rank / zscore)                        ║  │
│  ║  │ llm_refine  │                                                      ║  │
│  ║  └─────────────┘                                                      ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
│  Harness                                                                    │
│  ├── checkpoint.json        resume interrupted runs                         │
│  ├── factor_ledger.csv      all-time log of every factor evaluated          │
│  ├── MD5 dedup              never re-evaluate identical code                │
│  ├── Telegram notify        progress messages after each generation         │
│  └── max_run_hours          safety wall-clock limit                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

```
GENERATE                          EVALUATE                         SCREEN / ANALYSE
─────────────────────             ────────────────────────         ───────────────────────────
factor_code_agent.py              factor_agent_pipeline.py         factor_screener.py
 call_llm()             Python     compute_factor_df()     .pkl     grade_metrics()
  ├─ call_openai()     ─────────▶  ensure_multiindex()   ────────▶ screen_ic_csv()
  └─ _call_anthropic()  code       save .pkl                        grade: A / B / C / D / F
 extract_python_code()                   │
 validate_python_syntax()                │ .pkl + data.pkl          ic_decay_analysis.py
                                         ▼                           compute_decay([1,5,10,20])
factor_mutator.py             factor_rankic_analysis.py             half-life, peak IC, PNG
 window_sweep()                FactorRankICAnalyzer
 add_transform()                .load_market_data()                factor_neutralizer.py
 llm_refine()                   .calculate_rank_ic()                cross_demean
                                 Rank IC / IC IR / win rate          mktcap OLS residual
                                                                     industry group mean
```

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
│   ├── factor_code_agent.py      LLM code generation (OpenAI + Anthropic)
│   ├── factor_mutator.py         systematic factor variants
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
