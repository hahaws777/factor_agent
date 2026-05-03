# Factor Research Workspace

This repository contains a local factor research workflow, including:

- `agent/`: LLM-assisted factor generation and pipeline runner
- `scripts/`: analysis, batch tools, plotting, and data utilities

## Quick Start

### 1) Environment

- Python 3.10+
- Install dependencies as needed in your environment
- Optional: set `OPENAI_API_KEY` when using LLM generation

### 2) Required Local Data

Place market panel data at project root:

- `data.pkl`

> Note: local data files and generated outputs are ignored by `.gitignore` and are not uploaded.

### 3) Run Preset Pipeline (No LLM Required)

From project root:

```bash
python agent/factor_agent_pipeline.py --preset volatility_20d --artifact-dir agent_runs/smoke --data data.pkl
```

### 4) Generate Factor From Natural Language

```bash
python agent/factor_code_agent.py -d "Your factor description"
python agent/factor_agent_pipeline.py -d "Your factor description" --data data.pkl --artifact-dir agent_runs/my_run
```

### 5) Run Chat UI

```bash
streamlit run agent/ui/streamlit_app.py
```

The UI now supports multi-level parallel controls:
- Level-2 per-day IC workers for pipeline runs
- Batch mode with Level-1 factor workers + Level-2 IC workers
- Backend/device selection (`pandas` or `torch`, with `cuda/cpu/auto`)

## Demo Video

Project demo video:

[Watch demo video](./dbe7618914e497eec52ffa3b81c2d99f_raw.mp4)

## Analysis Scripts

Single-factor Rank IC analysis:

```bash
python scripts/analysis/factor_rankic_analysis.py --factor path/to/factor.pkl
```

Batch analysis:

```bash
python scripts/analysis/batch_factor_analysis.py factors_by_type/alpha101
python scripts/analysis/batch_factor_analysis.py factors_by_type/alpha101 --output-dir rankic_batch_results --factor-workers 4 --ic-workers 8 --backend pandas
```

## Run Metadata

Core workflows now write `run_metadata.json` automatically (unless overridden by CLI):

- `agent/factor_agent_pipeline.py` → `<artifact_dir>/run_metadata.json`
- `scripts/analysis/batch_factor_analysis.py` → `<output_dir>/run_metadata.json`
- `scripts/analysis/factor_compare_dashboard.py` → `<out_dir>/run_metadata.json`

## Alpha Mining Agent Architecture

The agent system is an **evolutionary factor discovery loop** driven by an LLM, with full harness
engineering for production-grade runs (checkpointing, deduplication, Telegram alerts, run ledger).

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
 call_openai()          Python     compute_factor_df()     .pkl     grade_metrics()
 extract_python_code() ─────────▶  ensure_multiindex()   ────────▶ screen_ic_csv()
 validate_python_       code       save .pkl                        grade: A / B / C / D / F
   syntax()                              │
                                         │ .pkl + data.pkl          ic_decay_analysis.py
factor_mutator.py                        ▼                           compute_decay([1,5,10,20])
 window_sweep()             factor_rankic_analysis.py               half-life, peak IC, PNG
 add_transform()             FactorRankICAnalyzer
 llm_refine()                .load_market_data()                   factor_neutralizer.py
                             .calculate_rank_ic()                   cross_demean
                             Rank IC / IC IR / win rate             mktcap OLS residual
                                                                    industry group mean
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
├── agent/
│   ├── alpha_miner.py            main evolutionary loop + harness
│   ├── alpha_mining_config.yaml  all tuning parameters
│   ├── factor_code_agent.py      LLM code generation (OpenAI)
│   ├── factor_mutator.py         systematic factor variants
│   ├── factor_agent_pipeline.py  compute → pickle → rank IC
│   ├── factor_screener.py        grading (A/B/C/D/F) + thresholds
│   ├── factor_neutralizer.py     mktcap / industry neutralization
│   └── factor_orchestrator.py   single-factor + batch CLI
├── scripts/
│   └── analysis/
│       ├── factor_rankic_analysis.py   Rank IC engine
│       ├── batch_factor_analysis.py    bulk Rank IC with parallelism
│       └── ic_decay_analysis.py        multi-horizon IC decay curves
└── agent_runs/
    ├── mining/<run_id>/
    │   ├── checkpoint.json      resume state
    │   ├── factors/             generated .py + .pkl files
    │   ├── ic/                  per-factor Rank IC CSVs
    │   ├── mining_report.md     final summary report
    │   └── top_factors.csv      ranked survivors
    └── mining/factor_ledger.csv all-time evaluated factor log
```

### Running the Alpha Miner

```bash
# Start a new mining run (reads alpha_mining_config.yaml)
python agent/alpha_miner.py start

# Override key params from CLI
python agent/alpha_miner.py start --generations 3 --per-gen 6 --model gpt-4.1

# Resume after a crash
python agent/alpha_miner.py resume --run-id mining_20260503_161200

# Check all runs
python agent/alpha_miner.py status

# Regenerate report for a run
python agent/alpha_miner.py report --run-id mining_20260503_161200
```

Set these environment variables to enable Telegram notifications:

```bash
export TELEGRAM_BOT_TOKEN=<token>
export TELEGRAM_CHAT_ID=<your_chat_id>
```

## Repository Structure

- `agent/`: factor code generation, evolutionary miner, pipeline, Streamlit UI
- `scripts/analysis/`: Rank IC, batch analysis, IC decay
- `scripts/plotting/`: result plotting helpers
- `scripts/download/`: factor download scripts
- `scripts/tools/`: data preparation and overview tools

## Notes

- Keep API keys in `.env` only (never commit secrets).
- Keep large datasets/results local; do not upload data artifacts.
