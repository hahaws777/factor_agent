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

## Repository Structure

- `agent/`: factor code generation, pipeline, Streamlit UI
- `scripts/analysis/`: Rank IC and batch analysis
- `scripts/plotting/`: result plotting helpers
- `scripts/download/`: factor download scripts
- `scripts/tools/`: data preparation and overview tools

## Notes

- Keep API keys in `.env` only (never commit secrets).
- Keep large datasets/results local; do not upload data artifacts.
