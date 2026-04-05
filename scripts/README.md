# Python scripts (project root = `e:\data`)

Run from project root when possible. Several tools `chdir` to `e:\data` internally.

## Data vs agent (what is what)

| Path | What it is |
|------|------------|
| **`data.pkl`** | Daily panel for Python analysis: `order_book_id`, `date`, `close`, `ST`, `limit_up_flag`, `limit_down_flag`, `suspended`, … Used by `analysis/factor_rankic_analysis.py` (merge + Rank IC + decile backtest). |
| **`data.npz`** | Same market in array form (`data`, `columns`). |
| **`market_data_cpp.npz`** + **`market_data_cpp_arrays/`** | Compact market aligned with C++/npz factor indices. |
| **`factors_by_type/`** | RQ downloaded factors, one `.pkl` per factor (MultiIndex + one column). |
| **`factors_by_type_npy/factors_by_type/`** | Raw `.npz` factors (`values`, `index_order_book_id`, `index_date`). |
| **`generated_factors/`** | LLM output: `.py` (must define `compute_factor_df()`) and `.pkl` saved by pipeline. |
| **`rankic_batch_results/`**, **`rankic_cpp_batch_results/`** | Rank IC CSV outputs (Python / C++ batches). |
| **`decile_cpp_batch_results/`**, **`decile_plots/`** | C++ decile daily/cum CSV and plots. |
| **`.env`** | `OPENAI_API_KEY` for `agent/factor_code_agent.py`. |

The LLM system prompt in `agent/factor_code_agent.py` embeds the same table for code generation.

## Factor agent → auto backtest

All agent entrypoints live under **`agent/`** (see that folder). `scripts\factor_*.py` still forwards to `agent/` for old commands.

1. **Generate code only** (OpenAI):

```bash
cd /d e:\data
python agent\factor_code_agent.py -d "20-day return momentum, cross-sectional rank each day"
```

2. **Generate + run `compute_factor_df()` + save `.pkl` + Rank IC / decile** (needs working rq init in generated code and existing **`data.pkl`**):

```bash
python agent\factor_agent_pipeline.py -d "20-day return momentum" --data data.pkl
```

Options: `--skip-generate path\to\gen.py`, `--no-backtest`, `--no-next-day`, `--rankic-out path.csv`, `--preset volatility_20d`, `--artifact-dir`, `--no-plot-backtest`.

## Folder index

| Folder | Scripts |
|--------|---------|
| `download/` | Factor download (rqdatac) |
| `analysis/` | Rank IC single & batch, rankic summary |
| `composite/` | PCA + LightGBM factor composite, return plots |
| `plotting/` | Batch decile result plots |
| `tools/` | `prepare_data`, `data_overview` |
| `factors_npy/` | Pack/helper npz scripts |
| **`agent/`** (project root) | `factor_code_agent.py`, `factor_agent_pipeline.py`, `templates/` |
| (root `scripts/`) | Thin wrappers that call `agent/*.py` |

## Examples

```bash
cd /d e:\data
python scripts\analysis\factor_rankic_analysis.py --factor factors_by_type\alpha101\xxx.pkl --data data.pkl --next-day --backtest-decile

python scripts\composite\factor_composite_pca_lgb.py --decile_dir decile_cpp_batch_results

python scripts\download\download_factors_by_type_priority.py
```
