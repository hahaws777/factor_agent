# Factor agent (LLM → pickle → Rank IC / decile / plots)

## Put this repo inside your quant project

This package expects to live in **`<project_root>/agent/`**, where `project_root` also contains **`data.pkl`** and **`scripts/analysis/factor_rankic_analysis.py`** (sibling paths to `agent/`).

```bash
cd /path/to/your_quant_project
git clone https://github.com/hahaws777/factor-agent.git agent
cd agent
copy .env.example .env   # Windows: set OPENAI_API_KEY
```

If your project already has an `agent/` folder, merge or replace with the clone.

### Maintainer: push this repo (after creating it on GitHub)

1. On GitHub, **New repository** → name e.g. `factor-agent` (empty, no README).  
2. In this folder (the git root):  
   `git remote add origin https://github.com/hahaws777/factor-agent.git` (adjust name if needed)  
   `git push -u origin main`

---

Run from **project root** (example `e:\data`):

```bash
python agent\factor_code_agent.py -d "Your factor description"
python agent\factor_agent_pipeline.py -d "Your factor description" --data data.pkl --artifact-dir agent_runs\my_run
python agent\factor_agent_pipeline.py --preset volatility_20d --artifact-dir agent_runs\smoke --data data.pkl
```

- **`factor_code_agent.py`** — OpenAI only, writes `generated_factors/*.py`.
- **`factor_agent_pipeline.py`** — generate (or `--preset`), run `compute_factor_df()`, save `.pkl`, call `scripts/analysis/factor_rankic_analysis.py` (IC, decile backtest, plots).
- **`templates/`** — preset factor sources copied into `generated_factors/` when using `--preset`.

`scripts/factor_code_agent.py` and `scripts/factor_agent_pipeline.py` are thin wrappers that forward here.

Requires `.env` with `OPENAI_API_KEY` when not using `--preset`. Market panel: `data.pkl` at project root.

**Data rule:** The LLM is instructed to load **only** from `data.pkl` (`Path(__file__).resolve().parents[1] / "data.pkl"`). It must **not** use rqdatac, Tushare, or other remote APIs in generated `compute_factor_df()` code.

## Chat UI (local)

```bash
cd /d e:\data
pip install streamlit
streamlit run agent/ui/streamlit_app.py
```

Opens a browser (default `http://localhost:8501`): multi-turn chat, streaming replies, save generated code to `generated_factors/`, and one-click **Run pipeline** (pickle + Rank IC + decile + plots) using `factor_agent_pipeline.py`. After a successful run, **scroll below the chat**: the main column shows **Rank IC metrics**, a **daily IC line chart**, and **decile PNGs** from `backtest_plots/` (this block renders after the sidebar updates session state).

### Multi-level parallel in UI

- Pipeline panel: configure Level-2 per-day IC workers + backend/device (`pandas` / `torch` + `auto/cuda/cpu`)
- Batch panel: run `scripts/analysis/batch_factor_analysis.py` with:
  - Level-1 `factor-workers` (cross-factor parallel)
  - Level-2 `ic-workers` (inside each factor/day IC parallel)
