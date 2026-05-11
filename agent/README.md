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

## Alpha miner safety mode

`alpha_miner.py` now defaults to a safer constrained workflow:

1. The LLM returns a JSON factor spec, not a full Python module.
2. `factor_dsl.py` validates and compiles the DSL expression into vectorized pandas code.
3. `factor_safety.py` rejects unsafe code before evaluation, including lookahead patterns, target/label columns, external APIs, `eval`/`exec`, shell calls, file writes, and obvious non-DataFrame returns.
4. The normal pipeline still computes `.pkl` and Rank IC, then the miner adds walk-forward/recent IC, coverage, autocorrelation, turnover, complexity, inverse-alpha labeling, factor-value correlation, and IC-series similarity.
5. Survivor selection is family-aware, so one generation cannot be dominated by near-identical momentum/reversal variants.

Use `agent/alpha_mining_config.yaml` to tune:

```yaml
generation:
  mode: "dsl"   # set to "python" only for legacy full-module generation

evaluation:
  train_start: ""
  validation_start: ""
  test_start: ""
  recent_start: ""
  pipeline_queue_enabled: true
  pipeline_queue_size: 0
  compute_trade_metrics: false
```

Unsafe or duplicate candidates are retained in the ledger/report with rejection reasons instead of being silently discarded.

Queue mode submits each candidate to the evaluation pool as soon as it is generated, so LLM generation and `factor_agent_pipeline.py` evaluation can overlap within the same generation. Set `pipeline_queue_enabled: false` to restore the old generate-then-evaluate batch flow.

## Async Job Queue

Long-running tasks — pipeline runs, batch analysis, alpha mining — are submitted to a SQLite job queue (`agent_runs/jobs.db`) and executed by a background worker. This prevents the Streamlit UI from blocking.

**`job_queue.py`** — the queue API (no external dependencies, stdlib `sqlite3` only):

| Function | Description |
|----------|-------------|
| `init_db()` | Create `jobs` table if absent (idempotent) |
| `submit(job_type, params, run_id) → id` | Enqueue a new pending job |
| `claim_next() → dict \| None` | Atomically claim one pending job (`BEGIN IMMEDIATE`) |
| `update_status(job_id, status, *, error, pid, log_path)` | Partial update; sets `finished_at` on terminal status |
| `get_job(job_id) → dict \| None` | Fetch one job row |
| `list_jobs(limit) → list[dict]` | Most recent jobs, newest first |
| `cancel_job(job_id)` | SIGTERM stored pid, mark cancelled |

**`job_worker.py`** — the consumer process:

```bash
# From the project root:
python agent/job_worker.py

# Slower polling when idle:
python agent/job_worker.py --poll-interval 5
```

The worker claims one pending job at a time, pipes stdout/stderr to `agent_runs/job_logs/<job_id>.log`, stores the subprocess pid in the DB so the UI can cancel it, and marks the job `success` (exit 0) or `failed` (non-zero). A crash in one job never crashes the worker. Stop cleanly with `Ctrl-C` or `SIGTERM`.

## FastAPI UI (local)

```bash
# Start worker + FastAPI UI in one tmux session
cd e:\data
bash start.sh
```

Opens a browser (default `http://localhost:8510`): multi-turn chat, factor generation, save generated code to `generated_factors/`.

**Run pipeline / batch / mining** actions submit jobs to the queue instantly — the UI never blocks. Use the **Jobs / Logs** tab to monitor job status, cancel running jobs, and inspect log tails.

After a pipeline or mining job succeeds, use **Alpha Mining Console** and **Jobs / Logs** tabs to inspect run summaries, candidate pages, generated files, and logs.

### Multi-level parallel in UI

- Pipeline panel: configure Level-2 per-day IC workers + backend/device (`pandas` / `torch` + `auto/cuda/cpu`)
- Batch panel: run `scripts/analysis/batch_factor_analysis.py` with:
  - Level-1 `factor-workers` (cross-factor parallel)
  - Level-2 `ic-workers` (inside each factor/day IC parallel)

### Run metadata

Pipeline runs write `run_metadata.json` automatically in the artifact directory (or to a custom path via `--metadata-out`).
