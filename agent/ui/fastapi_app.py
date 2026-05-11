#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI backend + static frontend replacing Streamlit UI flows.

Run from project root:
  uvicorn agent.ui.fastapi_app:app --host 0.0.0.0 --port 8510
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    import job_queue as _jq
except Exception:
    _jq = None  # type: ignore[assignment]

from factor_code_agent import SYSTEM_PROMPT, extract_python_code  # noqa: E402


DOTENV = ROOT / ".env"
MINING_DIR = ROOT / "agent_runs" / "mining"
MINING_UI_STATE = MINING_DIR / "ui_state.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"
GENERATED_FACTORS_DIR = ROOT / "generated_factors"

PROVIDER_MODELS = {
    "openai": ["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"],
}

CHAT_SYSTEM = (
    SYSTEM_PROMPT
    + """

## Interactive chat mode (FastAPI UI)
- You may use short replies in English or 中文 before/after code.
- When you generate or revise factor code, put the **entire** runnable module in **one** markdown ```python code block.
- Data: only `data.pkl` via `Path(__file__).resolve().parents[1] / "data.pkl"`. Do not suggest or use rqdatac/Tushare/cloud APIs.
- Pandas: never index groupby columns with `series.name` (often None). Assign expressions to `df["col_name"]` first, then `groupby("date")["col_name"]`.
- If the user only asks a conceptual question, answer without code.
"""
)

app = FastAPI(title="Factor Agent API", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(default_factory=list)
    provider: str = "openai"
    model: str = "gpt-4.1"
    temperature: float = 0.2


class SaveCodeRequest(BaseModel):
    code: str
    filename: str
    model: str = ""


class PipelineSubmitRequest(BaseModel):
    code: str = ""
    factor_path: str = ""
    save_name: str = ""
    data_pkl: str = "data.pkl"
    artifact_dir: str = "agent_runs/chat_ui"
    workers: int = 1
    backend: str = "pandas"
    device: str = "auto"
    provider: str = "openai"
    return_horizon: str = "next-day forward"


class BatchSubmitRequest(BaseModel):
    factor_dir: str = "factors_by_type"
    output_dir: str = "rankic_batch_results"
    data_pkl: str = "data.pkl"
    factor_workers: int = 1
    ic_workers: int = 1
    backend: str = "pandas"
    device: str = "auto"
    return_horizon: str = "next-day forward"


class MiningStartRequest(BaseModel):
    run_id: str
    config_path: str
    data_pkl: str = "data.pkl"
    generations: int = 1
    per_gen: int = 3
    outer_workers: int = 1
    provider: str = "openai"
    model: str = "gpt-4.1"


class MiningResumeRequest(BaseModel):
    run_id: str
    config_path: str


class UiStateRequest(BaseModel):
    run_id: str = ""
    run_dir: str = ""
    pid: int | None = None


def _load_dotenv() -> None:
    if not DOTENV.is_file():
        return
    for line in DOTENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _ensure_provider_key(provider: str) -> None:
    _load_dotenv()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY is not set.")
    else:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set.")


def _safe_root_path(raw: str, must_exist: bool = False) -> Path:
    p = Path(str(raw))
    text = str(raw).replace("\\", "/")
    if not p.is_absolute():
        p = ROOT / p
    elif re.match(r"^[A-Za-z]:/", text):
        # Allow canonical project Windows path when running from WSL-like contexts.
        for prefix in ("E:/data", "e:/data"):
            if text.startswith(prefix):
                p = ROOT / text[len(prefix):].lstrip("/")
                break
    rp = p.resolve()
    if not rp.is_relative_to(ROOT):
        raise HTTPException(status_code=400, detail="Path must be under project root.")
    if must_exist and not rp.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {rp}")
    return rp


def _tail_text(path: Path, max_lines: int = 120, max_bytes: int = 256 * 1024) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_bytes = min(size, int(max_bytes))
            if read_bytes <= 0:
                return ""
            f.seek(size - read_bytes)
            buf = f.read(read_bytes)
        lines = buf.decode("utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"Could not read log: {exc}"
    return "\n".join(lines[-max_lines:])


def _require_job_queue():
    if _jq is None:
        raise HTTPException(status_code=503, detail="job_queue is unavailable in current environment.")
    return _jq


def _list_mining_runs(limit: int = 50) -> list[Path]:
    if not MINING_DIR.is_dir():
        return []
    markers = {"checkpoint.json", "ui_alpha_miner.log", "config_used.yaml", "mining_report.md", "top_factors.csv"}
    dirs: list[Path] = []
    for d in MINING_DIR.iterdir():
        if not d.is_dir():
            continue
        if any((d / marker).exists() for marker in markers) or (d / "factors").is_dir():
            dirs.append(d)
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dirs[: max(1, int(limit))]


def _read_checkpoint(run_dir: Path) -> dict[str, Any]:
    cp = run_dir / "checkpoint.json"
    if not cp.is_file():
        return {}
    try:
        return json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_summary(run_dir: Path) -> dict[str, Any]:
    state = _read_checkpoint(run_dir)
    candidates = state.get("all_candidates", []) if state else []
    if not isinstance(candidates, list):
        candidates = []
    run_id = run_dir.name
    best_grade = state.get("best_grade", "?") if state else "?"
    best_mean_ric = state.get("best_mean_ric", 0.0) if state else 0.0
    generations_done = state.get("generations_done", 0) if state else 0
    survivors = state.get("survivors", []) if state else []
    top_csv = run_dir / "top_factors.csv"
    log_path = run_dir / "ui_alpha_miner.log"
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "checkpoint_exists": bool(state),
        "generations_done": int(generations_done or 0),
        "best_grade": str(best_grade),
        "best_mean_rank_ic": float(best_mean_ric or 0.0),
        "candidate_count": len(candidates),
        "survivor_count": len(survivors) if isinstance(survivors, list) else 0,
        "top_factors_csv": str(top_csv) if top_csv.is_file() else "",
        "log_path": str(log_path) if log_path.is_file() else "",
        "updated_at": state.get("updated_at") if isinstance(state, dict) else "",
    }


def _chat_completion(req: ChatRequest) -> str:
    _ensure_provider_key(req.provider)
    if req.provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())
        with client.messages.stream(
            model=req.model,
            max_tokens=8096,
            system=CHAT_SYSTEM,
            messages=req.messages,
            temperature=float(req.temperature),
        ) as stream:
            chunks: list[str] = []
            for token in stream.text_stream:
                if token:
                    chunks.append(token)
        return "".join(chunks)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    stream = client.chat.completions.create(
        model=req.model,
        messages=[{"role": "system", "content": CHAT_SYSTEM}] + req.messages,
        temperature=float(req.temperature),
        stream=True,
    )
    chunks = []
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            chunks.append(token)
    return "".join(chunks)


def _save_generated_code(code: str, filename: str, model: str = "") -> Path:
    GENERATED_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    if not safe_name.endswith(".py"):
        safe_name = f"{safe_name}.py"
    save_path = GENERATED_FACTORS_DIR / safe_name
    header = (
        f"# Saved from FastAPI UI — {datetime.now().isoformat(timespec='seconds')}\n"
        f"# Model: {model or 'unknown'}\n\n"
    )
    save_path.write_text(header + code, encoding="utf-8")
    return save_path


def _write_mining_ui_state(run_dir: Path, pid: int | None = None) -> None:
    MINING_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_run_id": run_dir.name,
        "current_run_dir": str(run_dir.resolve()),
        "pid": int(pid) if pid else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    MINING_UI_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_mining_ui_state() -> dict[str, Any]:
    if not MINING_UI_STATE.is_file():
        return {}
    try:
        return json.loads(MINING_UI_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


@app.on_event("startup")
def _startup() -> None:
    if _jq is not None:
        try:
            _jq.init_db()
        except Exception:
            pass
    GENERATED_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    MINING_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    _load_dotenv()
    return {
        "ok": True,
        "root": str(ROOT),
        "job_queue_available": _jq is not None,
        "providers": PROVIDER_MODELS,
        "has_openai_key": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "has_anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
    }


@app.post("/api/chat/completion")
def chat_completion(req: ChatRequest) -> dict[str, Any]:
    if req.provider not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {req.provider}")
    if req.model not in PROVIDER_MODELS.get(req.provider, []):
        raise HTTPException(status_code=400, detail=f"Model {req.model} does not belong to provider {req.provider}")
    try:
        text = _chat_completion(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {exc}") from exc
    extracted = extract_python_code(text)
    return {"assistant": text, "extracted_code": extracted, "has_compute_factor": "def compute_factor_df" in extracted}


@app.post("/api/code/save")
def save_code(req: SaveCodeRequest) -> dict[str, Any]:
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="code is empty.")
    path = _save_generated_code(req.code, req.filename, req.model)
    return {
        "ok": True,
        "path": str(path),
        "relative_path": str(path.relative_to(ROOT)),
    }


@app.post("/api/jobs/pipeline")
def submit_pipeline(req: PipelineSubmitRequest) -> dict[str, Any]:
    jq = _require_job_queue()
    factor_path = req.factor_path.strip()
    if req.code.strip():
        save_name = req.save_name.strip() or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        save_path = _save_generated_code(req.code, save_name, model="")
        factor_path = str(save_path)
    if not factor_path:
        raise HTTPException(status_code=400, detail="Either code or factor_path is required.")
    factor_abs = _safe_root_path(factor_path, must_exist=True)
    pipe = AGENT_DIR / "factor_agent_pipeline.py"
    cmd = [
        sys.executable,
        str(pipe),
        "--skip-generate",
        str(factor_abs),
        "--data",
        str(req.data_pkl),
        "--artifact-dir",
        str(req.artifact_dir),
    ]
    if int(req.workers) > 1:
        cmd.extend(["--workers", str(int(req.workers))])
    cmd.extend(
        [
            "--backend",
            str(req.backend),
            "--device",
            str(req.device).strip() or "auto",
            "--provider",
            str(req.provider),
        ]
    )
    if req.return_horizon == "same-day diagnostic":
        cmd.append("--no-next-day")

    job_id = jq.submit("pipeline", {"cmd": cmd, "cwd": str(ROOT), "artifact_dir": str(req.artifact_dir)})
    return {"ok": True, "job_id": int(job_id), "cmd": cmd}


@app.post("/api/jobs/batch")
def submit_batch(req: BatchSubmitRequest) -> dict[str, Any]:
    jq = _require_job_queue()
    script = ROOT / "scripts" / "analysis" / "batch_factor_analysis.py"
    cmd = [
        sys.executable,
        str(script),
        str(req.factor_dir),
        "--output-dir",
        str(req.output_dir),
        "--data",
        str(req.data_pkl),
        "--factor-workers",
        str(int(req.factor_workers)),
        "--ic-workers",
        str(int(req.ic_workers)),
        "--backend",
        str(req.backend),
        "--device",
        str(req.device).strip() or "auto",
    ]
    if req.return_horizon == "same-day diagnostic":
        cmd.append("--no-next-day")
    job_id = jq.submit("batch_analysis", {"cmd": cmd, "cwd": str(ROOT), "artifact_dir": str(req.output_dir)})
    return {"ok": True, "job_id": int(job_id), "cmd": cmd}


@app.post("/api/jobs/mining/start")
def submit_mining_start(req: MiningStartRequest) -> dict[str, Any]:
    jq = _require_job_queue()
    run_id = req.run_id.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required.")
    run_dir = MINING_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(AGENT_DIR / "alpha_miner.py"),
        "start",
        "--run-id",
        run_id,
        "--config",
        str(req.config_path),
        "--generations",
        str(int(req.generations)),
        "--per-gen",
        str(int(req.per_gen)),
        "--model",
        str(req.model),
        "--provider",
        str(req.provider),
        "--outer-workers",
        str(int(req.outer_workers)),
        "--data",
        str(req.data_pkl),
    ]
    job_id = jq.submit("mining_start", {"cmd": cmd, "cwd": str(ROOT)}, run_id=run_id)
    _write_mining_ui_state(run_dir, pid=None)
    return {"ok": True, "job_id": int(job_id), "run_id": run_id, "cmd": cmd}


@app.post("/api/jobs/mining/resume")
def submit_mining_resume(req: MiningResumeRequest) -> dict[str, Any]:
    jq = _require_job_queue()
    run_id = req.run_id.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required.")
    run_dir = MINING_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(AGENT_DIR / "alpha_miner.py"),
        "resume",
        "--run-id",
        run_id,
        "--config",
        str(req.config_path),
    ]
    job_id = jq.submit("mining_resume", {"cmd": cmd, "cwd": str(ROOT)}, run_id=run_id)
    _write_mining_ui_state(run_dir, pid=None)
    return {"ok": True, "job_id": int(job_id), "run_id": run_id, "cmd": cmd}


@app.get("/api/jobs")
def list_jobs(limit: int = Query(50, ge=1, le=300)) -> dict[str, Any]:
    if _jq is None:
        return {"items": [], "count": 0}
    try:
        items = _jq.list_jobs(limit=int(limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {exc}") from exc
    return {"items": items, "count": len(items)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int) -> dict[str, Any]:
    jq = _require_job_queue()
    try:
        job = jq.get_job(int(job_id))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {exc}") from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/jobs/run/{run_id}")
def get_jobs_for_run(run_id: str, limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    jq = _require_job_queue()
    try:
        jobs = jq.list_jobs_for_run(run_id, limit=int(limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs for run: {exc}") from exc
    return {"items": jobs, "count": len(jobs)}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int) -> dict[str, Any]:
    jq = _require_job_queue()
    try:
        jq.cancel_job(int(job_id))
        job = jq.get_job(int(job_id))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {exc}") from exc
    return {"ok": True, "job": job}


@app.get("/api/ui-state/mining")
def get_mining_ui_state() -> dict[str, Any]:
    return _read_mining_ui_state()


@app.post("/api/ui-state/mining")
def set_mining_ui_state(req: UiStateRequest) -> dict[str, Any]:
    run_dir = req.run_dir.strip()
    run_id = req.run_id.strip()
    if run_dir:
        rp = _safe_root_path(run_dir, must_exist=False)
        _write_mining_ui_state(rp, req.pid)
        return {"ok": True, "run_dir": str(rp)}
    if run_id:
        rp = (MINING_DIR / run_id).resolve()
        _write_mining_ui_state(rp, req.pid)
        return {"ok": True, "run_dir": str(rp)}
    raise HTTPException(status_code=400, detail="run_id or run_dir is required.")


@app.get("/api/runs")
def list_runs(limit: int = Query(30, ge=1, le=200)) -> dict[str, Any]:
    runs = _list_mining_runs(limit=limit)
    return {"items": [_run_summary(r) for r in runs], "count": len(runs)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run_dir = MINING_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    state = _read_checkpoint(run_dir)
    summary = _run_summary(run_dir)
    return {
        "summary": summary,
        "config": state.get("config", {}) if isinstance(state, dict) else {},
        "survivors": state.get("survivors", []) if isinstance(state, dict) else [],
    }


@app.get("/api/runs/{run_id}/candidates")
def get_run_candidates(
    run_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_rejected: bool = Query(True),
    family: str = Query("all"),
) -> dict[str, Any]:
    run_dir = MINING_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    state = _read_checkpoint(run_dir)
    candidates = state.get("all_candidates", []) if isinstance(state, dict) else []
    if not isinstance(candidates, list):
        candidates = []

    slim: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        err = str(item.get("error") or "")
        fam = str(item.get("family") or "")
        if not include_rejected and err:
            continue
        if family != "all" and fam != family:
            continue
        slim.append(
            {
                "name": item.get("name", ""),
                "generation": item.get("generation"),
                "origin": item.get("origin", ""),
                "family": fam,
                "grade": item.get("grade", ""),
                "mean_rank_ic": item.get("mean_rank_ic"),
                "rank_ic_ir": item.get("rank_ic_ir"),
                "rank_ic_win_rate": item.get("rank_ic_win_rate"),
                "alpha_direction": item.get("alpha_direction", ""),
                "error": err,
                "expression": item.get("expression", ""),
            }
        )
    slim.sort(key=lambda x: abs(float(x.get("mean_rank_ic") or 0.0)), reverse=True)
    total = len(slim)
    page = slim[offset : offset + limit]
    return {
        "items": page,
        "count": len(page),
        "total": total,
        "offset": offset,
        "limit": limit,
        "families": sorted({str(c.get("family") or "") for c in slim if str(c.get("family") or "")}),
    }


@app.get("/api/runs/{run_id}/files")
def get_run_files(run_id: str) -> dict[str, Any]:
    run_dir = MINING_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    factors_dir = run_dir / "factors"
    ic_dir = run_dir / "ic"
    bt_dir = run_dir / "backtest_results"
    plot_dir = run_dir / "backtest_plots"
    return {
        "factors_py": sorted([str(p.relative_to(ROOT)) for p in factors_dir.glob("*.py")]) if factors_dir.is_dir() else [],
        "factors_pkl": sorted([str(p.relative_to(ROOT)) for p in factors_dir.glob("*.pkl")]) if factors_dir.is_dir() else [],
        "ic_csv": sorted([str(p.relative_to(ROOT)) for p in ic_dir.glob("*.csv")]) if ic_dir.is_dir() else [],
        "backtest_csv": sorted([str(p.relative_to(ROOT)) for p in bt_dir.glob("*.csv")]) if bt_dir.is_dir() else [],
        "backtest_png": sorted([str(p.relative_to(ROOT)) for p in plot_dir.glob("*.png")]) if plot_dir.is_dir() else [],
    }


@app.get("/api/logs/tail")
def get_log_tail(
    path: str = Query(..., description="Path under project root"),
    lines: int = Query(120, ge=10, le=500),
) -> dict[str, Any]:
    rp = _safe_root_path(path, must_exist=True)
    if not rp.is_file():
        raise HTTPException(status_code=404, detail="Log file not found.")
    return {"path": str(rp), "tail": _tail_text(rp, max_lines=int(lines))}
