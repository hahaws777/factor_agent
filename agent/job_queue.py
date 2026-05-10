#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite-backed job queue for async request processing.

Jobs are submitted by the Streamlit UI and consumed by job_worker.py.
All persistence lives in agent_runs/jobs.db — no external broker needed.

Status lifecycle:  pending → running → success | failed | cancelled
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "agent_runs" / "jobs.db"

_TERMINAL = {"success", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def _conn():
    """Open a connection, commit on clean exit, rollback on exception."""
    con = sqlite3.connect(str(DB_PATH), timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create the jobs table if it does not exist (idempotent)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type     TEXT    NOT NULL,
                params_json  TEXT    NOT NULL DEFAULT '{}',
                status       TEXT    NOT NULL DEFAULT 'pending',
                created_at   TEXT    NOT NULL,
                started_at   TEXT,
                finished_at  TEXT,
                log_path     TEXT    DEFAULT '',
                error        TEXT    DEFAULT '',
                pid          INTEGER,
                run_id       TEXT    DEFAULT ''
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_run_id ON jobs (run_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs (status)")


def submit(job_type: str, params: dict, run_id: str = "") -> int:
    """
    Insert a pending job and return its auto-assigned id.

    params must contain at least {"cmd": [...], "cwd": "..."}.
    Any extra keys (e.g. artifact_dir) are preserved for the UI to read back.
    """
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO jobs (job_type, params_json, status, created_at, run_id)
               VALUES (?, ?, 'pending', ?, ?)""",
            (job_type, json.dumps(params, default=str), _now(), run_id),
        )
        return cur.lastrowid  # type: ignore[return-value]


def claim_next() -> dict | None:
    """
    Atomically claim one pending job and mark it as running.

    Uses BEGIN IMMEDIATE to prevent two workers from claiming the same row.
    Returns the claimed job as a plain dict, or None if the queue is empty.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=10)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            con.rollback()
            return None
        started = _now()
        con.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
            (started, row["id"]),
        )
        con.commit()
        # Return the post-update state (SELECT snapshot predates the UPDATE).
        result = dict(row)
        result["status"] = "running"
        result["started_at"] = started
        return result
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def update_status(
    job_id: int,
    status: str,
    *,
    error: str = "",
    pid: int | None = None,
    log_path: str = "",
) -> None:
    """
    Update a job's status and any subset of (error, pid, log_path).

    finished_at is set automatically when status is terminal.
    Existing values are preserved when the new value is empty/None.
    """
    finished = _now() if status in _TERMINAL else None
    with _conn() as con:
        con.execute(
            """UPDATE jobs
               SET status      = ?,
                   finished_at = COALESCE(?, finished_at),
                   error       = COALESCE(NULLIF(?, ''), error),
                   pid         = COALESCE(?, pid),
                   log_path    = COALESCE(NULLIF(?, ''), log_path)
               WHERE id = ?""",
            (status, finished, error or None, pid, log_path or None, job_id),
        )


def get_job(job_id: int) -> dict | None:
    """Return the job row as a plain dict, or None if not found."""
    with _conn() as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` jobs, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def list_jobs_for_run(run_id: str, limit: int = 12) -> list[dict]:
    """Return jobs matching run_id, newest first — avoids full-table fetch."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM jobs WHERE run_id = ? ORDER BY id DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def cancel_job(job_id: int) -> None:
    """
    Cancel a pending or running job.

    Sends SIGTERM to the worker subprocess if a pid is stored,
    then marks the job cancelled regardless of whether the signal succeeded.
    """
    job = get_job(job_id)
    if job is None:
        return
    if job["status"] == "running" and job.get("pid"):
        try:
            os.kill(int(job["pid"]), signal.SIGTERM)
        except OSError:
            pass
    with _conn() as con:
        con.execute(
            """UPDATE jobs SET status = 'cancelled', finished_at = ?
               WHERE id = ? AND status IN ('pending', 'running')""",
            (_now(), job_id),
        )
