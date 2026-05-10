#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Job worker — polls agent_runs/jobs.db and executes pending jobs as subprocesses.

The Streamlit UI submits jobs to the queue; this worker must be running
separately to process them.  One failed job never crashes the worker.

Usage
-----
  # From the project root:
  python agent/job_worker.py

  # Custom poll interval (seconds between queue checks when idle):
  python agent/job_worker.py --poll-interval 5

Stop with Ctrl-C or SIGTERM.  The current job finishes before the worker exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT / "agent_runs" / "job_logs"

sys.path.insert(0, str(AGENT_DIR))
import job_queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("job_worker")

_shutdown = threading.Event()


def _handle_signal(signum, _frame):
    log.info("Signal %d received — finishing current job then stopping.", signum)
    _shutdown.set()


def _run_job(job: dict) -> tuple[int, str]:
    """
    Execute a job's command as a subprocess, streaming output to a log file.

    Returns (returncode, error_message).  returncode 0 means success.
    The log file path is written back to the DB as soon as the process starts.
    """
    try:
        params = json.loads(job.get("params_json") or "{}")
    except Exception as exc:
        return -1, f"bad params_json: {exc}"

    cmd = params.get("cmd")
    if not cmd:
        return -1, "params.cmd is missing"

    cwd = params.get("cwd", str(ROOT))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['id']}.log"
    job_queue.update_status(job["id"], "running", log_path=str(log_path))

    log.info(
        "Job %d [%s] run_id=%s → %s",
        job["id"], job["job_type"],
        job.get("run_id") or "",
        " ".join(str(x) for x in cmd),
    )

    try:
        with open(log_path, "a", encoding="utf-8", buffering=1) as lf:
            lf.write(f"[worker] job_id={job['id']} type={job['job_type']}\n")
            lf.write(f"[worker] cmd: {' '.join(str(x) for x in cmd)}\n\n")
            proc = subprocess.Popen(
                [str(x) for x in cmd],
                cwd=cwd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        # Store pid so the UI can cancel if needed
        job_queue.update_status(job["id"], "running", pid=proc.pid)
        rc = proc.wait()
    except Exception as exc:
        return -1, str(exc)

    return rc, ""


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    parser = argparse.ArgumentParser(
        description="Factor Agent job worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--poll-interval", type=float, default=2.0, metavar="SEC",
        help="Seconds to sleep between queue polls when no job is pending",
    )
    args = parser.parse_args()

    job_queue.init_db()
    log.info("Worker ready. DB: %s", job_queue.DB_PATH)
    log.info("Logs:          %s", LOG_DIR)
    log.info("Poll interval: %.0fs.  Stop with Ctrl-C.", args.poll_interval)

    while not _shutdown.is_set():
        # ── Claim one pending job ─────────────────────────────────────────
        try:
            job = job_queue.claim_next()
        except Exception as exc:
            log.error("Queue error: %s — retrying in %.0fs", exc, args.poll_interval)
            _shutdown.wait(args.poll_interval)
            continue

        if job is None:
            _shutdown.wait(args.poll_interval)
            continue

        # ── Run it ───────────────────────────────────────────────────────
        try:
            rc, err = _run_job(job)
        except Exception as exc:
            log.exception("Unexpected error in job %d", job["id"])
            rc, err = -1, str(exc)

        # ── Record outcome ────────────────────────────────────────────────
        latest = job_queue.get_job(job["id"])
        if latest and latest.get("status") == "cancelled":
            log.info("Job %d was cancelled; preserving cancelled status.", job["id"])
            continue

        if rc == 0:
            job_queue.update_status(job["id"], "success")
            log.info("Job %d finished OK.", job["id"])
        else:
            job_queue.update_status(
                job["id"], "failed",
                error=err or f"subprocess exit code {rc}",
            )
            log.warning("Job %d failed (rc=%d): %s", job["id"], rc, err)


if __name__ == "__main__":
    main()
