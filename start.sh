#!/usr/bin/env bash
# start.sh — launch the job worker and Streamlit UI in a tmux session.
#
# Usage:
#   bash start.sh              # start (or attach to existing session)
#   bash start.sh stop         # kill the session and all child processes
#
# Requirements: tmux (apt install tmux)
# Python:       conda env "qwen" at C:\Users\PC\Miniconda3\envs\qwen

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_QWEN="/mnt/c/Users/PC/Miniconda3/envs/qwen"
PYTHON="$CONDA_QWEN/python.exe"
STREAMLIT="$CONDA_QWEN/Scripts/streamlit.exe"
SESSION="factor_agent"
STREAMLIT_PORT=8501

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

check_python() {
    [[ -f "$PYTHON" ]] || die "Python not found at $PYTHON — is the conda 'qwen' environment installed?"
    [[ -f "$STREAMLIT" ]] || die "streamlit.exe not found at $STREAMLIT — run: conda run -n qwen pip install streamlit"
    "$PYTHON" -c "import streamlit" 2>/dev/null \
        || die "streamlit not importable — run: conda run -n qwen pip install -r requirements.txt"
}

check_env() {
    local env_file="$ROOT/.env"
    if [[ -f "$env_file" ]]; then
        # shellcheck disable=SC1090
        set -a; source "$env_file"; set +a
    fi
    if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "WARNING: Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set."
        echo "         Set one in $ROOT/.env before generating factors."
    fi
}

# ── Stop ──────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "stop" ]]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
        echo "Stopped session '$SESSION'."
    else
        echo "No session '$SESSION' running."
    fi
    exit 0
fi

# ── Attach if already running ─────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running — attaching."
    exec tmux attach-session -t "$SESSION"
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
check_python
check_env

# ── Create tmux session ───────────────────────────────────────────────────────
#   Window layout:
#     pane 0 (left)  — job worker
#     pane 1 (right) — Streamlit UI

tmux new-session  -d -s "$SESSION" -x 220 -y 50

# Pane 0: job worker
tmux send-keys -t "$SESSION:0" \
    "cd '$ROOT' && echo '=== Job Worker ===' && '$PYTHON' agent/job_worker.py" Enter

# Pane 1: Streamlit (split right)
tmux split-window -t "$SESSION:0" -h
tmux send-keys -t "$SESSION:0.1" \
    "cd '$ROOT' && echo '=== Streamlit UI ===' && '$STREAMLIT' run agent/ui/streamlit_app.py --server.port $STREAMLIT_PORT" Enter

# Sensible pane titles
tmux select-pane -t "$SESSION:0.0" -T "worker"
tmux select-pane -t "$SESSION:0.1" -T "ui"

# Focus the UI pane
tmux select-pane -t "$SESSION:0.1"

echo ""
echo "Started session '$SESSION'."
echo "  Worker  → left pane"
echo "  UI      → http://localhost:$STREAMLIT_PORT"
echo ""
echo "Attaching now. To detach: Ctrl-B then D"
echo "To stop everything: bash start.sh stop"
echo ""

exec tmux attach-session -t "$SESSION"
