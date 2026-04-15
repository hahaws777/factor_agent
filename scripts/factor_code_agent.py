#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redirect to agent/factor_code_agent.py (run from project root). Kept for backward compatibility."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "agent" / "factor_code_agent.py"
sys.exit(subprocess.call([sys.executable, str(target)] + sys.argv[1:], cwd=str(ROOT)))
