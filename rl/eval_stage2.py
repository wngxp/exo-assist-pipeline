#!/usr/bin/env python3
"""Deprecated wrapper for the modular Stage 2 evaluation script."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.scripts.eval_exo_stage2 import OUT_DIR, run_eval


if __name__ == "__main__":
    run_eval(f"{OUT_DIR}/walker_policy", f"{OUT_DIR}/exo_policy")
