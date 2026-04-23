from __future__ import annotations

import argparse
from pathlib import Path

from rl_isaac.artifacts import project_root


def default_config_path() -> Path:
    return project_root() / "configs" / "experiments" / "humanoid_walk.toml"


def build_parser(description: str, *, include_policy_artifact: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to a TOML experiment config.",
    )
    if include_policy_artifact:
        parser.add_argument(
            "--policy-artifact",
            type=Path,
            default=None,
            help="Optional policy artifact or run metadata to attach to the run manifest.",
        )
    return parser
