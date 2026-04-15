from __future__ import annotations

import os
from pathlib import Path

import numpy as np

RL_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = RL_DIR / "baselines_DEPRL" / "myoLegWalk_20230514" / "myoLeg"
CHECKPOINT_PATH = BASELINE_DIR / "checkpoints" / "step_150000000.pt"
CONFIG_PATH = BASELINE_DIR / "config.yaml"


class FrozenDeprlWalker:
    """Small compatibility shim so Stage 2 can keep calling `.predict(...)`."""

    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs, deterministic=True):
        del deterministic
        action = self.policy(obs)
        return np.asarray(action), None


def wrap_deprl_env(base_env):
    try:
        from deprl import env_wrappers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "deprl is required to wrap the Stage 2 environment for the local DEP-RL reference walker."
        ) from exc

    return env_wrappers.GymWrapper(base_env)


def _resolve_baseline_dir(baseline_dir=None) -> Path:
    if baseline_dir is None:
        return BASELINE_DIR.resolve()

    candidate = Path(baseline_dir)
    if not candidate.is_absolute():
        candidate = (RL_DIR / candidate).resolve()
    return candidate


def resolve_deprl_reference_paths(baseline_dir=None) -> tuple[Path, Path, Path]:
    baseline_root = _resolve_baseline_dir(baseline_dir)
    checkpoint_path = baseline_root / "checkpoints" / "step_150000000.pt"
    config_path = baseline_root / "config.yaml"

    if not checkpoint_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Local DEP-RL baseline files not found. Expected:\n"
            f"  checkpoint: {checkpoint_path.resolve()}\n"
            f"  config: {config_path.resolve()}\n"
            "Stage 2 only loads the local DEP-RL reference walker and does not download baselines."
        )

    return baseline_root, checkpoint_path, config_path


def load_deprl_reference(env, baseline_dir=None):
    try:
        import deprl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "deprl is required to load the local Stage 1 reference walker for Stage 2."
        ) from exc

    baseline_root, checkpoint_path, config_path = resolve_deprl_reference_paths(
        baseline_dir
    )

    os.environ["DEPRL_BASELINE_PATH"] = str(baseline_root)
    policy = deprl.load_baseline(env)

    print("  Using local DEP-RL reference walker")
    print(f"    checkpoint: {checkpoint_path}")
    print(f"    config: {config_path}")

    return FrozenDeprlWalker(policy)
