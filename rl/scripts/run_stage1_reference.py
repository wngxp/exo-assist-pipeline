"""Stage 1 DEP-RL reference evaluation.

Purpose:
- load the local DEP-RL `myoLegWalk-v0` baseline
- run fixed evaluation episodes
- save a simple episode-length summary

Inputs:
- `rl/baselines_DEPRL/myoLegWalk_20230514/myoLeg/checkpoints/step_150000000.pt`
- `rl/baselines_DEPRL/myoLegWalk_20230514/myoLeg/config.yaml`

Outputs:
- `rl/rl_output/stage1/reference/stage1_reference.json`
- `rl/rl_output/stage1/reference/stage1_reference_lengths.csv`

Command:
- `python rl/scripts/run_stage1_reference.py`

Status:
- working on the original Linux setup
- not locally verified during this Mac cleanup pass
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import myosuite  # noqa: F401
from myosuite.utils import gym as myogym

import deprl

from rl.baselines.load_deprl_reference import (
    resolve_deprl_reference_paths,
    wrap_deprl_env,
)

ENV_ID = "myoLegWalk-v0"
N_EPISODES = 10
DEFAULT_RESET_TYPE = "random"

RL_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = RL_DIR / "rl_output" / "stage1" / "reference"
BASELINE_DIR = RL_DIR / "baselines_DEPRL" / "myoLegWalk_20230514" / "myoLeg"
CHECKPOINT_PATH = BASELINE_DIR / "checkpoints" / "step_150000000.pt"
CONFIG_PATH = BASELINE_DIR / "config.yaml"


def resolve_local_baseline() -> tuple[Path, Path]:
    if not CHECKPOINT_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Local DEP-RL baseline files not found. Expected:\n"
            f"  checkpoint: {CHECKPOINT_PATH.resolve()}\n"
            f"  config: {CONFIG_PATH.resolve()}\n"
            "This script only loads local files and does not download baselines."
        )
    return CHECKPOINT_PATH, CONFIG_PATH


def make_stage1_env(reset_type: str = DEFAULT_RESET_TYPE):
    base_env = myogym.make(ENV_ID, reset_type=reset_type)
    env = wrap_deprl_env(base_env)
    return base_env, env


def load_reference_policy(env, baseline_dir: str | Path | None = None):
    if baseline_dir is None:
        checkpoint_path, config_path = resolve_local_baseline()
        baseline_root = BASELINE_DIR.resolve()
    else:
        baseline_root, checkpoint_path, config_path = resolve_deprl_reference_paths(
            baseline_dir
        )

    print(f"Loading DEP-RL baseline for {ENV_ID} from local files ...")
    print(f"  checkpoint: {checkpoint_path.resolve()}")
    print(f"  config: {config_path.resolve()}")

    os.environ["DEPRL_BASELINE_PATH"] = str(baseline_root)
    return deprl.load_baseline(env)


def _normalize_reset_output(reset_out):
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def _normalize_step_output(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), info

    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, info

    raise ValueError(f"Unexpected step result length: {len(result)}")


def run_reference_episode(
    policy,
    env,
    *,
    capture_frames: bool = False,
    render_frame=None,
    max_steps: int | None = None,
) -> dict:
    if capture_frames and render_frame is None:
        raise ValueError("render_frame must be provided when capture_frames=True")

    reset_out = env.reset()
    obs = _normalize_reset_output(reset_out)

    frames = []
    if capture_frames:
        frames.append(render_frame())

    steps = 0
    terminated = False
    truncated = False
    info = {}

    while max_steps is None or steps < max_steps:
        action = policy(obs)
        obs, reward, terminated, truncated, info = _normalize_step_output(env.step(action))
        del reward

        steps += 1
        if capture_frames:
            frames.append(render_frame())

        if terminated or truncated:
            break

    reached_max_steps = bool(
        max_steps is not None and steps >= max_steps and not (terminated or truncated)
    )
    terminated_early = bool(
        max_steps is not None and (terminated or truncated) and steps < max_steps
    )

    return {
        "episode_length": steps,
        "terminated": terminated,
        "truncated": truncated,
        "terminated_early": terminated_early,
        "reached_max_steps": reached_max_steps,
        "frames": frames,
        "info": info,
    }


def evaluate_reference(
    policy,
    env,
    *,
    n_episodes: int = N_EPISODES,
    capture_frames: bool = False,
    render_frame=None,
    max_steps: int | None = None,
) -> list[dict]:
    episodes: list[dict] = []

    for episode_idx in range(1, n_episodes + 1):
        episode = run_reference_episode(
            policy,
            env,
            capture_frames=capture_frames,
            render_frame=render_frame,
            max_steps=max_steps,
        )
        episodes.append(episode)
        print(f"[REFERENCE] episode={episode_idx} length={episode['episode_length']}")

    return episodes


def save_results(
    lengths: list[int],
    *,
    checkpoint_path: Path = CHECKPOINT_PATH,
    config_path: Path = CONFIG_PATH,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "deprl_baseline",
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "env_id": ENV_ID,
        "n_episodes": len(lengths),
        "episode_lengths": lengths,
        "mean_episode_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_episode_length": min(lengths) if lengths else 0,
        "max_episode_length": max(lengths) if lengths else 0,
        "all_lengths_identical": len(set(lengths)) == 1 if lengths else False,
    }

    json_path = RESULTS_DIR / "stage1_reference.json"
    csv_path = RESULTS_DIR / "stage1_reference_lengths.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "length"])
        for episode, length in enumerate(lengths, start=1):
            writer.writerow([episode, length])

    print(f"Saved JSON summary to: {json_path.resolve()}")
    print(f"Saved episode lengths CSV to: {csv_path.resolve()}")


def main() -> None:
    _, env = make_stage1_env()

    try:
        checkpoint_path, config_path = resolve_local_baseline()
        policy = load_reference_policy(env)

        episodes = evaluate_reference(policy, env)
        lengths = [episode["episode_length"] for episode in episodes]
        save_results(lengths, checkpoint_path=checkpoint_path, config_path=config_path)

        print("\nSummary:")
        print(f"  mean episode length: {sum(lengths) / len(lengths):.2f}")
        print(f"  min episode length: {min(lengths)}")
        print(f"  max episode length: {max(lengths)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
