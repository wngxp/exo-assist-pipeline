from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import myosuite  # noqa: F401
from myosuite.utils import gym as myogym

import deprl
from deprl import env_wrappers

RL_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = RL_DIR / "rl_output" / "stage1" / "compare"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENV_ID = "myoLegWalk-v0"
N_EPISODES = 10

BASELINE_DIR = RL_DIR / "baselines_DEPRL" / "myoLegWalk_20230514" / "myoLeg"
CHECKPOINT_PATH = BASELINE_DIR / "checkpoints" / "step_150000000.pt"
CONFIG_PATH = BASELINE_DIR / "config.yaml"


def resolve_local_baseline() -> tuple[Path, Path]:
    """Resolve the local DEP-RL baseline files.

    This script is intended to run offline after the baseline has been
    downloaded once. We do not want to rely on Google Drive every run.
    """
    if not CHECKPOINT_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Local DEP-RL baseline files not found. Expected:\n"
            f"  checkpoint: {CHECKPOINT_PATH.resolve()}\n"
            f"  config: {CONFIG_PATH.resolve()}\n"
            "Download/cache the DEP-RL baseline once, then rerun this script."
        )
    return CHECKPOINT_PATH, CONFIG_PATH


def save_results(lengths: list[int]) -> None:
    summary = {
        "model": "deprl_baseline",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "config_path": str(CONFIG_PATH),
        "env_id": ENV_ID,
        "n_episodes": len(lengths),
        "episode_lengths": lengths,
        "mean_episode_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_episode_length": min(lengths) if lengths else 0,
        "max_episode_length": max(lengths) if lengths else 0,
        "all_lengths_identical": len(set(lengths)) == 1 if lengths else False,
    }

    json_path = RESULTS_DIR / "deprl_eval.json"
    csv_path = RESULTS_DIR / "deprl_episode_lengths.csv"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "length"])
        for i, length in enumerate(lengths, start=1):
            writer.writerow([i, length])

    print(f"Saved JSON summary to: {json_path.resolve()}")
    print(f"Saved episode lengths CSV to: {csv_path.resolve()}")


def main() -> None:
    env = myogym.make(ENV_ID, reset_type="random")
    env = env_wrappers.GymWrapper(env)

    checkpoint_path, config_path = resolve_local_baseline()
    print(f"Loading DEP-RL baseline for {ENV_ID} from local files ...")
    print(f"  checkpoint: {checkpoint_path.resolve()}")
    print(f"  config: {config_path.resolve()}")

    # Point deprl at the already-downloaded local baseline so evaluation is
    # reproducible and does not depend on Google Drive access on every run.
    os.environ["DEPRL_BASELINE_PATH"] = str(BASELINE_DIR.resolve())
    policy = deprl.load_baseline(env)

    lengths: list[int] = []

    for ep in range(1, N_EPISODES + 1):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        steps = 0

        while not done:
            action = policy(obs)
            result = env.step(action)

            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            steps += 1

        lengths.append(steps)
        print(f"[DEPRL] episode={ep} length={steps}")

    save_results(lengths)

    print("\nSummary:")
    print(f"  mean episode length: {sum(lengths)/len(lengths):.2f}")
    print(f"  min episode length: {min(lengths)}")
    print(f"  max episode length: {max(lengths)}")
    print(f"  all episode lengths identical: {len(set(lengths)) == 1}")

    env.close()


if __name__ == "__main__":
    main()
