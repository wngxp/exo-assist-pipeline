from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import myosuite  # noqa: F401  # required for env registration
from myosuite.utils import gym as myogym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rl.stage1_compare.common import (
    RL_OUTPUT_DIR,
    build_length_summary,
    first_existing_path,
    unique_output_path,
    write_episode_lengths_csv,
    write_json,
)

ENV_ID = "myoLegWalk-v0"
DEFAULT_EPISODES = 10


def make_env() -> Monitor:
    return Monitor(myogym.make(ENV_ID))


def resolve_simple_model_path(override: str | None = None) -> Path:
    if override:
        return first_existing_path([Path(override)], label="simple walker checkpoint override")

    return first_existing_path(
        [
            RL_OUTPUT_DIR / "walker_policy_1p5M",
            RL_OUTPUT_DIR / "walker_policy",
        ],
        label="simple PPO walker checkpoint",
    )


def run_eval(model_path: Path, num_episodes: int = DEFAULT_EPISODES, save_csv: bool = True) -> dict:
    env = make_env()
    model = PPO.load(str(model_path), env=env, device="cpu")

    episode_lengths: list[int] = []

    try:
        for episode_idx in range(1, num_episodes + 1):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = result

                steps += 1

            episode_lengths.append(steps)
            print(f"[SIMPLE] episode={episode_idx} length={steps}")
    finally:
        env.close()

    summary = build_length_summary(
        evaluation_name="simple_ppo_walker",
        model_path=model_path,
        episode_lengths=episode_lengths,
        num_episodes=num_episodes,
    )

    json_path = unique_output_path("simple_1p5M_eval.json")
    write_json(json_path, summary)
    print(f"Saved JSON summary to: {json_path}")

    if save_csv:
        csv_path = unique_output_path("simple_1p5M_episode_lengths.csv")
        write_episode_lengths_csv(csv_path, episode_lengths)
        print(f"Saved episode lengths CSV to: {csv_path}")

    print("\nSummary:")
    print(f"  mean episode length: {summary['mean_episode_length']:.2f}")
    print(f"  min episode length: {summary['min_episode_length']}")
    print(f"  max episode length: {summary['max_episode_length']}")
    print(f"  all episode lengths identical: {summary['all_episode_lengths_identical']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the simple PPO walker checkpoint.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--no-csv", action="store_true", help="Skip writing the episode lengths CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_model_path = resolve_simple_model_path(args.model_path)
    run_eval(resolved_model_path, num_episodes=args.episodes, save_csv=not args.no_csv)
