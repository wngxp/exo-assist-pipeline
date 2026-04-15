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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


def resolve_vecnormalize_path(override: str | None = None) -> Path:
    candidates = [Path(override)] if override else [RL_OUTPUT_DIR / "walker_policy_stage1_v2_vecnormalize.pkl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find Stage 1 v2 VecNormalize stats. Checked:\n{candidate_text}")


def resolve_model_path(which: str, override: str | None = None) -> Path:
    if override:
        return first_existing_path([Path(override)], label=f"Stage 1 v2 {which} model override")

    if which == "final":
        return first_existing_path(
            [RL_OUTPUT_DIR / "walker_policy_stage1_v2"],
            label="Stage 1 v2 final model",
        )
    if which == "best":
        return first_existing_path(
            [RL_OUTPUT_DIR / "walker_stage1_v2_best" / "best_model"],
            label="Stage 1 v2 best model",
        )
    raise ValueError(f"Unsupported model selector: {which}")


def build_eval_env(vecnormalize_path: Path) -> VecNormalize:
    base_env = DummyVecEnv([make_env])
    env = VecNormalize.load(str(vecnormalize_path), base_env)
    env.training = False
    env.norm_reward = False
    return env


def run_single_eval(
    *,
    label: str,
    model_path: Path,
    vecnormalize_path: Path,
    num_episodes: int,
    save_csv: bool,
) -> dict:
    env = build_eval_env(vecnormalize_path)
    model = PPO.load(str(model_path), env=env, device="cpu")

    episode_lengths: list[int] = []
    current_steps = 0
    obs = env.reset()

    try:
        while len(episode_lengths) < num_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            current_steps += 1
            done = bool(dones[0])
            if not done:
                continue

            info = infos[0]
            episode_summary = info.get("episode", {})
            length = int(episode_summary.get("l", current_steps))
            episode_lengths.append(length)
            print(f"[V2 {label.upper()}] episode={len(episode_lengths)} length={length}")
            current_steps = 0
    finally:
        env.close()

    summary = build_length_summary(
        evaluation_name=f"stage1_v2_{label}",
        model_path=model_path,
        episode_lengths=episode_lengths,
        num_episodes=num_episodes,
        extra={
            "vecnormalize_path": str(vecnormalize_path),
            "model_variant": label,
            "env_training": False,
            "env_norm_reward": False,
        },
    )

    json_path = unique_output_path(f"v2_{label}_eval.json")
    write_json(json_path, summary)
    print(f"Saved JSON summary to: {json_path}")

    if save_csv:
        csv_path = unique_output_path(f"v2_{label}_episode_lengths.csv")
        write_episode_lengths_csv(csv_path, episode_lengths)
        print(f"Saved episode lengths CSV to: {csv_path}")

    print("\nSummary:")
    print(f"  mean episode length: {summary['mean_episode_length']:.2f}")
    print(f"  min episode length: {summary['min_episode_length']}")
    print(f"  max episode length: {summary['max_episode_length']}")
    print(f"  all episode lengths identical: {summary['all_episode_lengths_identical']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 v2 walker checkpoints.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--which",
        choices=["final", "best", "both"],
        default="both",
        help="Choose whether to evaluate the final model, the best checkpoint, or both.",
    )
    parser.add_argument("--final-model-path", type=str, default=None)
    parser.add_argument("--best-model-path", type=str, default=None)
    parser.add_argument("--vecnormalize-path", type=str, default=None)
    parser.add_argument("--no-csv", action="store_true", help="Skip writing the episode lengths CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vecnormalize_path = resolve_vecnormalize_path(args.vecnormalize_path)

    selectors = ["final", "best"] if args.which == "both" else [args.which]
    completed = 0
    for selector in selectors:
        override = args.final_model_path if selector == "final" else args.best_model_path
        try:
            model_path = resolve_model_path(selector, override)
        except FileNotFoundError:
            if args.which != "both":
                raise
            print(f"Skipping missing Stage 1 v2 {selector} checkpoint.")
            continue

        run_single_eval(
            label=selector,
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            num_episodes=args.episodes,
            save_csv=not args.no_csv,
        )
        completed += 1

    if completed == 0:
        raise FileNotFoundError("No Stage 1 v2 checkpoints were available to evaluate.")
