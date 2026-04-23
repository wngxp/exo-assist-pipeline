from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

from rl_isaac.artifacts import prepare_run_directory, write_json, write_jsonl
from rl_isaac.cli import build_parser
from rl_isaac.config.loader import load_project_config
from rl_isaac.config.schema import ProjectConfig
from rl_isaac.rollout import heuristic_policy, rollout_episode
from rl_isaac.tasks.registry import make_env


def run(config_path: Path, *, policy_artifact: Path | None = None) -> Path:
    return run_evaluation(load_project_config(config_path), policy_artifact=policy_artifact)


def run_evaluation(config: ProjectConfig, *, policy_artifact: Path | None = None) -> Path:
    _ensure_scaffold_backend(config)
    run_dir = prepare_run_directory(
        config.paths.output_root,
        stage="eval",
        experiment_name=config.experiment.name,
    )
    policy_meta = _load_policy_artifact(policy_artifact)
    policy = partial(heuristic_policy, action_dim=config.task.action.dimension)

    episode_summaries: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    for episode_index in range(config.eval.episodes):
        env = make_env(config)
        frames, summary = rollout_episode(env, steps=config.eval.steps_per_episode, policy=policy)
        episode_summaries.append({"episode_index": episode_index, **summary})
        all_frames.extend(
            {"episode_index": episode_index, **frame.to_dict()}
            for frame in frames
        )

    write_json(run_dir / "run_manifest.json", _manifest(config, policy_meta))
    write_json(run_dir / "resolved_config.json", config.to_dict())
    write_json(
        run_dir / "eval_summary.json",
        {
            "episodes": config.eval.episodes,
            "policy": policy_meta,
            "episode_summaries": episode_summaries,
        },
    )
    write_jsonl(run_dir / "rollout_trace.jsonl", all_frames)
    return run_dir


def main() -> int:
    parser = build_parser(
        "Run the rl_isaac humanoid walking evaluation scaffold.",
        include_policy_artifact=True,
    )
    args = parser.parse_args()
    run_dir = run(args.config, policy_artifact=args.policy_artifact)
    print(f"Evaluation artifacts written to {run_dir}")
    return 0


def _manifest(config: ProjectConfig, policy_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "entrypoint": "eval",
        "experiment_name": config.experiment.name,
        "task": config.experiment.task,
        "backend": config.runtime.backend,
        "mode": config.runtime.mode,
        "policy": policy_meta,
    }


def _load_policy_artifact(policy_artifact: Path | None) -> dict[str, Any]:
    if policy_artifact is None:
        return {"source": "heuristic_velocity_tracking", "path": None}
    with policy_artifact.expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {"source": "artifact", "path": policy_artifact.expanduser().resolve().as_posix(), "payload": payload}


def _ensure_scaffold_backend(config: ProjectConfig) -> None:
    if config.runtime.backend != "scaffold":
        raise NotImplementedError(
            "Only the 'scaffold' backend is implemented today. "
            "Isaac Lab / RSL-RL integration is planned for a later phase."
        )
