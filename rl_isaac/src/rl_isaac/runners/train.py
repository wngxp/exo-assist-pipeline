from __future__ import annotations

from functools import partial
from pathlib import Path

from rl_isaac.artifacts import prepare_run_directory, write_json, write_jsonl
from rl_isaac.cli import build_parser
from rl_isaac.config.loader import load_project_config
from rl_isaac.config.schema import ProjectConfig
from rl_isaac.rollout import heuristic_policy, rollout_episode
from rl_isaac.tasks.registry import make_env


def run(config_path: Path) -> Path:
    return run_training(load_project_config(config_path))


def run_training(config: ProjectConfig) -> Path:
    _ensure_scaffold_backend(config)
    env = make_env(config)
    run_dir = prepare_run_directory(
        config.paths.output_root,
        stage="train",
        experiment_name=config.experiment.name,
    )
    policy = partial(heuristic_policy, action_dim=config.task.action.dimension)
    frames, summary = rollout_episode(env, steps=config.train.rollout_steps, policy=policy)
    write_json(run_dir / "run_manifest.json", _manifest(config, entrypoint="train"))
    write_json(run_dir / "resolved_config.json", config.to_dict())
    write_json(
        run_dir / "policy_stub.json",
        {
            "policy_name": "heuristic_velocity_tracking",
            "notes": "Placeholder policy artifact until RSL-RL integration is added.",
            "action_dimension": config.task.action.dimension,
        },
    )
    write_json(run_dir / "train_summary.json", summary)
    write_jsonl(run_dir / "rollout_trace.jsonl", (frame.to_dict() for frame in frames))
    return run_dir


def main() -> int:
    parser = build_parser("Run the rl_isaac humanoid walking training scaffold.")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"Training artifacts written to {run_dir}")
    return 0


def _manifest(config: ProjectConfig, *, entrypoint: str) -> dict[str, str]:
    return {
        "entrypoint": entrypoint,
        "experiment_name": config.experiment.name,
        "task": config.experiment.task,
        "backend": config.runtime.backend,
        "mode": config.runtime.mode,
    }


def _ensure_scaffold_backend(config: ProjectConfig) -> None:
    if config.runtime.backend != "scaffold":
        raise NotImplementedError(
            "Only the 'scaffold' backend is implemented today. "
            "Isaac Lab / RSL-RL integration is planned for a later phase."
        )
