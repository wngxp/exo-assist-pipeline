from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping

from rl_isaac.artifacts import project_root
from rl_isaac.config.schema import (
    EvalConfig,
    ExperimentConfig,
    PathsConfig,
    ProjectConfig,
    RenderConfig,
    RuntimeConfig,
    TrainConfig,
)
from rl_isaac.tasks.humanoid_walk.config import build_humanoid_walk_task_config


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    experiment_raw = _mapping(raw, "experiment")
    runtime_raw = _mapping(raw, "runtime")
    paths_raw = _mapping(raw, "paths")
    train_raw = _mapping(raw, "train")
    eval_raw = _mapping(raw, "eval")
    render_raw = _mapping(raw, "render")
    task_raw = _mapping(raw, "task")

    task_name = str(experiment_raw.get("task", task_raw.get("name", "humanoid_walk")))
    if task_name != "humanoid_walk":
        raise ValueError(f"Unsupported task '{task_name}'. Only 'humanoid_walk' is scaffolded today.")

    project_dir = project_root()
    return ProjectConfig(
        experiment=ExperimentConfig(
            name=str(experiment_raw.get("name", "humanoid_walk_baseline")),
            task=task_name,
            seed=int(experiment_raw.get("seed", 0)),
            tags=tuple(str(value) for value in experiment_raw.get("tags", [])),
        ),
        runtime=RuntimeConfig(
            backend=str(runtime_raw.get("backend", "scaffold")),
            mode=str(runtime_raw.get("mode", "dry_run")),
            device=str(runtime_raw.get("device", "cpu")),
            headless=bool(runtime_raw.get("headless", True)),
            dt=float(runtime_raw.get("dt", 0.02)),
        ),
        paths=PathsConfig(
            project_root=project_dir,
            config_path=config_path,
            data_root=_resolve_path(paths_raw.get("data_root", "data"), project_dir),
            output_root=_resolve_path(paths_raw.get("output_root", "outputs"), project_dir),
        ),
        train=TrainConfig(
            iterations=int(train_raw.get("iterations", 1)),
            rollout_steps=int(train_raw.get("rollout_steps", 128)),
            checkpoint_interval=int(train_raw.get("checkpoint_interval", 1)),
        ),
        eval=EvalConfig(
            episodes=int(eval_raw.get("episodes", 1)),
            steps_per_episode=int(eval_raw.get("steps_per_episode", 128)),
        ),
        render=RenderConfig(
            frames=int(render_raw.get("frames", 64)),
            camera=str(render_raw.get("camera", "follow")),
        ),
        task=build_humanoid_walk_task_config(task_raw),
    )


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected TOML table '{key}' to be a mapping.")
    return value


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
