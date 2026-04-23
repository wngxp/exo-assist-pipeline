from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_isaac.tasks.humanoid_walk.config import HumanoidWalkTaskConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    task: str
    seed: int
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    backend: str
    mode: str
    device: str
    headless: bool
    dt: float


@dataclass(frozen=True, slots=True)
class PathsConfig:
    project_root: Path
    config_path: Path
    data_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class TrainConfig:
    iterations: int
    rollout_steps: int
    checkpoint_interval: int


@dataclass(frozen=True, slots=True)
class EvalConfig:
    episodes: int
    steps_per_episode: int


@dataclass(frozen=True, slots=True)
class RenderConfig:
    frames: int
    camera: str


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    experiment: ExperimentConfig
    runtime: RuntimeConfig
    paths: PathsConfig
    train: TrainConfig
    eval: EvalConfig
    render: RenderConfig
    task: HumanoidWalkTaskConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {
                "name": self.experiment.name,
                "task": self.experiment.task,
                "seed": self.experiment.seed,
                "tags": list(self.experiment.tags),
            },
            "runtime": {
                "backend": self.runtime.backend,
                "mode": self.runtime.mode,
                "device": self.runtime.device,
                "headless": self.runtime.headless,
                "dt": self.runtime.dt,
            },
            "paths": {
                "project_root": self.paths.project_root.as_posix(),
                "config_path": self.paths.config_path.as_posix(),
                "data_root": self.paths.data_root.as_posix(),
                "output_root": self.paths.output_root.as_posix(),
            },
            "train": {
                "iterations": self.train.iterations,
                "rollout_steps": self.train.rollout_steps,
                "checkpoint_interval": self.train.checkpoint_interval,
            },
            "eval": {
                "episodes": self.eval.episodes,
                "steps_per_episode": self.eval.steps_per_episode,
            },
            "render": {
                "frames": self.render.frames,
                "camera": self.render.camera,
            },
            "task": self.task.to_dict(),
        }
