from __future__ import annotations

from collections.abc import Callable

from rl_isaac.assistance.residual import NullResidualExoAssistance
from rl_isaac.config.schema import ProjectConfig
from rl_isaac.features.gait_phase import SinusoidalGaitPhaseFeatureProvider
from rl_isaac.tasks.humanoid_walk.env import HumanoidWalkEnv

TaskFactory = Callable[[ProjectConfig], HumanoidWalkEnv]


def build_humanoid_walk_env(config: ProjectConfig) -> HumanoidWalkEnv:
    if config.task.future.enable_mocap_reference:
        raise NotImplementedError("Mocap reference loading is not wired into the scaffold backend yet.")
    if config.task.future.enable_residual_exo:
        raise NotImplementedError("Residual exoskeleton assistance is planned but not implemented yet.")

    gait_phase_provider = None
    if config.task.future.enable_gait_phase_features:
        gait_phase_provider = SinusoidalGaitPhaseFeatureProvider(config.task.step_frequency_hz)

    return HumanoidWalkEnv(
        config=config.task,
        dt=config.runtime.dt,
        gait_phase_provider=gait_phase_provider,
        residual_assistance=NullResidualExoAssistance(),
    )


TASK_REGISTRY: dict[str, TaskFactory] = {
    "humanoid_walk": build_humanoid_walk_env,
}


def make_env(config: ProjectConfig) -> HumanoidWalkEnv:
    try:
        factory = TASK_REGISTRY[config.experiment.task]
    except KeyError as exc:
        raise ValueError(f"Task '{config.experiment.task}' is not registered.") from exc
    return factory(config)
