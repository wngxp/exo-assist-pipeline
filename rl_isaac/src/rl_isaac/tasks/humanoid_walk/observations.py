from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from rl_isaac.tasks.humanoid_walk.actions import action_stats
from rl_isaac.tasks.humanoid_walk.config import HumanoidWalkTaskConfig

if TYPE_CHECKING:
    from rl_isaac.tasks.humanoid_walk.env import HumanoidWalkState


def build_observation(
    *,
    state: "HumanoidWalkState",
    config: HumanoidWalkTaskConfig,
    previous_action: Sequence[float],
    gait_phase_features: Mapping[str, float] | None = None,
    reference_features: Mapping[str, float] | None = None,
) -> dict[str, float]:
    stats = action_stats(previous_action)
    available_features = {
        "base_height": state.base_height,
        "forward_velocity": state.forward_velocity,
        "lateral_velocity": state.lateral_velocity,
        "yaw_rate": state.yaw_rate,
        "torso_upright": state.torso_upright,
        "foot_clearance": state.foot_clearance,
        "target_velocity_error": config.reward.target_forward_velocity - state.forward_velocity,
        "previous_action_mean": stats["action_mean"],
        "previous_action_abs_mean": stats["action_abs_mean"],
    }

    observation: dict[str, float] = {}
    for name in config.observation.features:
        try:
            observation[name] = float(available_features[name])
        except KeyError as exc:
            raise KeyError(f"Unsupported observation feature '{name}'.") from exc

    if config.observation.include_gait_phase and gait_phase_features:
        observation.update({key: float(value) for key, value in gait_phase_features.items()})
    if config.observation.include_reference_tracking and reference_features:
        observation.update({key: float(value) for key, value in reference_features.items()})
    return observation
