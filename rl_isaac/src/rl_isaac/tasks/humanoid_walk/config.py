from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionConfig:
    dimension: int
    scale: float
    clamp: float
    joint_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "scale": self.scale,
            "clamp": self.clamp,
            "joints": list(self.joint_names),
        }


@dataclass(frozen=True)
class ObservationConfig:
    features: tuple[str, ...]
    include_gait_phase: bool
    include_reference_tracking: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "include_gait_phase": self.include_gait_phase,
            "include_reference_tracking": self.include_reference_tracking,
        }


@dataclass(frozen=True)
class RewardConfig:
    weights: dict[str, float]
    target_forward_velocity: float
    target_foot_clearance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_forward_velocity": self.target_forward_velocity,
            "target_foot_clearance": self.target_foot_clearance,
            "weights": dict(self.weights),
        }


@dataclass(frozen=True)
class TerminationConfig:
    max_steps: int
    min_base_height: float
    min_upright: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "min_base_height": self.min_base_height,
            "min_upright": self.min_upright,
        }


@dataclass(frozen=True)
class FutureFeaturesConfig:
    enable_mocap_reference: bool
    enable_gait_phase_features: bool
    enable_residual_exo: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_mocap_reference": self.enable_mocap_reference,
            "enable_gait_phase_features": self.enable_gait_phase_features,
            "enable_residual_exo": self.enable_residual_exo,
        }


@dataclass(frozen=True)
class HumanoidWalkTaskConfig:
    name: str
    nominal_base_height: float
    step_frequency_hz: float
    action: ActionConfig
    observation: ObservationConfig
    reward: RewardConfig
    termination: TerminationConfig
    future: FutureFeaturesConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nominal_base_height": self.nominal_base_height,
            "step_frequency_hz": self.step_frequency_hz,
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict(),
            "reward": self.reward.to_dict(),
            "termination": self.termination.to_dict(),
            "future": self.future.to_dict(),
        }


def build_humanoid_walk_task_config(raw: Mapping[str, Any]) -> HumanoidWalkTaskConfig:
    action_raw = _mapping(raw, "action")
    observation_raw = _mapping(raw, "observation")
    reward_raw = _mapping(raw, "reward")
    reward_weights_raw = _mapping(reward_raw, "weights")
    termination_raw = _mapping(raw, "termination")
    future_raw = _mapping(raw, "future")

    joint_names = tuple(str(name) for name in action_raw.get("joints", ()))
    dimension = int(action_raw.get("dimension", len(joint_names)))
    if dimension <= 0:
        raise ValueError("Action dimension must be positive.")
    if joint_names and len(joint_names) != dimension:
        raise ValueError("Action joint list length must match the configured action dimension.")

    features = tuple(str(feature) for feature in observation_raw.get("features", ()))
    if not features:
        raise ValueError("Observation feature list must not be empty.")

    max_steps = int(termination_raw.get("max_steps", 256))
    if max_steps <= 0:
        raise ValueError("Termination max_steps must be positive.")

    return HumanoidWalkTaskConfig(
        name=str(raw.get("name", "humanoid_walk")),
        nominal_base_height=float(raw.get("nominal_base_height", 0.92)),
        step_frequency_hz=float(raw.get("step_frequency_hz", 1.8)),
        action=ActionConfig(
            dimension=dimension,
            scale=float(action_raw.get("scale", 0.25)),
            clamp=float(action_raw.get("clamp", 1.0)),
            joint_names=joint_names,
        ),
        observation=ObservationConfig(
            features=features,
            include_gait_phase=bool(observation_raw.get("include_gait_phase", False)),
            include_reference_tracking=bool(
                observation_raw.get("include_reference_tracking", False)
            ),
        ),
        reward=RewardConfig(
            target_forward_velocity=float(reward_raw.get("target_forward_velocity", 1.25)),
            target_foot_clearance=float(reward_raw.get("target_foot_clearance", 0.08)),
            weights={key: float(value) for key, value in reward_weights_raw.items()},
        ),
        termination=TerminationConfig(
            max_steps=max_steps,
            min_base_height=float(termination_raw.get("min_base_height", 0.72)),
            min_upright=float(termination_raw.get("min_upright", 0.6)),
        ),
        future=FutureFeaturesConfig(
            enable_mocap_reference=bool(future_raw.get("enable_mocap_reference", False)),
            enable_gait_phase_features=bool(future_raw.get("enable_gait_phase_features", False)),
            enable_residual_exo=bool(future_raw.get("enable_residual_exo", False)),
        ),
    )


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected task table '{key}' to be a mapping.")
    return value
