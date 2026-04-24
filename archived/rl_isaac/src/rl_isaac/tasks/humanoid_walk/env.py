from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rl_isaac.assistance.residual import NullResidualExoAssistance, ResidualExoAssistance
from rl_isaac.features.gait_phase import GaitPhaseFeatureProvider
from rl_isaac.tasks.humanoid_walk.actions import resolve_action
from rl_isaac.tasks.humanoid_walk.config import HumanoidWalkTaskConfig
from rl_isaac.tasks.humanoid_walk.observations import build_observation
from rl_isaac.tasks.humanoid_walk.rewards import compute_reward
from rl_isaac.tasks.humanoid_walk.termination import check_termination


@dataclass
class HumanoidWalkState:
    step_index: int = 0
    time_s: float = 0.0
    base_height: float = 0.92
    forward_velocity: float = 0.0
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    torso_upright: float = 1.0
    foot_clearance: float = 0.05

    def to_dict(self) -> dict[str, int | float]:
        return {
            "step_index": self.step_index,
            "time_s": self.time_s,
            "base_height": self.base_height,
            "forward_velocity": self.forward_velocity,
            "lateral_velocity": self.lateral_velocity,
            "yaw_rate": self.yaw_rate,
            "torso_upright": self.torso_upright,
            "foot_clearance": self.foot_clearance,
        }


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, float]
    reward: float
    terminated: bool
    termination_reason: str | None
    state: dict[str, int | float]
    info: dict[str, Any]


@dataclass
class HumanoidWalkEnv:
    config: HumanoidWalkTaskConfig
    dt: float
    gait_phase_provider: GaitPhaseFeatureProvider | None = None
    residual_assistance: ResidualExoAssistance = field(default_factory=NullResidualExoAssistance)
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    state: HumanoidWalkState = field(init=False)
    _previous_action: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state = self._initial_state()
        self._previous_action = [0.0] * self.config.action.dimension

    def reset(self, seed: int | None = None) -> dict[str, float]:
        if seed is not None:
            self._rng.seed(seed)
        self.state = self._initial_state()
        self.state.forward_velocity = self._rng.uniform(-0.05, 0.05)
        self._previous_action = [0.0] * self.config.action.dimension
        return self._build_observation()

    def step(self, action: Sequence[float]) -> StepResult:
        applied_action = resolve_action(
            action,
            config=self.config.action,
            state=self.state.to_dict(),
            residual_assistance=self.residual_assistance,
        )
        self._advance_state(applied_action)
        reward = compute_reward(state=self.state, action=applied_action, config=self.config.reward)
        termination = check_termination(state=self.state, config=self.config.termination)
        self._previous_action = applied_action
        observation = self._build_observation()
        info = {
            "reward_breakdown": reward.to_dict(),
            "applied_action": applied_action,
        }
        return StepResult(
            observation=observation,
            reward=reward.total,
            terminated=termination.terminated,
            termination_reason=termination.reason,
            state=self.state.to_dict(),
            info=info,
        )

    def _advance_state(self, action: Sequence[float]) -> None:
        action_mean = sum(action) / float(len(action)) if action else 0.0
        action_abs_mean = sum(abs(value) for value in action) / float(len(action)) if action else 0.0
        target_error = self.config.reward.target_forward_velocity - self.state.forward_velocity

        self.state.step_index += 1
        self.state.time_s += self.dt
        self.state.forward_velocity += 0.08 * target_error + 0.04 * action_mean
        self.state.lateral_velocity = 0.7 * self.state.lateral_velocity + 0.02 * action_mean
        self.state.yaw_rate = 0.1 * (action[0] - action[6]) if len(action) >= 7 else 0.0
        self.state.base_height = max(
            0.0,
            self.config.nominal_base_height - 0.03 * action_abs_mean,
        )
        self.state.torso_upright = max(
            0.0,
            1.0 - 0.25 * abs(self.state.yaw_rate) - 0.10 * action_abs_mean,
        )

        phase = (self.state.time_s * self.config.step_frequency_hz) % 1.0
        self.state.foot_clearance = 0.05 + 0.03 * max(0.0, math.sin(2.0 * math.pi * phase))

    def _build_observation(self) -> dict[str, float]:
        gait_phase_features = None
        if self.config.observation.include_gait_phase and self.gait_phase_provider is not None:
            gait_phase_features = self.gait_phase_provider.compute(
                step_index=self.state.step_index,
                time_s=self.state.time_s,
            ).to_dict()
        return build_observation(
            state=self.state,
            config=self.config,
            previous_action=self._previous_action,
            gait_phase_features=gait_phase_features,
            reference_features=None,
        )

    def _initial_state(self) -> HumanoidWalkState:
        return HumanoidWalkState(
            step_index=0,
            time_s=0.0,
            base_height=self.config.nominal_base_height,
            forward_velocity=0.0,
            lateral_velocity=0.0,
            yaw_rate=0.0,
            torso_upright=1.0,
            foot_clearance=0.05,
        )
