from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from rl_isaac.tasks.humanoid_walk.env import HumanoidWalkEnv

PolicyFn = Callable[[Mapping[str, float], int], Sequence[float]]


@dataclass(frozen=True, slots=True)
class RolloutFrame:
    step_index: int
    action: list[float]
    reward: float
    terminated: bool
    termination_reason: str | None
    observation: dict[str, float]
    state: dict[str, int | float]
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "reward": self.reward,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "observation": self.observation,
            "state": self.state,
            "info": self.info,
        }


def heuristic_policy(
    observation: Mapping[str, float],
    step_index: int,
    action_dim: int,
) -> list[float]:
    del step_index
    velocity_error = float(observation.get("target_velocity_error", 0.0))
    command = max(-1.0, min(1.0, velocity_error))
    return [command] * action_dim


def rollout_episode(
    env: HumanoidWalkEnv,
    *,
    steps: int,
    policy: PolicyFn,
) -> tuple[list[RolloutFrame], dict[str, Any]]:
    observation = env.reset()
    frames: list[RolloutFrame] = []
    total_reward = 0.0
    termination_reason: str | None = None

    for step_index in range(steps):
        action = [float(value) for value in policy(observation, step_index)]
        result = env.step(action)
        total_reward += result.reward
        termination_reason = result.termination_reason
        frames.append(
            RolloutFrame(
                step_index=step_index,
                action=action,
                reward=result.reward,
                terminated=result.terminated,
                termination_reason=result.termination_reason,
                observation=result.observation,
                state=result.state,
                info=result.info,
            )
        )
        observation = result.observation
        if result.terminated:
            break

    final_state = frames[-1].state if frames else env.state.to_dict()
    summary = {
        "steps_completed": len(frames),
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "final_state": final_state,
    }
    return frames, summary
