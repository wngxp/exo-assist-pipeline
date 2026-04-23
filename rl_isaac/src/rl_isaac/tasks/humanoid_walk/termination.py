from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rl_isaac.tasks.humanoid_walk.config import TerminationConfig

if TYPE_CHECKING:
    from rl_isaac.tasks.humanoid_walk.env import HumanoidWalkState


@dataclass(frozen=True, slots=True)
class TerminationStatus:
    terminated: bool
    reason: str | None


def check_termination(
    *,
    state: "HumanoidWalkState",
    config: TerminationConfig,
) -> TerminationStatus:
    if state.step_index >= config.max_steps:
        return TerminationStatus(terminated=True, reason="max_steps")
    if state.base_height < config.min_base_height:
        return TerminationStatus(terminated=True, reason="base_height")
    if state.torso_upright < config.min_upright:
        return TerminationStatus(terminated=True, reason="torso_upright")
    return TerminationStatus(terminated=False, reason=None)
