from __future__ import annotations

from typing import Mapping, Sequence

from rl_isaac.assistance.residual import ResidualExoAssistance
from rl_isaac.tasks.humanoid_walk.config import ActionConfig


def resolve_action(
    raw_action: Sequence[float],
    *,
    config: ActionConfig,
    state: Mapping[str, int | float],
    residual_assistance: ResidualExoAssistance | None = None,
) -> list[float]:
    values = _coerce_action(raw_action, config.dimension)
    clipped = [max(-config.clamp, min(config.clamp, value)) for value in values]
    scaled = [value * config.scale for value in clipped]
    if residual_assistance is None:
        return scaled
    return residual_assistance.apply(scaled, state)


def action_stats(action: Sequence[float]) -> dict[str, float]:
    if not action:
        return {"action_mean": 0.0, "action_abs_mean": 0.0}
    mean = sum(float(value) for value in action) / float(len(action))
    abs_mean = sum(abs(float(value)) for value in action) / float(len(action))
    return {"action_mean": mean, "action_abs_mean": abs_mean}


def _coerce_action(raw_action: Sequence[float], dimension: int) -> list[float]:
    action = [float(value) for value in raw_action[:dimension]]
    if len(action) < dimension:
        action.extend([0.0] * (dimension - len(action)))
    return action
