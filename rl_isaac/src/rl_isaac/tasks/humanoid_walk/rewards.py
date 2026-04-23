from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from rl_isaac.tasks.humanoid_walk.actions import action_stats
from rl_isaac.tasks.humanoid_walk.config import RewardConfig

if TYPE_CHECKING:
    from rl_isaac.tasks.humanoid_walk.env import HumanoidWalkState


@dataclass(frozen=True)
class RewardBreakdown:
    raw_terms: dict[str, float]
    weighted_terms: dict[str, float]
    total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_terms": dict(self.raw_terms),
            "weighted_terms": dict(self.weighted_terms),
            "total": self.total,
        }


def compute_reward(
    *,
    state: "HumanoidWalkState",
    action: Sequence[float],
    config: RewardConfig,
) -> RewardBreakdown:
    stats = action_stats(action)
    raw_terms = {
        "forward_velocity_tracking": max(
            0.0,
            1.0 - abs(config.target_forward_velocity - state.forward_velocity),
        ),
        "upright_posture": max(0.0, min(1.0, state.torso_upright)),
        "foot_clearance": max(
            0.0,
            min(1.0, state.foot_clearance / max(config.target_foot_clearance, 1e-6)),
        ),
        "action_rate": -stats["action_abs_mean"],
        "survival": 1.0,
    }

    unknown_terms = set(config.weights) - set(raw_terms)
    if unknown_terms:
        unknown = ", ".join(sorted(unknown_terms))
        raise KeyError(f"Unsupported reward terms in config: {unknown}")

    weighted_terms = {
        name: raw_terms[name] * float(config.weights[name])
        for name in config.weights
    }
    total = sum(weighted_terms.values())
    return RewardBreakdown(raw_terms=raw_terms, weighted_terms=weighted_terms, total=total)
