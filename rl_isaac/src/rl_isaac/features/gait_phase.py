from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GaitPhaseFeatures:
    phase: float
    sin_phase: float
    cos_phase: float

    def to_dict(self) -> dict[str, float]:
        return {
            "gait_phase": self.phase,
            "gait_phase_sin": self.sin_phase,
            "gait_phase_cos": self.cos_phase,
        }


class GaitPhaseFeatureProvider(Protocol):
    def compute(self, *, step_index: int, time_s: float) -> GaitPhaseFeatures:
        """Compute gait phase features for the current environment step."""


class SinusoidalGaitPhaseFeatureProvider:
    """Minimal deterministic phase encoder used by the scaffold backend."""

    def __init__(self, cadence_hz: float) -> None:
        self._cadence_hz = cadence_hz

    def compute(self, *, step_index: int, time_s: float) -> GaitPhaseFeatures:
        del step_index
        phase = (time_s * self._cadence_hz) % 1.0
        theta = 2.0 * math.pi * phase
        return GaitPhaseFeatures(phase=phase, sin_phase=math.sin(theta), cos_phase=math.cos(theta))
