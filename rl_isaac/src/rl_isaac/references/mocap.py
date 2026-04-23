from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True, slots=True)
class MocapFrame:
    time_s: float
    joint_targets: dict[str, float]


@dataclass(frozen=True, slots=True)
class MocapSequence:
    sequence_id: str
    frames: tuple[MocapFrame, ...]
    metadata: dict[str, str]


class MocapReferenceLoader(Protocol):
    def load(self, sequence_id: str) -> MocapSequence:
        """Load a reference motion sequence by identifier."""


class UnconfiguredMocapReferenceLoader:
    """Explicit placeholder until a dataset contract is selected and implemented."""

    def load(self, sequence_id: str) -> MocapSequence:
        raise NotImplementedError(
            f"Mocap reference loading for '{sequence_id}' is planned but not implemented yet."
        )


def reference_error_features(
    current_joint_targets: Mapping[str, float],
    reference_joint_targets: Mapping[str, float],
) -> dict[str, float]:
    """Small helper reserved for future reference-tracking observation/reward terms."""
    shared_keys = set(current_joint_targets) & set(reference_joint_targets)
    if not shared_keys:
        return {"reference_tracking_error": 0.0}
    total_error = sum(
        abs(float(current_joint_targets[key]) - float(reference_joint_targets[key]))
        for key in shared_keys
    )
    return {"reference_tracking_error": total_error / float(len(shared_keys))}
