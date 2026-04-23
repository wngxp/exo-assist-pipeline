from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class ResidualExoAssistance(Protocol):
    def apply(
        self,
        policy_action: Sequence[float],
        state: Mapping[str, int | float],
    ) -> list[float]:
        """Apply residual assistive action on top of the base policy action."""


class NullResidualExoAssistance:
    """Default pass-through assistance model used until exoskeleton support is added."""

    def apply(
        self,
        policy_action: Sequence[float],
        state: Mapping[str, int | float],
    ) -> list[float]:
        del state
        return [float(value) for value in policy_action]
