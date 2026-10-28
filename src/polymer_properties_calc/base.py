"""Shared data structures and calculator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from .exceptions import InvalidSimulationError


@dataclass(frozen=True)
class RawData:
    """Raw numeric observations and the units/metadata needed to interpret them."""

    values: dict[str, NDArray[np.float64]]
    units: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PropertyResult:
    """A scalar estimate and basic uncertainty/fit diagnostics."""

    value: float
    unit: str
    standard_error: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class PropertyCalculator(ABC):
    """Base class for calculators operating on a caller-provided Simulation.

    ``run`` advances and may otherwise modify the supplied simulation. It
    stores observations on ``raw_data`` and returns this calculator so callers
    can inspect the samples before calling ``analyze``.
    """

    raw_data: RawData | None

    def __init__(self) -> None:
        self.raw_data = None

    def run(self, simulation: Any) -> Self:
        if not hasattr(simulation, "step") or not hasattr(simulation, "context"):
            raise InvalidSimulationError(
                "simulation must be an initialized OpenMM app.Simulation"
            )
        self.raw_data = None
        values, units, metadata = self._collect(simulation)
        self.raw_data = RawData(
            values={key: np.asarray(value, dtype=np.float64) for key, value in values.items()},
            units=units,
            metadata=metadata,
        )
        return self

    def analyze(self) -> PropertyResult:
        if self.raw_data is None:
            raise InvalidSimulationError("call run(simulation) before analyze()")
        return self._analyze(self.raw_data)

    @abstractmethod
    def _collect(
        self, simulation: Any
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Run the protocol and return named observations, units, and metadata."""

    @abstractmethod
    def _analyze(self, raw_data: RawData) -> PropertyResult:
        """Convert the collected observations into a property estimate."""


def standard_error(values: NDArray[np.float64]) -> float | None:
    """Return the standard error of the mean, or None for fewer than 2 values."""

    if values.size < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(values.size))
