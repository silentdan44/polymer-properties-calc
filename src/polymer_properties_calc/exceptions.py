"""Exceptions raised by polymer property calculators."""


class PolymerPropertiesError(Exception):
    """Base exception for this package."""


class InvalidSimulationError(PolymerPropertiesError, ValueError):
    """The supplied OpenMM simulation cannot run the requested protocol."""


class AnalysisError(PolymerPropertiesError, ValueError):
    """Collected data are insufficient or unsuitable for analysis."""
