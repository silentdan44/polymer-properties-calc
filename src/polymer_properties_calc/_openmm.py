"""Small helpers around OpenMM quantities and simulation contexts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InvalidSimulationError

try:
    from openmm import (
        AndersenThermostat,
        MonteCarloAnisotropicBarostat,
        MonteCarloBarostat,
        Vec3,
        unit,
    )
except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
    raise ImportError(
        "polymer-properties-calc requires OpenMM; install it with `pip install openmm`"
    ) from exc


def _system(simulation: Any) -> Any:
    system = getattr(simulation, "system", None)
    if system is None:
        raise InvalidSimulationError("simulation has no OpenMM System")
    return system


def require_barostat(simulation: Any) -> None:
    if not any(
        isinstance(force, (MonteCarloBarostat, MonteCarloAnisotropicBarostat))
        for force in _system(simulation).getForces()
    ):
        raise InvalidSimulationError(
            "this protocol requires an existing MonteCarloBarostat or "
            "MonteCarloAnisotropicBarostat in the Simulation System"
        )


def set_temperature(simulation: Any, temperature_kelvin: float) -> None:
    """Update thermostat and barostat temperature parameters where available."""

    temperature = temperature_kelvin * unit.kelvin
    integrator = simulation.integrator
    setter = getattr(integrator, "setTemperature", None)
    if setter is not None:
        setter(temperature)

    parameters_updated = _set_context_parameters(simulation, "temperature", temperature_kelvin)
    has_andersen_thermostat = any(
        isinstance(force, AndersenThermostat) for force in _system(simulation).getForces()
    )
    if setter is None and not has_andersen_thermostat:
        raise InvalidSimulationError(
            "the integrator does not expose setTemperature() and the System has "
            "no AndersenThermostat; configure a thermostat before running"
        )
    if has_andersen_thermostat and not parameters_updated:
        raise InvalidSimulationError("AndersenThermostat temperature Context parameter is missing")


def set_pressure(simulation: Any, pressure_bar: float) -> None:
    require_barostat(simulation)
    if not _set_context_parameters(simulation, "pressure", pressure_bar):
        raise InvalidSimulationError(
            "the existing barostat does not expose a pressure Context parameter"
        )


def _set_context_parameters(simulation: Any, word: str, value: float) -> bool:
    parameters = simulation.context.getParameters()
    matches = [name for name in parameters if word in name.lower()]
    for name in matches:
        simulation.context.setParameter(name, value)
    return bool(matches)


def volume_nm3(simulation: Any) -> float:
    state = simulation.context.getState()
    vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    vectors = np.asarray(vectors, dtype=float)
    volume = abs(float(np.linalg.det(vectors)))
    if not np.isfinite(volume) or volume <= 0:
        raise InvalidSimulationError("Simulation has no valid periodic box")
    return volume


def mass_dalton(simulation: Any) -> float:
    total = sum(
        simulation.system.getParticleMass(index).value_in_unit(unit.dalton)
        for index in range(simulation.system.getNumParticles())
    )
    if total <= 0:
        raise InvalidSimulationError("Simulation System has zero total particle mass")
    return float(total)


def density_g_cm3(simulation: Any) -> float:
    # 1 Da/nm^3 = 1.66053906660 g/cm^3.
    return mass_dalton(simulation) / volume_nm3(simulation) * 1.66053906660


def set_box_vectors_nm(simulation: Any, vectors_nm: np.ndarray) -> None:
    vectors = np.asarray(vectors_nm, dtype=float)
    simulation.context.setPeriodicBoxVectors(
        *(vector * unit.nanometer for vector in vectors)
    )


def configure_mechanical_barostat(
    simulation: Any,
    *,
    stretch_axis: int,
    relax_transverse_axes: bool,
    transverse_pressure_bar: float,
    temperature_kelvin: float | None,
    barostat_frequency: int,
) -> tuple[Any, float]:
    """Install a pressure estimator and optional transverse-axis barostat."""

    system = _system(simulation)
    old_barostats = [
        force
        for force in system.getForces()
        if isinstance(force, (MonteCarloBarostat, MonteCarloAnisotropicBarostat))
    ]
    temperature = _simulation_temperature(simulation, old_barostats, temperature_kelvin)

    # A single anisotropic barostat both exposes the pressure tensor estimator
    # and, when requested, relaxes only the two axes perpendicular to tension.
    for index in range(system.getNumForces() - 1, -1, -1):
        if isinstance(system.getForce(index), (MonteCarloBarostat, MonteCarloAnisotropicBarostat)):
            system.removeForce(index)
    scale_axes = [False, False, False]
    if relax_transverse_axes:
        scale_axes = [axis != stretch_axis for axis in range(3)]
        frequency = barostat_frequency
    else:
        frequency = 0
    pressure = Vec3(
        transverse_pressure_bar,
        transverse_pressure_bar,
        transverse_pressure_bar,
    ) * unit.bar
    barostat = MonteCarloAnisotropicBarostat(
        pressure,
        temperature * unit.kelvin,
        *scale_axes,
        frequency=frequency,
    )
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    set_temperature(simulation, temperature)
    return barostat, temperature


def _simulation_temperature(
    simulation: Any,
    barostats: list[Any],
    requested_temperature: float | None,
) -> float:
    if requested_temperature is not None:
        temperature = float(requested_temperature)
    else:
        getter = getattr(simulation.integrator, "getTemperature", None)
        if getter is not None:
            value = getter()
            temperature = (
                value.value_in_unit(unit.kelvin)
                if hasattr(value, "value_in_unit")
                else float(value)
            )
        elif barostats:
            temperature = float(barostats[0].getDefaultTemperature())
        else:
            thermostat_temperatures = [
                force.getDefaultTemperature()
                for force in _system(simulation).getForces()
                if isinstance(force, AndersenThermostat)
            ]
            if not thermostat_temperatures:
                raise InvalidSimulationError(
                    "cannot infer the simulation temperature; pass temperature in K"
                )
            temperature = float(thermostat_temperatures[0])
    if not np.isfinite(temperature) or temperature <= 0:
        raise InvalidSimulationError("temperature must be a positive finite value in K")
    return temperature
