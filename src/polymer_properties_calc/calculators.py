"""First-pass calculators for common polymer properties.

Protocols are deliberately explicit: callers provide an initialized OpenMM
Simulation with a compatible force field, integrator, and periodic box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from ._openmm import (
    configure_mechanical_barostat,
    density_g_cm3,
    mass_dalton,
    require_barostat,
    set_box_vectors_nm,
    set_pressure,
    set_temperature,
    unit,
    volume_nm3,
)
from .base import PropertyCalculator, PropertyResult, RawData, standard_error
from .exceptions import AnalysisError, InvalidSimulationError


@dataclass(frozen=True)
class SamplingProtocol:
    equilibration_steps: int = 10_000
    samples: int = 100
    steps_per_sample: int = 100

    def __post_init__(self) -> None:
        if self.equilibration_steps < 0:
            raise ValueError("equilibration_steps must be >= 0")
        if self.samples < 2:
            raise ValueError("samples must be >= 2")
        if self.steps_per_sample < 1:
            raise ValueError("steps_per_sample must be >= 1")


def _sample_scalar(
    simulation: Any,
    protocol: SamplingProtocol,
    sampler: Any,
) -> NDArray[np.float64]:
    if protocol.equilibration_steps:
        simulation.step(protocol.equilibration_steps)
    values = np.empty(protocol.samples, dtype=float)
    for index in range(protocol.samples):
        simulation.step(protocol.steps_per_sample)
        values[index] = sampler(simulation)
    return values


class DensityCalculator(PropertyCalculator):
    """Estimate mass density from an NPT trajectory (g/cm^3)."""

    def __init__(
        self,
        temperature: float,
        pressure: float = 1.0,
        protocol: SamplingProtocol = SamplingProtocol(),
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.pressure = float(pressure)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a positive finite value in K")
        if not np.isfinite(self.pressure):
            raise ValueError("pressure must be finite and is interpreted in bar")
        self.protocol = protocol

    def _collect(self, simulation: Any):
        require_barostat(simulation)
        set_temperature(simulation, self.temperature)
        set_pressure(simulation, self.pressure)
        values = _sample_scalar(simulation, self.protocol, density_g_cm3)
        return (
            {"density": values},
            {"density": "g/cm^3"},
            {
                "temperature_K": self.temperature,
                "pressure_bar": self.pressure,
                "ensemble": "NPT",
                "protocol": self.protocol,
            },
        )

    def _analyze(self, raw_data: RawData) -> PropertyResult:
        values = raw_data.values["density"]
        return PropertyResult(
            value=float(np.mean(values)),
            unit="g/cm^3",
            standard_error=standard_error(values),
            details={"standard_deviation": float(np.std(values, ddof=1))},
        )


@dataclass(frozen=True)
class HeatCapacityResult(PropertyResult):
    """Mass-specific heat capacity with the corresponding molar estimate."""

    molar_heat_capacity: float | None = None
    molar_unit: str = "kJ/(mol K)"
    ensemble: str = "NVT"


class HeatCapacityCalculator(PropertyCalculator):
    """Estimate Cv (NVT) or Cp (NPT) from energy/enthalpy autocorrelation.

    The fluctuation estimate uses the zero-lag energy autocovariance. The full
    autocorrelation is retained in ``raw_data`` and used to report its
    integrated correlation time and effective sample count.
    """

    _GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324

    def __init__(
        self,
        temperature: float,
        ensemble: Literal["NVT", "NPT"] = "NPT",
        pressure: float | None = None,
        protocol: SamplingProtocol = SamplingProtocol(),
    ) -> None:
        super().__init__()
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive finite and in K")
        if ensemble not in ("NVT", "NPT"):
            raise ValueError("ensemble must be 'NVT' or 'NPT'")
        if ensemble == "NPT" and pressure is None:
            raise ValueError("pressure in bar is required for an NPT heat-capacity calculation")
        if pressure is not None and not np.isfinite(pressure):
            raise ValueError("pressure must be finite and in bar")
        self.temperature = float(temperature)
        self.ensemble = ensemble
        self.pressure = None if pressure is None else float(pressure)
        self.protocol = protocol

    def _collect(self, simulation: Any):
        forces = [simulation.system.getForce(i) for i in range(simulation.system.getNumForces())]
        has_barostat = any("Barostat" in type(force).__name__ for force in forces)
        if self.ensemble == "NPT":
            require_barostat(simulation)
            set_pressure(simulation, float(self.pressure))
        elif has_barostat:
            raise InvalidSimulationError(
                "NVT heat capacity requires a Simulation without a MonteCarloBarostat"
            )
        set_temperature(simulation, self.temperature)

        if self.protocol.equilibration_steps:
            simulation.step(self.protocol.equilibration_steps)
        energy = np.empty(self.protocol.samples, dtype=float)
        volume = np.empty(self.protocol.samples, dtype=float)
        for index in range(self.protocol.samples):
            simulation.step(self.protocol.steps_per_sample)
            state = simulation.context.getState(getEnergy=True)
            total_energy = state.getPotentialEnergy() + state.getKineticEnergy()
            energy[index] = total_energy.value_in_unit(unit.kilojoule_per_mole)
            volume[index] = volume_nm3(simulation)

        if self.ensemble == "NPT":
            # 1 bar nm^3 = 0.0602214 kJ/mol.
            energy = energy + float(self.pressure) * volume * 0.0602214076
        step_size_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        sample_interval_ps = step_size_ps * self.protocol.steps_per_sample
        return (
            {"fluctuating_energy": energy, "volume": volume},
            {"fluctuating_energy": "kJ/mol", "volume": "nm^3"},
            {
                "temperature_K": self.temperature,
                "pressure_bar": self.pressure,
                "ensemble": self.ensemble,
                "protocol": self.protocol,
                "sample_interval_ps": float(sample_interval_ps),
                "energy_definition": "total_energy" if self.ensemble == "NVT" else "enthalpy",
                "mass_dalton": mass_dalton(simulation),
            },
        )

    def _analyze(self, raw_data: RawData) -> HeatCapacityResult:
        values = raw_data.values["fluctuating_energy"]
        centered = values - np.mean(values)
        autocovariance = _autocovariance(centered)
        variance = float(autocovariance[0])
        if not np.isfinite(variance) or variance <= 0:
            raise AnalysisError("energy/enthalpy variance must be positive to estimate heat capacity")
        molar_capacity = variance / (
            self._GAS_CONSTANT_KJ_PER_MOL_K * self.temperature**2
        )
        mass_capacity = molar_capacity * 1000.0 / raw_data.metadata["mass_dalton"]
        normalized = autocovariance / variance
        positive = normalized[1:]
        stop = next((i for i, value in enumerate(positive) if value <= 0), positive.size)
        integrated_time = raw_data.metadata["sample_interval_ps"] * (
            0.5 + float(np.sum(positive[:stop]))
        )
        effective_samples = min(
            float(values.size),
            float(values.size * raw_data.metadata["sample_interval_ps"] / (2.0 * integrated_time)),
        )
        return HeatCapacityResult(
            value=float(mass_capacity),
            unit="J/(g K)",
            molar_heat_capacity=float(molar_capacity),
            ensemble=self.ensemble,
            details={
                "mass_specific_heat_capacity_J_g_K": float(mass_capacity),
                "energy_variance_kJ2_mol2": variance,
                "integrated_autocorrelation_time_ps": float(integrated_time),
                "effective_sample_count": effective_samples,
                "mean_volume_nm3": float(np.mean(raw_data.values["volume"])),
                "energy_autocorrelation_kJ2_mol2": autocovariance.tolist(),
                "correlation_lag_ps": (
                    np.arange(autocovariance.size)
                    * raw_data.metadata["sample_interval_ps"]
                ).tolist(),
            },
        )


class GreenKuboThermalConductivityCalculator(PropertyCalculator):
    """Estimate isotropic thermal conductivity from a Green–Kubo heat current.

    OpenMM does not expose a generic microscopic heat-current operator. Supply
    ``heat_current_provider`` for the force field in use; it must return the
    extensive total heat-current vector in kJ nm/(mol ps), including the
    convective and interaction/virial terms required by that force field.
    The Simulation is sampled at its current temperature; this calculator does
    not change the integrator or thermostat settings.
    """

    _BOLTZMANN_J_PER_K = 1.380649e-23
    _AVOGADRO = 6.02214076e23

    def __init__(
        self,
        temperature: float,
        heat_current_provider: Callable[[Any], Sequence[float]],
        protocol: SamplingProtocol = SamplingProtocol(),
        max_lag_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive finite and in K")
        if not callable(heat_current_provider):
            raise ValueError("heat_current_provider must be callable")
        if protocol.samples < 3:
            raise ValueError("at least three heat-current samples are required")
        if not np.isfinite(max_lag_fraction) or not 0 < max_lag_fraction <= 1:
            raise ValueError("max_lag_fraction must be in (0, 1]")
        self.temperature = float(temperature)
        self.heat_current_provider = heat_current_provider
        self.protocol = protocol
        self.max_lag_fraction = float(max_lag_fraction)

    def _collect(self, simulation: Any):
        if self.protocol.equilibration_steps:
            simulation.step(self.protocol.equilibration_steps)
        current = np.empty((self.protocol.samples, 3), dtype=float)
        volumes = np.empty(self.protocol.samples, dtype=float)
        for index in range(self.protocol.samples):
            simulation.step(self.protocol.steps_per_sample)
            vector = np.asarray(self.heat_current_provider(simulation), dtype=float)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise InvalidSimulationError(
                    "heat_current_provider must return three finite values in kJ nm/(mol ps)"
                )
            current[index] = vector
            volumes[index] = volume_nm3(simulation)
        step_size_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        return (
            {"heat_current": current, "volume": volumes},
            {"heat_current": "kJ nm/(mol ps)", "volume": "nm^3"},
            {
                "temperature_K": self.temperature,
                "protocol": self.protocol,
                "sample_interval_ps": float(step_size_ps * self.protocol.steps_per_sample),
                "mean_volume_nm3": float(np.mean(volumes)),
                "autocorrelation_normalization": "unbiased; mean current removed",
            },
        )

    def _analyze(self, raw_data: RawData) -> PropertyResult:
        current = raw_data.values["heat_current"]
        if current.shape[0] < 3:
            raise AnalysisError("at least three heat-current samples are required")
        centered = current - np.mean(current, axis=0)
        component_covariances = np.column_stack(
            [_autocovariance(centered[:, component]) for component in range(3)]
        )
        current_autocorrelation = np.mean(component_covariances, axis=1)
        max_lag = min(
            current_autocorrelation.size - 1,
            max(1, int(np.floor((current_autocorrelation.size - 1) * self.max_lag_fraction))),
        )
        interval_ps = raw_data.metadata["sample_interval_ps"]
        time_ps = np.arange(max_lag + 1, dtype=float) * interval_ps
        correlation = current_autocorrelation[: max_lag + 1]
        integral = float(np.trapz(correlation, time_ps * 1e-12))
        # Convert kJ nm/(mol ps) to W m per molecular system, then apply SI GK.
        current_to_si = 1e6 / self._AVOGADRO
        volume_m3 = raw_data.metadata["mean_volume_nm3"] * 1e-27
        conductivity = (
            integral
            * current_to_si**2
            / (volume_m3 * self._BOLTZMANN_J_PER_K * self.temperature**2)
        )
        cumulative = np.zeros_like(correlation)
        if cumulative.size > 1:
            cumulative[1:] = np.cumsum(
                0.5 * (correlation[:-1] + correlation[1:]) * np.diff(time_ps * 1e-12)
            )
        conductivity_by_lag = (
            cumulative
            * current_to_si**2
            / (volume_m3 * self._BOLTZMANN_J_PER_K * self.temperature**2)
        )
        return PropertyResult(
            value=float(conductivity),
            unit="W/(m K)",
            details={
                "integration_cutoff_ps": float(time_ps[-1]),
                "sample_interval_ps": float(interval_ps),
                "mean_volume_nm3": raw_data.metadata["mean_volume_nm3"],
                "conductivity_by_lag_W_m_K": conductivity_by_lag.tolist(),
                "lag_time_ps": time_ps.tolist(),
            },
        )


class CTECalculator(PropertyCalculator):
    """Estimate volumetric CTE from equilibrium NPT volumes versus temperature.

    The reported coefficient is alpha_V = (1/V) dV/dT, in 1/K.
    """

    def __init__(
        self,
        temperatures: Sequence[float],
        pressure: float = 1.0,
        protocol: SamplingProtocol = SamplingProtocol(),
    ) -> None:
        super().__init__()
        self.temperatures = np.asarray(temperatures, dtype=float)
        if self.temperatures.ndim != 1 or self.temperatures.size < 3:
            raise ValueError("temperatures must contain at least three values")
        if np.any(self.temperatures <= 0) or np.unique(self.temperatures).size < 3:
            raise ValueError("temperatures must be positive and include 3 distinct values")
        self.pressure = float(pressure)
        if not np.isfinite(self.pressure):
            raise ValueError("pressure must be finite and is interpreted in bar")
        self.protocol = protocol

    def _collect(self, simulation: Any):
        require_barostat(simulation)
        volumes = np.empty((self.temperatures.size, self.protocol.samples), dtype=float)
        for index, temperature in enumerate(self.temperatures):
            set_temperature(simulation, float(temperature))
            set_pressure(simulation, self.pressure)
            volumes[index] = _sample_scalar(simulation, self.protocol, volume_nm3)
        return (
            {"temperature": self.temperatures.copy(), "volume": volumes},
            {"temperature": "K", "volume": "nm^3"},
            {
                "pressure_bar": self.pressure,
                "ensemble": "NPT",
                "protocol": self.protocol,
                "coefficient": "volumetric",
            },
        )

    def _analyze(self, raw_data: RawData) -> PropertyResult:
        temperatures = raw_data.values["temperature"]
        mean_volumes = np.mean(raw_data.values["volume"], axis=1)
        slope, intercept = np.polyfit(temperatures, mean_volumes, 1)
        coefficient = float(slope / np.mean(mean_volumes))
        fitted = slope * temperatures + intercept
        return PropertyResult(
            value=coefficient,
            unit="1/K",
            details={
                "volume_slope_nm3_per_K": float(slope),
                "fit_r_squared": _r_squared(mean_volumes, fitted),
                "temperatures_K": temperatures.tolist(),
            },
        )


@dataclass(frozen=True)
class GlassTransitionResult(PropertyResult):
    """Glass-transition estimate with optional CTE estimates on both sides."""

    cte_below_tg: float | None = None
    cte_above_tg: float | None = None


class GlassTransitionCalculator(PropertyCalculator):
    """Estimate Tg during continuous-rate heating and/or cooling scans.

    The target temperature follows a linear ramp in K/ns. To avoid updating the
    integrator every MD step, the target is refreshed every
    ``temperature_update_interval_steps`` and density is sampled separately at
    ``sample_interval_steps``.
    """

    def __init__(
        self,
        temperature_range: Sequence[float],
        rate_k_per_ns: float,
        mode: Literal["cooling", "heating", "heating_cooling", "cycles"] = "heating_cooling",
        cycles: int = 1,
        pressure: float = 1.0,
        equilibration_steps: int = 10_000,
        temperature_update_interval_steps: int = 100,
        sample_interval_steps: int = 10_000,
    ) -> None:
        super().__init__()
        self.temperature_range = np.asarray(temperature_range, dtype=float)
        if (
            self.temperature_range.shape != (2,)
            or not np.all(np.isfinite(self.temperature_range))
            or self.temperature_range[0] <= 0
            or self.temperature_range[0] >= self.temperature_range[1]
        ):
            raise ValueError("temperature_range must be (minimum_K, maximum_K), with 0 < minimum < maximum")
        if not np.isfinite(rate_k_per_ns) or rate_k_per_ns <= 0:
            raise ValueError("rate_k_per_ns must be a positive finite temperature ramp rate")
        if mode not in {"cooling", "heating", "heating_cooling", "cycles"}:
            raise ValueError("mode must be cooling, heating, heating_cooling, or cycles")
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        self.pressure = float(pressure)
        if not np.isfinite(self.pressure):
            raise ValueError("pressure must be finite and is interpreted in bar")
        if equilibration_steps < 0:
            raise ValueError("equilibration_steps must be >= 0")
        if temperature_update_interval_steps < 1 or sample_interval_steps < 1:
            raise ValueError("temperature_update_interval_steps and sample_interval_steps must be >= 1")
        self.rate_k_per_ns = float(rate_k_per_ns)
        self.mode = mode
        self.cycles = int(cycles) if mode == "cycles" else 1
        self.equilibration_steps = int(equilibration_steps)
        self.temperature_update_interval_steps = int(temperature_update_interval_steps)
        self.sample_interval_steps = int(sample_interval_steps)

    def _segments(self) -> list[tuple[float, float, int, str]]:
        low, high = (float(value) for value in self.temperature_range)
        if self.mode == "cooling":
            return [(high, low, 1, "cooling")]
        if self.mode == "heating":
            return [(low, high, 1, "heating")]
        cycle_count = self.cycles if self.mode == "cycles" else 1
        return [
            segment
            for cycle in range(1, cycle_count + 1)
            for segment in ((low, high, cycle, "heating"), (high, low, cycle, "cooling"))
        ]

    def _collect(self, simulation: Any):
        require_barostat(simulation)
        segments = self._segments()
        step_size_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        if not np.isfinite(step_size_ps) or step_size_ps <= 0:
            raise InvalidSimulationError("integrator must have a positive finite step size")
        delta_temperature_per_step = self.rate_k_per_ns * step_size_ps / 1000.0
        segment_step_counts = [
            int(np.ceil(abs(end - start) / delta_temperature_per_step))
            for start, end, _, _ in segments
        ]
        required_samples = [
            int(np.ceil(steps / self.sample_interval_steps)) + 1
            for steps in segment_step_counts
        ]
        if any(count < 5 for count in required_samples):
            raise ValueError(
                "the scan settings produce fewer than five samples per ramp; "
                "reduce sample_interval_steps or use a wider range/slower rate"
            )

        set_pressure(simulation, self.pressure)
        set_temperature(simulation, segments[0][0])
        if self.equilibration_steps:
            simulation.step(self.equilibration_steps)

        temperatures: list[float] = []
        densities: list[float] = []
        elapsed_steps: list[int] = []
        segment_indices: list[int] = []
        cycles: list[int] = []
        directions: list[int] = []
        total_elapsed_steps = self.equilibration_steps
        metadata_segments: list[dict[str, Any]] = []

        def record(temperature: float, segment_index: int, cycle: int, direction: str) -> None:
            temperatures.append(float(temperature))
            densities.append(density_g_cm3(simulation))
            elapsed_steps.append(total_elapsed_steps)
            segment_indices.append(segment_index)
            cycles.append(cycle)
            directions.append(1 if direction == "heating" else -1)

        for segment_index, (start, end, cycle, direction) in enumerate(segments):
            metadata_segments.append(
                {
                    "segment_index": segment_index,
                    "cycle": cycle,
                    "direction": direction,
                    "start_temperature_K": start,
                    "end_temperature_K": end,
                }
            )
            if segment_index == 0:
                record(start, segment_index, cycle, direction)
            else:
                # Record the shared turning point as the start of the new pass too.
                record(start, segment_index, cycle, direction)

            segment_steps = segment_step_counts[segment_index]
            elapsed_in_segment = 0
            next_sample_step = self.sample_interval_steps
            direction_sign = 1.0 if end > start else -1.0
            while elapsed_in_segment < segment_steps:
                steps = min(
                    self.temperature_update_interval_steps,
                    next_sample_step - elapsed_in_segment,
                    segment_steps - elapsed_in_segment,
                )
                midpoint_step = elapsed_in_segment + steps / 2.0
                midpoint_temperature = start + direction_sign * delta_temperature_per_step * midpoint_step
                midpoint_temperature = float(np.clip(midpoint_temperature, min(start, end), max(start, end)))
                set_temperature(simulation, midpoint_temperature)
                simulation.step(steps)
                elapsed_in_segment += steps
                total_elapsed_steps += steps

                current_temperature = start + direction_sign * delta_temperature_per_step * elapsed_in_segment
                if elapsed_in_segment >= segment_steps:
                    current_temperature = end
                current_temperature = float(np.clip(current_temperature, min(start, end), max(start, end)))
                if elapsed_in_segment == next_sample_step or elapsed_in_segment == segment_steps:
                    if elapsed_in_segment == segment_steps:
                        set_temperature(simulation, end)
                    record(current_temperature, segment_index, cycle, direction)
                    if elapsed_in_segment == next_sample_step:
                        next_sample_step += self.sample_interval_steps

        return (
            {
                "temperature": np.asarray(temperatures),
                "density": np.asarray(densities),
                "elapsed_steps": np.asarray(elapsed_steps),
                "segment_index": np.asarray(segment_indices),
                "cycle": np.asarray(cycles),
                "direction": np.asarray(directions),
            },
            {
                "temperature": "K",
                "density": "g/cm^3",
                "elapsed_steps": "steps",
                "segment_index": "index",
                "cycle": "index",
                "direction": "+1 heating, -1 cooling",
            },
            {
                "pressure_bar": self.pressure,
                "ensemble": "NPT",
                "mode": self.mode,
                "rate_K_per_ns": self.rate_k_per_ns,
                "temperature_range_K": self.temperature_range.tolist(),
                "temperature_update_interval_steps": self.temperature_update_interval_steps,
                "sample_interval_steps": self.sample_interval_steps,
                "equilibration_steps": self.equilibration_steps,
                "segments": metadata_segments,
            },
        )

    def analyze(self, *, include_cte: bool = False) -> GlassTransitionResult:
        if self.raw_data is None:
            raise InvalidSimulationError("call run(simulation) before analyze()")
        return self._analyze(self.raw_data, include_cte=include_cte)

    def _analyze(
        self, raw_data: RawData, *, include_cte: bool = False
    ) -> GlassTransitionResult:
        segment_results = []
        for segment in raw_data.metadata["segments"]:
            indices = raw_data.values["segment_index"] == segment["segment_index"]
            temperatures = raw_data.values["temperature"][indices]
            densities = raw_data.values["density"][indices]
            tg, fit_details = _fit_glass_transition(temperatures, densities)
            result = {**segment, "tg_K": tg, **fit_details}
            if include_cte:
                low_density_at_tg = (
                    fit_details["low_temperature_density_slope"] * tg
                    + fit_details["low_temperature_density_intercept"]
                )
                high_density_at_tg = (
                    fit_details["high_temperature_density_slope"] * tg
                    + fit_details["high_temperature_density_intercept"]
                )
                reference_density = 0.5 * (low_density_at_tg + high_density_at_tg)
                if not np.isfinite(reference_density) or reference_density <= 0:
                    raise AnalysisError("cannot calculate CTE: fitted density at Tg is not positive")
                result["cte_below_tg_1_per_K"] = (
                    -fit_details["low_temperature_density_slope"] / reference_density
                )
                result["cte_above_tg_1_per_K"] = (
                    -fit_details["high_temperature_density_slope"] / reference_density
                )
            segment_results.append(result)
        estimates = np.asarray([result["tg_K"] for result in segment_results], dtype=float)
        by_direction = {
            direction: float(
                np.mean([result["tg_K"] for result in segment_results if result["direction"] == direction])
            )
            for direction in {result["direction"] for result in segment_results}
        }
        cte_below = None
        cte_above = None
        mean_cte_by_direction: dict[str, dict[str, float]] = {}
        if include_cte:
            cte_below = float(np.mean([result["cte_below_tg_1_per_K"] for result in segment_results]))
            cte_above = float(np.mean([result["cte_above_tg_1_per_K"] for result in segment_results]))
            for direction in {result["direction"] for result in segment_results}:
                same_direction = [result for result in segment_results if result["direction"] == direction]
                mean_cte_by_direction[direction] = {
                    "below_tg_1_per_K": float(
                        np.mean([result["cte_below_tg_1_per_K"] for result in same_direction])
                    ),
                    "above_tg_1_per_K": float(
                        np.mean([result["cte_above_tg_1_per_K"] for result in same_direction])
                    ),
                }
        return GlassTransitionResult(
            value=float(np.mean(estimates)),
            unit="K",
            standard_error=standard_error(estimates),
            cte_below_tg=cte_below,
            cte_above_tg=cte_above,
            details={
                "mean_tg_by_direction_K": by_direction,
                "per_pass": segment_results,
                "rate_K_per_ns": raw_data.metadata["rate_K_per_ns"],
                "mode": raw_data.metadata["mode"],
                "mean_cte_by_direction_1_per_K": mean_cte_by_direction,
            },
        )


@dataclass(frozen=True)
class MechanicalPropertiesResult(PropertyResult):
    """Uniaxial Young's modulus and, when available, Poisson ratio."""

    poisson_ratio: float | None = None
    yield_stress: float | None = None
    yield_strain: float | None = None
    max_tensile_stress: float | None = None
    strain_at_max_tensile_stress: float | None = None

    @property
    def young_modulus(self) -> float:
        """Young's modulus in the unit returned by ``unit`` (GPa)."""

        return self.value


class MechanicalPropertiesCalculator(PropertyCalculator):
    """Run a uniaxial tensile protocol and analyze its pressure-strain curve.

    ``strain_rate`` is engineering strain per second (s⁻¹). The selected box
    vector is extended in discrete updates every ``stretch_interval_steps``;
    between updates the simulation advances normally. An anisotropic OpenMM
    barostat measures the instantaneous pressure and optionally relaxes the
    two transverse axes to ``transverse_pressure``.

    The Young's modulus and Poisson ratio are fitted only within
    ``elastic_strain_range``. Yield detection is controlled by ``yield_method``
    and ``yield_tolerance``; reported stresses are in GPa.
    """

    _AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        axis: str = "x",
        relax_transverse_axes: bool = True,
        strain_rate: float = 1e8,
        stretch_interval_steps: int = 100,
        max_strain: float = 0.1,
        elastic_strain_range: tuple[float, float] = (0.0, 0.02),
        yield_method: Literal["first_deviation", "stress_drop"] = "first_deviation",
        yield_tolerance: float = 0.05,
        transverse_pressure: float = 1.0,
        temperature: float | None = None,
        equilibration_steps: int = 5_000,
        transverse_barostat_frequency: int = 25,
    ) -> None:
        super().__init__()
        axis = axis.lower()
        if axis not in self._AXIS_INDEX:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")
        if not np.isfinite(strain_rate) or strain_rate <= 0:
            raise ValueError("strain_rate must be positive finite engineering strain per second")
        if stretch_interval_steps < 1:
            raise ValueError("stretch_interval_steps must be >= 1")
        if not np.isfinite(max_strain) or max_strain <= 0:
            raise ValueError("max_strain must be a positive finite engineering strain")
        if len(elastic_strain_range) != 2:
            raise ValueError("elastic_strain_range must be a (minimum, maximum) pair")
        elastic_min, elastic_max = map(float, elastic_strain_range)
        if (
            not np.isfinite(elastic_min)
            or not np.isfinite(elastic_max)
            or elastic_min < 0
            or elastic_max <= elastic_min
            or elastic_max > max_strain
        ):
            raise ValueError("elastic_strain_range must satisfy 0 <= min < max <= max_strain")
        if yield_method not in ("first_deviation", "stress_drop"):
            raise ValueError("yield_method must be 'first_deviation' or 'stress_drop'")
        if not np.isfinite(yield_tolerance) or not 0 < yield_tolerance < 1:
            raise ValueError("yield_tolerance must be between 0 and 1")
        if not np.isfinite(transverse_pressure):
            raise ValueError("transverse_pressure must be finite and in bar")
        if temperature is not None and (not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("temperature must be positive finite and in K")
        if equilibration_steps < 0 or transverse_barostat_frequency < 1:
            raise ValueError("invalid equilibration or barostat frequency")
        self.axis = axis
        self.axis_index = self._AXIS_INDEX[axis]
        self.relax_transverse_axes = bool(relax_transverse_axes)
        self.strain_rate = float(strain_rate)
        self.stretch_interval_steps = int(stretch_interval_steps)
        self.max_strain = float(max_strain)
        self.elastic_strain_range = (elastic_min, elastic_max)
        self.yield_method = yield_method
        self.yield_tolerance = float(yield_tolerance)
        self.transverse_pressure = float(transverse_pressure)
        self.temperature = None if temperature is None else float(temperature)
        self.equilibration_steps = int(equilibration_steps)
        self.transverse_barostat_frequency = int(transverse_barostat_frequency)

    def _collect(self, simulation: Any):
        step_size_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        if not np.isfinite(step_size_ps) or step_size_ps <= 0:
            raise InvalidSimulationError("integrator must have a positive finite step size")
        strain_per_step = self.strain_rate * step_size_ps * 1e-12
        strain_per_interval = strain_per_step * self.stretch_interval_steps
        if self.max_strain / strain_per_interval < 2:
            raise ValueError(
                "max_strain and strain_rate produce fewer than two stretch intervals; "
                "increase max_strain or reduce strain_rate/stretch_interval_steps"
            )

        barostat, temperature = configure_mechanical_barostat(
            simulation,
            stretch_axis=self.axis_index,
            relax_transverse_axes=self.relax_transverse_axes,
            transverse_pressure_bar=self.transverse_pressure,
            temperature_kelvin=self.temperature,
            barostat_frequency=self.transverse_barostat_frequency,
        )

        if self.equilibration_steps:
            simulation.step(self.equilibration_steps)

        initial = simulation.context.getState(getPositions=True)
        initial_positions = initial.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        initial_box = np.asarray(
            initial.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=float,
        )
        if abs(float(np.linalg.det(initial_box))) <= 0:
            raise InvalidSimulationError("mechanical test requires a valid periodic box")
        initial_lengths = np.linalg.norm(initial_box, axis=1)

        strains = [0.0]
        pressures = [_pressure_component(barostat.computeCurrentPressure(simulation.context), self.axis_index)]
        lateral_strains = [np.zeros(2, dtype=float)]
        elapsed_steps = [0]
        completed_steps = 0
        current_strain = 0.0

        while current_strain < self.max_strain:
            remaining_strain = self.max_strain - current_strain
            steps = min(
                self.stretch_interval_steps,
                max(1, int(np.ceil(remaining_strain / strain_per_step))),
            )
            target_strain = min(
                self.max_strain,
                current_strain + strain_per_step * steps,
            )
            stretch_factor = (1.0 + target_strain) / (1.0 + current_strain)
            box = np.asarray(
                simulation.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    unit.nanometer
                ),
                dtype=float,
            )
            old_positions = simulation.context.getState(getPositions=True).getPositions(
                asNumpy=True
            ).value_in_unit(unit.nanometer)
            new_box = box.copy()
            new_box[self.axis_index] *= stretch_factor
            fractional = np.asarray(old_positions) @ np.linalg.inv(box)
            set_box_vectors_nm(simulation, new_box)
            simulation.context.setPositions(fractional @ new_box * unit.nanometer)
            simulation.context.applyConstraints(1e-5)
            simulation.step(steps)

            state = simulation.context.getState()
            measured_box = np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=float,
            )
            measured_lengths = np.linalg.norm(measured_box, axis=1)
            lateral_axes = [index for index in range(3) if index != self.axis_index]
            pressures.append(
                _pressure_component(barostat.computeCurrentPressure(simulation.context), self.axis_index)
            )
            # Use the actual cell length so transverse barostat relaxation is included.
            strains.append(float(measured_lengths[self.axis_index] / initial_lengths[self.axis_index] - 1.0))
            lateral_strains.append(
                measured_lengths[lateral_axes] / initial_lengths[lateral_axes] - 1.0
            )
            completed_steps += steps
            elapsed_steps.append(completed_steps)
            current_strain = target_strain

        return (
            {
                "strain": np.asarray(strains),
                "axial_pressure": np.asarray(pressures),
                "transverse_strain": np.asarray(lateral_strains),
                "elapsed_steps": np.asarray(elapsed_steps),
            },
            {
                "strain": "dimensionless",
                "axial_pressure": "bar",
                "transverse_strain": "dimensionless",
                "elapsed_steps": "steps",
            },
            {
                "axis": self.axis,
                "max_strain": self.max_strain,
                "elastic_strain_range": self.elastic_strain_range,
                "yield_method": self.yield_method,
                "yield_tolerance": self.yield_tolerance,
                "strain_rate_per_s": self.strain_rate,
                "stretch_interval_steps": self.stretch_interval_steps,
                "relax_transverse_axes": self.relax_transverse_axes,
                "transverse_pressure_bar": self.transverse_pressure,
                "temperature_K": temperature,
                "pressure_method": "MonteCarloAnisotropicBarostat.computeCurrentPressure",
                "equilibration_steps": self.equilibration_steps,
            },
        )

    def _analyze(self, raw_data: RawData) -> MechanicalPropertiesResult:
        strain = raw_data.values["strain"]
        pressure = raw_data.values["axial_pressure"]
        # OpenMM pressure is positive in compression; convert to tensile stress.
        stress_gpa = -pressure * 1e-4
        elastic_min, elastic_max = self.elastic_strain_range
        elastic_mask = (strain >= elastic_min) & (strain <= elastic_max)
        elastic_strain = strain[elastic_mask]
        elastic_stress = stress_gpa[elastic_mask]
        if elastic_strain.size < 3 or np.unique(elastic_strain).size < 3:
            raise AnalysisError(
                "the elastic_strain_range must contain at least three distinct samples; "
                "increase max_strain or adjust strain_rate/stretch_interval_steps"
            )
        slope_gpa, intercept_gpa = np.polyfit(elastic_strain, elastic_stress, 1)
        modulus_gpa = float(slope_gpa)
        fitted_stress = slope_gpa * elastic_strain + intercept_gpa
        poisson_ratio: float | None = None
        poisson_details: dict[str, float] = {}
        if self.relax_transverse_axes:
            transverse = raw_data.values["transverse_strain"]
            lateral_axes = [index for index in range(3) if index != self.axis_index]
            estimates = []
            for column, axis_index in enumerate(lateral_axes):
                lateral_slope = float(
                    np.polyfit(elastic_strain, transverse[elastic_mask, column], 1)[0]
                )
                estimate = -lateral_slope
                estimates.append(estimate)
                poisson_details["xyz"[axis_index]] = estimate
            poisson_ratio = float(np.mean(estimates))

        yield_stress: float | None = None
        yield_strain: float | None = None
        beyond_elastic = np.flatnonzero(strain > elastic_max)
        if self.yield_method == "first_deviation" and beyond_elastic.size:
            predicted = slope_gpa * strain[beyond_elastic] + intercept_gpa
            stress_scale = max(
                abs(float(slope_gpa * elastic_max + intercept_gpa)),
                float(np.max(np.abs(elastic_stress))),
                np.finfo(float).eps,
            )
            deviations = np.abs(stress_gpa[beyond_elastic] - predicted)
            candidates = np.flatnonzero(deviations >= self.yield_tolerance * stress_scale)
            if candidates.size:
                index = int(beyond_elastic[candidates[0]])
                yield_stress = float(stress_gpa[index])
                yield_strain = float(strain[index])
        elif self.yield_method == "stress_drop" and beyond_elastic.size >= 2:
            running_peak_index = int(beyond_elastic[0])
            for raw_index in beyond_elastic[1:]:
                index = int(raw_index)
                if stress_gpa[index] > stress_gpa[running_peak_index]:
                    running_peak_index = index
                peak = float(stress_gpa[running_peak_index])
                if peak > 0 and stress_gpa[index] <= peak * (1.0 - self.yield_tolerance):
                    yield_stress = peak
                    yield_strain = float(strain[running_peak_index])
                    break

        max_index = int(np.argmax(stress_gpa))
        return MechanicalPropertiesResult(
            value=modulus_gpa,
            unit="GPa",
            details={
                "elastic_strain_range": [elastic_min, elastic_max],
                "stress_strain_fit_r_squared": _r_squared(elastic_stress, fitted_stress),
                "strain_range": [float(np.min(strain)), float(np.max(strain))],
                "poisson_by_transverse_axis": poisson_details,
                "yield_method": self.yield_method,
                "yield_tolerance": self.yield_tolerance,
                "yield_detected": yield_strain is not None,
                "maximum_is_within_simulated_strain_range": True,
            },
            poisson_ratio=poisson_ratio,
            yield_stress=yield_stress,
            yield_strain=yield_strain,
            max_tensile_stress=float(stress_gpa[max_index]),
            strain_at_max_tensile_stress=float(strain[max_index]),
        )


def _pressure_component(pressure: Any, axis: int) -> float:
    """OpenMM barostat pressure components are reported in bar."""

    if hasattr(pressure, "value_in_unit"):
        return float(pressure.value_in_unit(unit.bar)[axis])
    return float(pressure[axis])


def _autocovariance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unbiased one-dimensional autocovariance, evaluated with an FFT."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("autocovariance input must be a non-empty one-dimensional array")
    fft_size = 1 << (2 * values.size - 1).bit_length()
    transformed = np.fft.rfft(values, n=fft_size)
    sums = np.fft.irfft(transformed * np.conjugate(transformed), n=fft_size)[: values.size]
    return np.asarray(sums / np.arange(values.size, 0, -1), dtype=float)


def _fit_glass_transition(
    temperatures: NDArray[np.float64], densities: NDArray[np.float64]
) -> tuple[float, dict[str, Any]]:
    if temperatures.size < 5 or np.unique(temperatures).size < 5:
        raise AnalysisError("each temperature ramp needs at least five distinct density samples")
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    densities = densities[order]
    best: tuple[float, int, NDArray[np.float64]] | None = None
    for split in range(2, temperatures.size - 1):
        low_fit = np.polyfit(temperatures[:split], densities[:split], 1)
        high_fit = np.polyfit(temperatures[split:], densities[split:], 1)
        residual = np.sum((densities[:split] - np.polyval(low_fit, temperatures[:split])) ** 2)
        residual += np.sum((densities[split:] - np.polyval(high_fit, temperatures[split:])) ** 2)
        if best is None or residual < best[0]:
            best = (float(residual), split, np.array([*low_fit, *high_fit]))
    if best is None:
        raise AnalysisError("not enough temperature points for a two-line Tg fit")
    _, split, coefficients = best
    low_slope, low_intercept, high_slope, high_intercept = coefficients
    if np.isclose(low_slope, high_slope):
        raise AnalysisError("the fitted density branches are parallel; Tg is undefined")
    tg = float((high_intercept - low_intercept) / (low_slope - high_slope))
    if not np.isfinite(tg):
        raise AnalysisError("the two-line Tg fit produced a non-finite intersection")
    return tg, {
        "fit_residual_sum_squares": best[0],
        "breakpoint_interval_K": [float(temperatures[split - 1]), float(temperatures[split])],
        "low_temperature_density_slope": float(low_slope),
        "low_temperature_density_intercept": float(low_intercept),
        "high_temperature_density_slope": float(high_slope),
        "high_temperature_density_intercept": float(high_intercept),
    }


def _r_squared(observed: NDArray[np.float64], fitted: NDArray[np.float64]) -> float:
    residual = float(np.sum((observed - fitted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if np.isclose(total, 0.0):
        return 1.0 if np.isclose(residual, 0.0) else 0.0
    return 1.0 - residual / total


# Descriptive spelling retained as an alias for discoverability.
ThermalExpansionCalculator = CTECalculator
