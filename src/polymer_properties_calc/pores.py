"""Pore geometry analysis of periodic OpenMM snapshots with Zeo++."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._openmm import unit
from .base import RawData, standard_error
from .exceptions import AnalysisError, InvalidSimulationError


@dataclass(frozen=True)
class PorePropertiesResult:
    """Frame-averaged pore descriptors and the Zeo++ PSD histogram.

    Pore diameters are in angstrom, accessible surface area is in m²/g,
    accessible volume fraction is dimensionless, and PSD is the derivative
    distribution reported by Zeo++ on its pore-diameter bins.
    """

    largest_cavity_diameter_angstrom: float
    pore_limiting_diameter_angstrom: float
    accessible_path_cavity_diameter_angstrom: float
    accessible_surface_area_m2_per_g: float
    accessible_volume_fraction: float
    channel_dimensionality: float
    pore_diameter_bins_angstrom: NDArray[np.float64]
    pore_size_distribution: NDArray[np.float64]
    standard_errors: dict[str, float | None]
    raw_data: RawData


class PorePropertiesCalculator:
    """Run Zeo++ pore analysis on periodically sampled OpenMM configurations.

    ``network`` is an external Zeo++ executable; install Zeo++ separately or
    pass its full path. The calculation advances and modifies the supplied
    Simulation by the requested equilibration and sampling steps.
    """

    def __init__(
        self,
        *,
        zeo_executable: str = "network",
        equilibration_steps: int = 0,
        n_frames: int = 10,
        sample_interval_steps: int = 10_000,
        probe_radius_angstrom: float = 0.0,
        monte_carlo_samples: int = 50_000,
        high_accuracy: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        if equilibration_steps < 0 or n_frames < 1 or sample_interval_steps < 1:
            raise ValueError("step counts must be nonnegative and frame/interval counts positive")
        if not np.isfinite(probe_radius_angstrom) or probe_radius_angstrom < 0:
            raise ValueError("probe_radius_angstrom must be finite and nonnegative")
        if monte_carlo_samples < 1:
            raise ValueError("monte_carlo_samples must be positive")
        self.zeo_executable = zeo_executable
        self.equilibration_steps = int(equilibration_steps)
        self.n_frames = int(n_frames)
        self.sample_interval_steps = int(sample_interval_steps)
        self.probe_radius_angstrom = float(probe_radius_angstrom)
        self.monte_carlo_samples = int(monte_carlo_samples)
        self.high_accuracy = bool(high_accuracy)
        self.timeout_seconds = timeout_seconds
        self.raw_data: RawData | None = None

    def run(self, simulation: Any) -> "PorePropertiesCalculator":
        if not hasattr(simulation, "step") or not hasattr(simulation, "context"):
            raise InvalidSimulationError("simulation must be an initialized OpenMM app.Simulation")
        executable = shutil.which(self.zeo_executable)
        if executable is None:
            raise InvalidSimulationError(
                f"Zeo++ executable {self.zeo_executable!r} was not found; "
                "install Zeo++ and set zeo_executable to its network binary"
            )

        frames: dict[str, list[float]] = {
            "largest_cavity_diameter_angstrom": [],
            "pore_limiting_diameter_angstrom": [],
            "accessible_path_cavity_diameter_angstrom": [],
            "accessible_surface_area_m2_per_g": [],
            "accessible_volume_fraction": [],
            "channel_dimensionality": [],
        }
        psd_rows: list[NDArray[np.float64]] = []
        bins: NDArray[np.float64] | None = None
        if self.equilibration_steps:
            simulation.step(self.equilibration_steps)

        with tempfile.TemporaryDirectory(prefix="polymer_pores_") as temp_dir:
            workdir = Path(temp_dir)
            for frame_index in range(self.n_frames):
                if self.sample_interval_steps:
                    simulation.step(self.sample_interval_steps)
                cif_path = workdir / f"snapshot_{frame_index:05d}.cif"
                state = simulation.context.getState(getPositions=True)
                positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                box_nm = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                _write_cif(cif_path, simulation, np.asarray(positions_nm), np.asarray(box_nm))
                outputs = self._run_zeo(executable, workdir, cif_path, frame_index)
                for key in frames:
                    frames[key].append(outputs[key])
                if bins is None:
                    bins = outputs["psd_bins"]
                elif not np.allclose(bins, outputs["psd_bins"]):
                    raise AnalysisError("Zeo++ returned inconsistent PSD bins across snapshots")
                psd_rows.append(outputs["psd"])

        arrays = {key: np.asarray(value, dtype=float) for key, value in frames.items()}
        arrays["pore_size_distribution"] = np.stack(psd_rows)
        arrays["pore_diameter_bins_angstrom"] = np.asarray(bins, dtype=float)
        self.raw_data = RawData(
            values=arrays,
            units={
                "largest_cavity_diameter_angstrom": "angstrom",
                "pore_limiting_diameter_angstrom": "angstrom",
                "accessible_path_cavity_diameter_angstrom": "angstrom",
                "accessible_surface_area_m2_per_g": "m^2/g",
                "accessible_volume_fraction": "dimensionless",
                "channel_dimensionality": "dimensionless",
                "pore_size_distribution": "1/angstrom",
                "pore_diameter_bins_angstrom": "angstrom",
            },
            metadata={
                "frames": self.n_frames,
                "probe_radius_angstrom": self.probe_radius_angstrom,
                "monte_carlo_samples": self.monte_carlo_samples,
                "zeo_executable": executable,
                "high_accuracy": self.high_accuracy,
            },
        )
        return self

    def _run_zeo(
        self, executable: str, workdir: Path, cif_path: Path, frame_index: int
    ) -> dict[str, Any]:
        stem = f"frame_{frame_index:05d}"
        res_path = workdir / f"{stem}.res"
        sa_path = workdir / f"{stem}.sa"
        vol_path = workdir / f"{stem}.vol"
        psd_path = workdir / f"{stem}.psd"
        chan_path = workdir / f"{stem}.chan"
        radius = f"{self.probe_radius_angstrom:.8g}"
        command = [executable]
        if self.high_accuracy:
            command.append("-ha")
        command += [
            "-res", str(res_path),
            "-sa", radius, radius, str(self.monte_carlo_samples), str(sa_path),
            "-vol", radius, radius, str(self.monte_carlo_samples), str(vol_path),
            "-psd", radius, radius, str(self.monte_carlo_samples), str(psd_path),
            "-chan", radius, str(chan_path),
            str(cif_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InvalidSimulationError(f"could not run Zeo++: {exc}") from exc
        if completed.returncode != 0:
            raise AnalysisError(
                "Zeo++ failed for snapshot "
                f"{frame_index} (exit {completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        try:
            res_tokens = res_path.read_text().split()
            # The first token is the output filename, followed by Di, Df, Dif.
            cavity, free, path_cavity = map(float, res_tokens[-3:])
            sa_text = sa_path.read_text()
            vol_text = vol_path.read_text()
            surface = _labelled_float(sa_text, r"ASA_m\^2/g")
            volume_fraction = _labelled_float(vol_text, "AV_Volume_fraction")
            psd_file = _find_psd_file(workdir, psd_path)
            psd_bins, psd = _parse_psd(psd_file)
            chan_text = chan_path.read_text()
            dimension_match = re.search(r"dimensionality\s+(\d+)", chan_text)
            dimensionality = float(dimension_match.group(1)) if dimension_match else 0.0
        except (OSError, ValueError, IndexError) as exc:
            raise AnalysisError(f"could not parse Zeo++ output for snapshot {frame_index}: {exc}") from exc
        return {
            "largest_cavity_diameter_angstrom": cavity,
            "pore_limiting_diameter_angstrom": free,
            "accessible_path_cavity_diameter_angstrom": path_cavity,
            "accessible_surface_area_m2_per_g": surface,
            "accessible_volume_fraction": volume_fraction,
            "channel_dimensionality": dimensionality,
            "psd_bins": psd_bins,
            "psd": psd,
        }

    def analyze(self) -> PorePropertiesResult:
        if self.raw_data is None:
            raise InvalidSimulationError("call run(simulation) before analyze()")
        raw = self.raw_data
        scalar_keys = (
            "largest_cavity_diameter_angstrom",
            "pore_limiting_diameter_angstrom",
            "accessible_path_cavity_diameter_angstrom",
            "accessible_surface_area_m2_per_g",
            "accessible_volume_fraction",
            "channel_dimensionality",
        )
        means = {key: float(np.mean(raw.values[key])) for key in scalar_keys}
        errors = {key: standard_error(raw.values[key]) for key in scalar_keys}
        return PorePropertiesResult(
            largest_cavity_diameter_angstrom=means["largest_cavity_diameter_angstrom"],
            pore_limiting_diameter_angstrom=means["pore_limiting_diameter_angstrom"],
            accessible_path_cavity_diameter_angstrom=means[
                "accessible_path_cavity_diameter_angstrom"
            ],
            accessible_surface_area_m2_per_g=means["accessible_surface_area_m2_per_g"],
            accessible_volume_fraction=means["accessible_volume_fraction"],
            channel_dimensionality=means["channel_dimensionality"],
            pore_diameter_bins_angstrom=raw.values["pore_diameter_bins_angstrom"],
            pore_size_distribution=np.mean(raw.values["pore_size_distribution"], axis=0),
            standard_errors=errors,
            raw_data=raw,
        )


def _write_cif(path: Path, simulation: Any, positions_nm: NDArray, box_nm: NDArray) -> None:
    topology = simulation.topology
    atoms = list(topology.atoms())
    system = simulation.system
    if len(atoms) > len(positions_nm):
        raise InvalidSimulationError("OpenMM Topology has more atoms than Context particles")
    if np.linalg.det(box_nm) <= 0:
        raise InvalidSimulationError("Zeo++ requires a right-handed periodic unit cell")
    lengths = np.linalg.norm(box_nm, axis=1)
    angles = (
        _angle_deg(box_nm[1], box_nm[2]),
        _angle_deg(box_nm[0], box_nm[2]),
        _angle_deg(box_nm[0], box_nm[1]),
    )
    fractional = np.asarray(positions_nm[: len(atoms)]) @ np.linalg.inv(box_nm)
    fractional %= 1.0
    rows = []
    for index, atom in enumerate(atoms):
        if system.isVirtualSite(index):
            continue
        if atom.element is None:
            raise InvalidSimulationError(
                f"atom {atom.name!r} has no element; Zeo++ needs element types or explicit radii"
            )
        symbol = atom.element.symbol
        fx, fy, fz = fractional[index]
        rows.append(f"{symbol}{index + 1} {symbol} {fx:.10f} {fy:.10f} {fz:.10f} 1")
    lines = [
        "data_openmm_snapshot",
        f"_cell_length_a {lengths[0] * 10.0:.10f}",
        f"_cell_length_b {lengths[1] * 10.0:.10f}",
        f"_cell_length_c {lengths[2] * 10.0:.10f}",
        f"_cell_angle_alpha {angles[0]:.10f}",
        f"_cell_angle_beta {angles[1]:.10f}",
        f"_cell_angle_gamma {angles[2]:.10f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        *rows,
        "",
    ]
    path.write_text("\n".join(lines))


def _angle_deg(a: NDArray, b: NDArray) -> float:
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _labelled_float(text: str, label: str) -> float:
    match = re.search(rf"{label}:\s*([-+0-9.eE]+)", text)
    if match is None:
        raise ValueError(f"missing {label} field")
    return float(match.group(1))


def _find_psd_file(workdir: Path, requested_path: Path) -> Path:
    candidates = [requested_path, workdir / f"{requested_path.name}_histo"]
    candidates.extend(sorted(workdir.glob("*.psd_histo")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Zeo++ PSD histogram output was not created")


def _parse_psd(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    bins: list[float] = []
    derivative: list[float] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            bins.append(float(fields[0]))
            derivative.append(float(fields[3]))
        except ValueError:
            continue
    if not bins:
        raise ValueError("PSD histogram contains no numeric rows")
    return np.asarray(bins, dtype=float), np.asarray(derivative, dtype=float)
