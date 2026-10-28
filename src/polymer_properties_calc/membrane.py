"""Membrane gas-transport runners, literature presets, and analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from openmm import (
    Context,
    NonbondedForce,
    TwoParticleAverageSite,
    VerletIntegrator,
    XmlSerializer,
)
from openmm.app import Simulation, Topology, element

from ._openmm import unit
from .base import PropertyCalculator, PropertyResult, RawData, standard_error
from .calculators import SamplingProtocol
from .exceptions import AnalysisError, InvalidSimulationError


GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324
AVOGADRO = 6.02214076e23
BARRER_IN_MOL_M_PER_M2_S_PA = 3.348e-16


@dataclass(frozen=True)
class GasSitePreset:
    """One interaction site in a literature or OpenFF penetrant model."""

    name: str
    epsilon_over_k_kelvin: float
    sigma_angstrom: float
    charge_e: float
    mass_dalton: float
    position_angstrom: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def epsilon_kj_per_mol(self) -> float:
        return self.epsilon_over_k_kelvin * GAS_CONSTANT_KJ_PER_MOL_K

    @property
    def sigma_nm(self) -> float:
        return self.sigma_angstrom * 0.1


@dataclass(frozen=True)
class PenetrantPreset:
    """Nonbonded site model, optional molecular geometry, and residue aliases."""

    name: str
    model: str
    sites: tuple[GasSitePreset, ...]
    residue_names: tuple[str, ...]
    reference: str
    bond_constraints_nm: tuple[tuple[int, int, float], ...] = ()
    virtual_site: tuple[int, int, int, float, float] | None = None
    nonbonded_exceptions: tuple[tuple[int, int, float, float, float], ...] = ()
    molecular_bonds: tuple[tuple[int, int], ...] = ()
    openff_system_xml: str | None = None

    @property
    def molar_mass_g_per_mol(self) -> float:
        return float(sum(site.mass_dalton for site in self.sites))

    @classmethod
    def from_smiles(
        cls,
        name: str,
        smiles: str,
        *,
        force_field: str = "openff-2.3.0.offxml",
        residue_name: str | None = None,
    ) -> "PenetrantPreset":
        """Parameterize an organic penetrant with OpenFF and extract OpenMM LJ/charges.

        Requires the optional ``openff-toolkit`` and ``openff-interchange``
        packages. The host System must use compatible Lorentz-Berthelot
        nonbonded mixing to combine these atom parameters with the polymer. A
        single generated conformer is stored; Widom estimates for flexible
        penetrants should be treated as rigid-conformer estimates.
        """
        try:
            from openff.toolkit import ForceField, Molecule
        except ImportError as exc:
            raise ImportError(
                "OpenFF parameterization needs openff-toolkit and openff-interchange"
            ) from exc

        molecule = Molecule.from_smiles(smiles)
        molecule.add_hydrogens()
        molecule.generate_conformers(n_conformers=1)
        try:
            interchange = ForceField(force_field).create_interchange(molecule.to_topology())
            system = interchange.to_openmm(combine_nonbonded_forces=True)
            positions = np.asarray(
                interchange.positions.to_openmm().value_in_unit(unit.nanometer), dtype=float
            )
        except Exception as exc:
            raise ValueError(
                f"OpenFF could not parameterize {name!r} from SMILES {smiles!r}: {exc}"
            ) from exc
        nonbonded = [
            force for force in system.getForces()
            if force.__class__.__name__ == "NonbondedForce"
        ]
        if len(nonbonded) != 1:
            raise ValueError("OpenFF export must contain exactly one standard NonbondedForce")
        force = nonbonded[0]
        if system.getNumParticles() != molecule.n_atoms or positions.shape != (molecule.n_atoms, 3):
            raise ValueError("OpenFF virtual sites are not yet supported for inserted molecules")
        sites = []
        for atom_index, atom in enumerate(molecule.atoms):
            charge, sigma, epsilon = force.getParticleParameters(atom_index)
            pos_angstrom = tuple((positions[atom_index] * 10.0).tolist())
            sites.append(
                GasSitePreset(
                    name=f"{atom.element.symbol}{atom_index + 1}",
                    epsilon_over_k_kelvin=(
                        epsilon.value_in_unit(unit.kilojoule_per_mole)
                        / GAS_CONSTANT_KJ_PER_MOL_K
                    ),
                    sigma_angstrom=sigma.value_in_unit(unit.angstrom),
                    charge_e=charge.value_in_unit(unit.elementary_charge),
                    mass_dalton=system.getParticleMass(atom_index).value_in_unit(unit.dalton),
                    position_angstrom=pos_angstrom,
                )
            )
        constraints = tuple(
            (
                int(system.getConstraintParameters(index)[0]),
                int(system.getConstraintParameters(index)[1]),
                system.getConstraintParameters(index)[2].value_in_unit(unit.nanometer),
            )
            for index in range(system.getNumConstraints())
        )
        exceptions = []
        for index in range(force.getNumExceptions()):
            atom1, atom2, charge_product, sigma, epsilon = force.getExceptionParameters(index)
            exceptions.append(
                (
                    int(atom1), int(atom2),
                    charge_product.value_in_unit(unit.elementary_charge**2),
                    sigma.value_in_unit(unit.nanometer),
                    epsilon.value_in_unit(unit.kilojoule_per_mole),
                )
            )
        return cls(
            name=name.upper(),
            model=f"OpenFF {force_field}",
            sites=tuple(sites),
            residue_names=((residue_name or name).upper(),),
            reference="https://docs.openforcefield.org/projects/interchange/en/latest/",
            bond_constraints_nm=constraints,
            nonbonded_exceptions=tuple(exceptions),
            molecular_bonds=tuple(
                (int(bond.atom1_index), int(bond.atom2_index)) for bond in molecule.bonds
            ),
            openff_system_xml=XmlSerializer.serialize(system),
        )


TRAPPE_CH4 = PenetrantPreset(
    name="CH4",
    model="TraPPE united-atom, single-site rigid methane",
    sites=(GasSitePreset("CH4", 148.0, 3.73, 0.0, 16.043),),
    residue_names=("CH4", "MET", "METHANE"),
    reference="https://doi.org/10.1002/adts.201900135",
)

TRAPPE_CO2 = PenetrantPreset(
    name="CO2",
    model="TraPPE rigid three-site carbon dioxide",
    sites=(
        GasSitePreset("C", 27.0, 2.80, 0.70, 12.011),
        GasSitePreset("O1", 79.0, 3.05, -0.35, 15.999, (0.0, 0.0, 1.16)),
        GasSitePreset("O2", 79.0, 3.05, -0.35, 15.999, (0.0, 0.0, -1.16)),
    ),
    residue_names=("CO2", "CARBONDIOXIDE"),
    reference="https://doi.org/10.1002/adts.201900135",
    bond_constraints_nm=((0, 1, 0.116), (0, 2, 0.116), (1, 2, 0.232)),
)

TRAPPE_N2 = PenetrantPreset(
    name="N2",
    model="TraPPE rigid three-site nitrogen with a central quadrupole site",
    sites=(
        GasSitePreset("N1", 36.0, 3.31, -0.482, 14.007, (0.0, 0.0, 0.55)),
        GasSitePreset("N2", 36.0, 3.31, -0.482, 14.007, (0.0, 0.0, -0.55)),
        GasSitePreset("M", 0.0, 0.0, 0.964, 0.0),
    ),
    residue_names=("N2", "NITROGEN"),
    reference="https://doi.org/10.1002/adts.201900135",
    bond_constraints_nm=((0, 1, 0.110),),
    virtual_site=(2, 0, 1, 0.5, 0.5),
)


LITERATURE_PENETRANTS = {gas.name: gas for gas in (TRAPPE_CH4, TRAPPE_CO2, TRAPPE_N2)}


@dataclass(frozen=True)
class MembraneLiteraturePreset:
    """Protocol conditions reported for one published membrane study."""

    name: str
    temperature_kelvin: float
    mixed_feed_pressure_bar: float
    feed_mole_fractions: tuple[tuple[str, float], ...]
    pure_gas_pressure_range_bar: tuple[float, float]
    diffusion_production_time_ns: tuple[float, float]
    polymer_force_field: str
    gas_models: tuple[tuple[str, str], ...]
    reference: str

    def diffusion_protocol(
        self,
        step_size_ps: float,
        sample_interval_ps: float = 1.0,
        production_time_ns: float | None = None,
        equilibration_steps: int = 0,
    ) -> SamplingProtocol:
        """Build a sampling protocol using this study's reported trajectory length.

        The default uses the shorter reported production trajectory, so the
        protocol is explicit and reproducible while remaining compatible with
        the user's integrator timestep.
        """
        production_ns = (
            self.diffusion_production_time_ns[0]
            if production_time_ns is None
            else float(production_time_ns)
        )
        if not np.isfinite(step_size_ps) or step_size_ps <= 0:
            raise ValueError("step_size_ps must be positive and finite")
        if not np.isfinite(sample_interval_ps) or sample_interval_ps <= 0:
            raise ValueError("sample_interval_ps must be positive and finite")
        if not self.diffusion_production_time_ns[0] <= production_ns <= self.diffusion_production_time_ns[1]:
            raise ValueError(
                "production_time_ns must lie within the literature range "
                f"{self.diffusion_production_time_ns} ns"
            )
        steps_per_sample = max(1, int(round(sample_interval_ps / step_size_ps)))
        actual_interval_ps = steps_per_sample * step_size_ps
        samples = int(round(production_ns * 1000.0 / actual_interval_ps))
        if samples < 5:
            raise ValueError("protocol must produce at least five trajectory frames")
        return SamplingProtocol(
            equilibration_steps=equilibration_steps,
            samples=samples,
            steps_per_sample=steps_per_sample,
        )


TRIPTYCENE_POLYIMIDE_N2_CH4_2023 = MembraneLiteraturePreset(
    name="Tanis et al. 2023: triptycene polyimide, N2/CH4",
    temperature_kelvin=338.5,
    mixed_feed_pressure_bar=65.5,
    feed_mole_fractions=(("CH4", 2.0 / 3.0), ("N2", 1.0 / 3.0)),
    pure_gas_pressure_range_bar=(0.0, 70.0),
    diffusion_production_time_ns=(15.0, 20.0),
    polymer_force_field="TRIPOS 5.2",
    gas_models=(("N2", "Vrabec 2CLJQ"), ("CH4", "Yin–MacKerell rigid five-site")),
    reference="https://doi.org/10.3390/polym15183811",
)


class MembraneGasTransportRunner:
    """Create a new Simulation containing randomly inserted gas molecules.

    OpenMM cannot change the particle count of a System after its Context has
    been created. ``prepare`` therefore returns a replacement Simulation;
    callers should pass that returned object to the MSD calculator.
    """

    def __init__(
        self,
        penetrant: str | PenetrantPreset,
        molecule_count: int,
        *,
        seed: int | None = None,
        minimum_distance_scale: float = 0.82,
        max_attempts_per_molecule: int = 20_000,
        equilibration_steps: int = 0,
    ) -> None:
        if isinstance(penetrant, str):
            try:
                self.preset = LITERATURE_PENETRANTS[penetrant.upper()]
            except KeyError as exc:
                raise ValueError(f"unknown gas {penetrant!r}") from exc
        else:
            self.preset = penetrant
        if molecule_count < 1:
            raise ValueError("molecule_count must be positive")
        if not 0 < minimum_distance_scale < 1:
            raise ValueError("minimum_distance_scale must be between 0 and 1")
        if max_attempts_per_molecule < 1:
            raise ValueError("max_attempts_per_molecule must be positive")
        if equilibration_steps < 0:
            raise ValueError("equilibration_steps must be nonnegative")
        self.molecule_count = int(molecule_count)
        self.rng = np.random.default_rng(seed)
        self.minimum_distance_scale = float(minimum_distance_scale)
        self.max_attempts_per_molecule = int(max_attempts_per_molecule)
        self.equilibration_steps = int(equilibration_steps)

    def prepare(self, simulation: Any) -> Simulation:
        """Clone and extend a prepared periodic OpenMM simulation with gas."""
        _validate_insertable_simulation(simulation)
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getParameters=True
        )
        old_positions = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float
        )

        old_velocities = np.asarray(
            state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
            dtype=float,
        )
        box = np.asarray(
            state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=float
        )
        new_system = XmlSerializer.deserialize(XmlSerializer.serialize(simulation.system))
        new_topology, original_atoms = _clone_topology(simulation.topology)
        nonbonded = _find_standard_nonbonded_force(new_system)
        old_particle_count = simulation.system.getNumParticles()
        if len(original_atoms) != old_particle_count:
            raise InvalidSimulationError(
                "Simulation topology atom count must match System particle count"
            )
        host_sigma = np.asarray([
            nonbonded.getParticleParameters(i)[1].value_in_unit(unit.nanometer)
            for i in range(old_particle_count)
        ])
        host_epsilon = np.asarray([
            nonbonded.getParticleParameters(i)[2].value_in_unit(unit.kilojoule_per_mole)
            for i in range(old_particle_count)
        ])
        topology_atoms = list(original_atoms)
        added_positions: list[np.ndarray] = []
        added_masses: list[float] = []
        residue_names = self.preset.residue_names
        residue_name = residue_names[0] if residue_names else self.preset.name
        gas_chain = new_topology.addChain()

        for molecule_index in range(self.molecule_count):
            molecule_positions = self._place_one(
                box,
                old_positions,
                host_sigma,
                host_epsilon,
                added_positions,
                new_system,
                nonbonded,
                old_particle_count,
            )
            residue = new_topology.addResidue(residue_name, gas_chain)
            first_particle = new_system.getNumParticles()
            atom_indices = []
            local_sites = self.preset.sites
            for site_index, site in enumerate(local_sites):
                particle = new_system.addParticle(site.mass_dalton * unit.dalton)
                atom_indices.append(particle)
                added_positions.append(molecule_positions[site_index])
                added_masses.append(site.mass_dalton)
                symbol = site.name[0].upper()
                atom_element = None if site.mass_dalton == 0 else element.get_by_symbol(symbol)
                atom = new_topology.addAtom(site.name, atom_element, residue)
                topology_atoms.append(atom)
                sigma_nm = site.sigma_nm if site.sigma_nm > 0 else 0.3
                nonbonded.addParticle(
                    site.charge_e * unit.elementary_charge,
                    sigma_nm * unit.nanometer,
                    site.epsilon_kj_per_mol * unit.kilojoule_per_mole,
                )
            if self.preset.virtual_site is not None:
                virtual_index, parent1, parent2, weight1, weight2 = self.preset.virtual_site
                new_system.setVirtualSite(
                    atom_indices[virtual_index],
                    TwoParticleAverageSite(
                        atom_indices[parent1], atom_indices[parent2], weight1, weight2
                    ),
                )
            constraints = self.preset.bond_constraints_nm
            for site1, site2, distance_nm in constraints:
                new_system.addConstraint(
                    atom_indices[site1], atom_indices[site2], distance_nm * unit.nanometer
                )
            bonds = self.preset.molecular_bonds or tuple(
                (i, j) for i, j, _ in constraints
            )
            for site1, site2 in bonds:
                new_topology.addBond(
                    topology_atoms[atom_indices[site1]], topology_atoms[atom_indices[site2]]
                )
            exceptions = self.preset.nonbonded_exceptions
            if not exceptions and len(atom_indices) > 1:
                exceptions = tuple(
                    (i, j, 0.0, 0.3, 0.0)
                    for i in range(len(atom_indices))
                    for j in range(i + 1, len(atom_indices))
                )
            for site1, site2, charge_product, sigma_nm, epsilon_kj in exceptions:
                nonbonded.addException(
                    atom_indices[site1], atom_indices[site2],
                    charge_product * unit.elementary_charge**2,
                    sigma_nm * unit.nanometer,
                    epsilon_kj * unit.kilojoule_per_mole,
                )
            if self.preset.openff_system_xml is not None:
                _copy_openff_valence_forces(
                    XmlSerializer.deserialize(self.preset.openff_system_xml),
                    new_system,
                    first_particle,
                )
            if new_system.getNumParticles() != first_particle + len(local_sites):
                raise InvalidSimulationError("internal error while extending OpenMM System")

        new_system.setDefaultPeriodicBoxVectors(
            *(vector * unit.nanometer for vector in box)
        )
        new_topology.setPeriodicBoxVectors(
            tuple(vector * unit.nanometer for vector in box)
        )
        integrator = XmlSerializer.deserialize(XmlSerializer.serialize(simulation.integrator))
        platform = simulation.context.getPlatform()
        properties = {
            name: platform.getPropertyValue(simulation.context, name)
            for name in platform.getPropertyNames()
        }
        prepared = Simulation(new_topology, new_system, integrator, platform, properties)
        prepared.context.setPeriodicBoxVectors(*(v * unit.nanometer for v in box))
        prepared.context.setPositions(
            np.vstack((old_positions, np.asarray(added_positions))) * unit.nanometer
        )
        if len(added_positions):
            temperature = _simulation_temperature_for_membrane(simulation)
            gas_velocities = np.zeros((len(added_positions), 3), dtype=float)
            for index, mass in enumerate(added_masses):
                if mass > 0:
                    sigma_v = np.sqrt(GAS_CONSTANT_KJ_PER_MOL_K * temperature / mass)
                    gas_velocities[index] = self.rng.normal(0.0, sigma_v, 3)
            prepared.context.setVelocities(
                np.vstack((old_velocities, gas_velocities)) * unit.nanometer / unit.picosecond
            )
        prepared.context.setTime(state.getTime())
        for name, value in state.getParameters().items():
            if name in prepared.context.getParameters():
                prepared.context.setParameter(name, value)
        if self.equilibration_steps:
            prepared.step(self.equilibration_steps)
        return prepared

    def _place_one(
        self,
        box: np.ndarray,
        host_positions: np.ndarray,
        host_sigma: np.ndarray,
        host_epsilon: np.ndarray,
        inserted_positions: list[np.ndarray],
        system: Any,
        nonbonded: Any,
        old_particle_count: int,
    ) -> np.ndarray:
        del system, nonbonded, old_particle_count
        inverse_box = np.linalg.inv(box)
        sites = self.preset.sites
        local = np.asarray([site.position_angstrom for site in sites], dtype=float) * 0.1
        gas_sigma = np.asarray([site.sigma_nm for site in sites])
        gas_epsilon = np.asarray([site.epsilon_kj_per_mol for site in sites])
        host_fractional = host_positions @ inverse_box
        inserted_array = np.asarray(inserted_positions, dtype=float).reshape((-1, 3))
        inserted_fractional = inserted_array @ inverse_box if inserted_array.size else inserted_array
        for _ in range(self.max_attempts_per_molecule):
            rotation = _random_rotation(self.rng)
            candidate = local @ rotation + self.rng.random(3) @ box
            if not _candidate_has_clearance(
                candidate, gas_sigma, gas_epsilon, host_fractional, host_sigma, host_epsilon,
                inverse_box, box, self.minimum_distance_scale,
            ):
                continue
            if inserted_array.size and not _candidate_has_clearance(
                candidate, gas_sigma, gas_epsilon, inserted_fractional, gas_sigma, gas_epsilon,
                inverse_box, box, self.minimum_distance_scale,
            ):
                continue
            return candidate
        raise InvalidSimulationError(
            f"could not place penetrant molecule without an initial overlap; "
            f"placed {len(inserted_positions) // len(sites)} of {self.molecule_count}"
        )


@dataclass(frozen=True)
class WidomLiteraturePreset:
    """Published Widom sampling target; frame allocation remains user-selectable."""

    name: str
    temperature_kelvin: float
    trajectory_window_ns: float
    total_insertions_per_gas: int
    reference: str


PHBV_WIDOM_2023 = WidomLiteraturePreset(
    name="PHBV membrane study (last 5 ns; 10^6 insertions per gas)",
    temperature_kelvin=298.0,
    trajectory_window_ns=5.0,
    total_insertions_per_gas=1_000_000,
    reference="https://doi.org/10.3390/polym15071805",
)


@dataclass(frozen=True)
class WidomSolubilityResult(PropertyResult):
    excess_chemical_potential_kj_per_mol: float = 0.0
    solubility_mol_per_m3_pa: float = 0.0
    penetrant: str = ""


class WidomInsertionRunner(PropertyCalculator):
    """Run MD snapshots and calculate infinite-dilution solubility by Widom insertion.

    Energies are evaluated by an OpenMM NonbondedForce cloned from the supplied
    System, including its configured electrostatics method. Only standard
    NonbondedForce host models are supported by this first implementation.
    """

    def __init__(
        self,
        penetrant: str | PenetrantPreset,
        temperature: float,
        protocol: SamplingProtocol,
        insertions_per_frame: int = 1000,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(penetrant, str):
            try:
                self.preset = LITERATURE_PENETRANTS[penetrant.upper()]
            except KeyError as exc:
                raise ValueError(f"unknown penetrant {penetrant!r}") from exc
        else:
            self.preset = penetrant
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and in K")
        if insertions_per_frame < 1:
            raise ValueError("insertions_per_frame must be positive")
        self.temperature = float(temperature)
        self.protocol = protocol
        self.insertions_per_frame = int(insertions_per_frame)
        self.rng = np.random.default_rng(seed)
        self.literature_preset: WidomLiteraturePreset | None = None

    @classmethod
    def from_literature(
        cls,
        simulation: Any,
        penetrant: str | PenetrantPreset,
        *,
        literature: WidomLiteraturePreset = PHBV_WIDOM_2023,
        frame_count: int = 100,
        equilibration_steps: int = 0,
        seed: int | None = None,
    ) -> "WidomInsertionRunner":
        """Use the study's trajectory window and total insertion target.

        The paper specifies the trajectory segment and total number of
        insertions but not a universal frames-per-insertion allocation. This
        factory distributes them evenly over ``frame_count`` snapshots.
        """
        if frame_count < 2:
            raise ValueError("frame_count must be at least 2")
        try:
            step_size_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        except Exception as exc:
            raise InvalidSimulationError("an initialized Simulation timestep is required") from exc
        samples = int(frame_count)
        target_interval_ps = literature.trajectory_window_ns * 1000.0 / samples
        steps_per_sample = max(1, int(round(target_interval_ps / step_size_ps)))
        protocol = SamplingProtocol(
            equilibration_steps=equilibration_steps,
            samples=samples,
            steps_per_sample=steps_per_sample,
        )
        total_per_gas = literature.total_insertions_per_gas
        insertions_per_frame = max(1, int(np.ceil(total_per_gas / samples)))
        if isinstance(penetrant, str):
            try:
                preset = LITERATURE_PENETRANTS[penetrant.upper()]
            except KeyError as exc:
                raise ValueError(f"unknown penetrant {penetrant!r}") from exc
        else:
            preset = penetrant
        runner = cls(
            preset,
            temperature=literature.temperature_kelvin,
            protocol=protocol,
            insertions_per_frame=insertions_per_frame,
            seed=seed,
        )
        runner.literature_preset = literature
        return runner

    def _collect(self, simulation: Any):
        _validate_insertable_simulation(simulation)
        widom_context, integrator, lambda_name, host_scale_name, probe_local = _create_widom_context(
            simulation, self.preset
        )
        if self.protocol.equilibration_steps:
            simulation.step(self.protocol.equilibration_steps)
        delta_u = np.empty(
            (self.protocol.samples, self.insertions_per_frame), dtype=float
        )
        volumes = np.empty(self.protocol.samples, dtype=float)
        for frame in range(self.protocol.samples):
            simulation.step(self.protocol.steps_per_sample)
            state = simulation.context.getState(getPositions=True)
            positions = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float
            )
            box = np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=float
            )
            volume = abs(float(np.linalg.det(box)))
            if not np.isfinite(volume) or volume <= 0:
                raise InvalidSimulationError("Widom insertion requires a valid periodic box")
            volumes[frame] = volume
            widom_context.setPeriodicBoxVectors(*(v * unit.nanometer for v in box))
            widom_context.setParameter(lambda_name, 0.0)
            widom_context.setParameter(host_scale_name, 1.0)
            widom_context.setPositions(
                np.vstack((positions, probe_local)) * unit.nanometer
            )
            widom_context.computeVirtualSites()
            host_energy = widom_context.getState(
                getEnergy=True, groups=1 << 31
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            widom_context.setParameter(host_scale_name, 0.0)
            zero_energy = widom_context.getState(
                getEnergy=True, groups=1 << 31
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            for insertion in range(self.insertions_per_frame):
                probe = _random_probe_positions(self.preset, box, self.rng)
                widom_context.setPositions(
                    np.vstack((positions, probe)) * unit.nanometer
                )
                widom_context.computeVirtualSites()
                widom_context.setParameter(lambda_name, 1.0)
                widom_context.setParameter(host_scale_name, 1.0)
                energy = widom_context.getState(
                    getEnergy=True, groups=1 << 31
                ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                widom_context.setParameter(host_scale_name, 0.0)
                probe_energy = widom_context.getState(
                    getEnergy=True, groups=1 << 31
                ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                delta_u[frame, insertion] = (
                    energy - host_energy - probe_energy + zero_energy
                )
                if not np.isfinite(delta_u[frame, insertion]):
                    raise AnalysisError(
                        "non-finite Widom insertion energy; check overlaps and force-field parameters"
                    )
                widom_context.setParameter(lambda_name, 0.0)
                widom_context.setParameter(host_scale_name, 1.0)
        del widom_context, integrator
        return (
            {"insertion_energy": delta_u, "volume": volumes},
            {"insertion_energy": "kJ/mol", "volume": "nm^3"},
            {
                "penetrant": self.preset.name,
                "penetrant_model": self.preset.model,
                "temperature_K": self.temperature,
                "frame_count": self.protocol.samples,
                "insertions_per_frame": self.insertions_per_frame,
                "total_insertions": self.protocol.samples * self.insertions_per_frame,
                "sample_interval_ps": float(
                    simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
                    * self.protocol.steps_per_sample
                ),
                "literature_protocol": (
                    self.literature_preset.name if self.literature_preset else None
                ),
                "literature_protocol_reference": (
                    self.literature_preset.reference if self.literature_preset else None
                ),
                "openmm_nonbonded_method": _find_standard_nonbonded_force(
                    simulation.system
                ).getNonbondedMethod(),
                "parameter_reference": self.preset.reference,
            },
        )

    def _analyze(self, raw_data: RawData) -> WidomSolubilityResult:
        energy = raw_data.values["insertion_energy"]
        volume = raw_data.values["volume"]
        rt = GAS_CONSTANT_KJ_PER_MOL_K * self.temperature
        log_weights = -energy / rt
        log_volume = np.log(volume)[:, None]
        log_weighted_sum = _logsumexp(log_weights + log_volume)
        log_mean_weight = log_weighted_sum - np.log(np.sum(volume) * energy.shape[1])
        mu_ex = -rt * log_mean_weight
        log_solubility = log_mean_weight - np.log(8.31446261815324 * self.temperature)
        if not np.isfinite(log_mean_weight):
            raise AnalysisError("all Widom Boltzmann weights are zero or non-finite")
        if log_solubility > np.log(np.finfo(float).max):
            raise AnalysisError("Widom solubility overflow; inspect force-field parameters and sampling")
        solubility = float(np.exp(log_solubility))
        per_frame_weights = np.exp(
            np.clip(_logsumexp(log_weights, axis=1) - np.log(energy.shape[1]), -745, 700)
        )
        frame_estimates = per_frame_weights * volume / np.mean(volume)
        frame_solubilities = frame_estimates / (8.31446261815324 * self.temperature)
        return WidomSolubilityResult(
            value=solubility,
            unit="mol/(m^3 Pa)",
            standard_error=standard_error(frame_solubilities),
            excess_chemical_potential_kj_per_mol=float(mu_ex),
            solubility_mol_per_m3_pa=solubility,
            penetrant=self.preset.name,
            details={
                "log_mean_boltzmann_weight": float(log_mean_weight),
                "mean_excess_chemical_potential_kj_per_mol": float(mu_ex),
                "mean_volume_nm3": float(np.mean(volume)),
                "frames": int(energy.shape[0]),
                "insertions_per_frame": int(energy.shape[1]),
                "total_insertions": int(energy.size),
                "estimate": "infinite-dilution Henry solubility from Widom insertion",
                "volume_weighting": "NPT Widom estimator, weighted by instantaneous volume",
                "parameter_model": self.preset.model,
                "literature_protocol": (
                    self.literature_preset.name if self.literature_preset else None
                ),
            },
        )


def _find_standard_nonbonded_force(system: Any) -> Any:
    forces = list(system.getForces())
    nonbonded = [force for force in forces if isinstance(force, NonbondedForce)]
    unsupported = [
        force.__class__.__name__ for force in forces
        if force.__class__.__name__ in {
            "CustomNonbondedForce", "CustomGBForce", "AmoebaVdwForce", "AmoebaMultipoleForce"
        }
    ]
    if unsupported:
        raise InvalidSimulationError(
            "gas insertion currently supports standard NonbondedForce interactions; "
            f"unsupported force(s): {', '.join(unsupported)}"
        )
    if len(nonbonded) != 1:
        raise InvalidSimulationError(
            "gas insertion requires exactly one standard OpenMM NonbondedForce"
        )
    return nonbonded[0]


def _create_widom_context(simulation: Any, preset: PenetrantPreset):
    """Build a private OpenMM context with one alchemically switched probe."""
    system = XmlSerializer.deserialize(XmlSerializer.serialize(simulation.system))
    force = _find_standard_nonbonded_force(system)
    if force.getNumParticleParameterOffsets() or force.getNumGlobalParameters():
        raise InvalidSimulationError(
            "Widom insertion currently requires a NonbondedForce without pre-existing "
            "global parameters or particle parameter offsets"
        )
    existing_particle_count = force.getNumParticles()
    host_scale_name = "widom_host_scale"
    parameter_names = {
        force.getGlobalParameterName(index)
        for index in range(force.getNumGlobalParameters())
    }
    lambda_name = "widom_probe_scale"
    suffix = 1
    while lambda_name in parameter_names:
        lambda_name = f"widom_probe_scale_{suffix}"
        suffix += 1
    force.addGlobalParameter(lambda_name, 0.0)
    force.addGlobalParameter(host_scale_name, 1.0)
    force.setForceGroup(31)

    for index in range(existing_particle_count):
        charge, sigma, epsilon = force.getParticleParameters(index)
        force.setParticleParameters(
            index,
            0.0 * unit.elementary_charge,
            sigma,
            0.0 * unit.kilojoule_per_mole,
        )
        force.addParticleParameterOffset(
            host_scale_name,
            index,
            charge,
            0.0 * unit.nanometer,
            epsilon,
        )

    probe_start = system.getNumParticles()
    positions = []
    for site in preset.sites:
        index = system.addParticle(site.mass_dalton * unit.dalton)
        positions.append(np.asarray(site.position_angstrom, dtype=float) * 0.1)
        sigma_nm = site.sigma_nm if site.sigma_nm > 0 else 0.3
        force.addParticle(
            0.0 * unit.elementary_charge,
            sigma_nm * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
        force.addParticleParameterOffset(
            lambda_name,
            index,
            site.charge_e * unit.elementary_charge,
            0.0 * unit.nanometer,
            site.epsilon_kj_per_mol * unit.kilojoule_per_mole,
        )
    if preset.virtual_site is not None:
        virtual_index, parent1, parent2, weight1, weight2 = preset.virtual_site
        system.setVirtualSite(
            probe_start + virtual_index,
            TwoParticleAverageSite(
                probe_start + parent1, probe_start + parent2, weight1, weight2
            ),
        )
    exception_map = {
        (min(i, j), max(i, j)): (charge_product, sigma, epsilon)
        for i, j, charge_product, sigma, epsilon in preset.nonbonded_exceptions
    }
    for site1 in range(len(preset.sites)):
        for site2 in range(site1 + 1, len(preset.sites)):
            charge_product, sigma_nm, epsilon_kj = exception_map.get(
                (site1, site2), (0.0, 0.3, 0.0)
            )
            force.addException(
                probe_start + site1,
                probe_start + site2,
                charge_product * unit.elementary_charge**2,
                sigma_nm * unit.nanometer,
                epsilon_kj * unit.kilojoule_per_mole,
            )
    box_nm = np.asarray(
        simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
        .value_in_unit(unit.nanometer),
        dtype=float,
    )
    box = tuple(vector * unit.nanometer for vector in box_nm)
    system.setDefaultPeriodicBoxVectors(*box)
    integrator = VerletIntegrator(0.001 * unit.picosecond)
    platform = simulation.context.getPlatform()
    properties = {
        name: platform.getPropertyValue(simulation.context, name)
        for name in platform.getPropertyNames()
    }
    context = Context(system, integrator, platform, properties)
    context.setPeriodicBoxVectors(*box)
    for name, value in simulation.context.getParameters().items():
        if name in context.getParameters() and name != lambda_name:
            context.setParameter(name, value)
    context.setParameter(lambda_name, 0.0)
    initial_probe = np.asarray(positions, dtype=float)
    initial_probe += (np.asarray([0.5, 0.5, 0.5]) @ box_nm)
    return context, integrator, lambda_name, host_scale_name, initial_probe


def _random_probe_positions(
    preset: PenetrantPreset, box: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    local = np.asarray([site.position_angstrom for site in preset.sites], dtype=float) * 0.1
    return local @ _random_rotation(rng) + rng.random(3) @ box


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.where(finite, values - maximum, -np.inf)
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.where(finite, maximum + np.log(summed), maximum)
    if axis is None:
        return float(np.squeeze(result))
    return np.squeeze(result, axis=axis)


def _validate_insertable_simulation(simulation: Any) -> None:
    if not hasattr(simulation, "context") or not hasattr(simulation, "system"):
        raise InvalidSimulationError("expected an initialized OpenMM Simulation")
    if getattr(simulation, "topology", None) is None:
        raise InvalidSimulationError("gas insertion requires Simulation.topology")
    if simulation.system.getNumParticles() == 0:
        raise InvalidSimulationError("cannot insert gas into an empty OpenMM System")
    _find_standard_nonbonded_force(simulation.system)


def _clone_topology(topology: Any) -> tuple[Topology, list[Any]]:
    cloned = Topology()
    atom_map = {}
    for chain in topology.chains():
        new_chain = cloned.addChain(chain.id)
        for residue in chain.residues():
            new_residue = cloned.addResidue(
                residue.name, new_chain, residue.id, getattr(residue, "insertionCode", " ")
            )
            for atom in residue.atoms():
                atom_map[atom.index] = cloned.addAtom(
                    atom.name, atom.element, new_residue, atom.id
                )
    for bond in topology.bonds():
        atom1, atom2 = bond.atom1, bond.atom2
        cloned.addBond(
            atom_map[atom1.index], atom_map[atom2.index],
            type=getattr(bond, "type", None), order=getattr(bond, "order", None),
        )
    box = topology.getPeriodicBoxVectors()
    if box is not None:
        cloned.setPeriodicBoxVectors(box)
    return cloned, [atom_map[index] for index in range(len(atom_map))]


def _candidate_has_clearance(
    candidate_nm: np.ndarray,
    gas_sigma_nm: np.ndarray,
    gas_epsilon_kj: np.ndarray,
    host_fractional: np.ndarray,
    host_sigma_nm: np.ndarray,
    host_epsilon_kj: np.ndarray,
    inverse_box: np.ndarray,
    box: np.ndarray,
    scale: float,
) -> bool:
    active_host = np.flatnonzero(host_epsilon_kj > 0)
    active_gas = np.flatnonzero(gas_epsilon_kj > 0)
    if not len(active_host) or not len(active_gas):
        return True
    candidate_fractional = candidate_nm @ inverse_box
    delta_fractional = candidate_fractional[active_gas, None, :] - host_fractional[None, active_host, :]
    delta_fractional -= np.rint(delta_fractional)
    delta = delta_fractional @ box
    distances = np.linalg.norm(delta, axis=2)
    contact = 0.5 * (
        gas_sigma_nm[active_gas, None] + host_sigma_nm[None, active_host]
    ) * scale
    return bool(np.all(distances >= contact))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3-D rotation matrix from a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    q = np.asarray([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _simulation_temperature_for_membrane(simulation: Any) -> float:
    getter = getattr(simulation.integrator, "getTemperature", None)
    if getter is not None:
        value = getter()
        return float(
            value.value_in_unit(unit.kelvin)
            if hasattr(value, "value_in_unit") else value
        )
    for force in simulation.system.getForces():
        getter = getattr(force, "getDefaultTemperature", None)
        if getter is not None:
            value = getter()
            return float(
                value.value_in_unit(unit.kelvin)
                if hasattr(value, "value_in_unit") else value
            )
    raise InvalidSimulationError(
        "cannot initialize penetrant velocities: pass a thermostat/barostat with temperature"
    )


def _copy_openff_valence_forces(source_system: Any, target_system: Any, offset: int) -> None:
    """Copy the common OpenFF harmonic bond/angle/torsion terms for one molecule."""
    from openmm import HarmonicAngleForce, HarmonicBondForce, PeriodicTorsionForce

    supported = (HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce)
    for force in source_system.getForces():
        if isinstance(force, NonbondedForce) or force.__class__.__name__ == "CMMotionRemover":
            continue
        if not isinstance(force, supported):
            raise InvalidSimulationError(
                "OpenFF molecule uses an intramolecular force not yet supported for insertion: "
                f"{force.__class__.__name__}"
            )
        copied = force.__class__()
        copied.setForceGroup(force.getForceGroup())
        if isinstance(force, HarmonicBondForce):
            for index in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(index)
                copied.addBond(p1 + offset, p2 + offset, length, k)
        elif isinstance(force, HarmonicAngleForce):
            for index in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(index)
                copied.addAngle(p1 + offset, p2 + offset, p3 + offset, angle, k)
        elif isinstance(force, PeriodicTorsionForce):
            for index in range(force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(index)
                copied.addTorsion(
                    p1 + offset, p2 + offset, p3 + offset, p4 + offset,
                    periodicity, phase, k,
                )
        target_system.addForce(copied)


@dataclass(frozen=True)
class MembraneGasTransportResult(PropertyResult):
    """Diffusion, single-pressure sorption, and solution-diffusion permeability."""

    diffusion_coefficient_m2_per_s: float = 0.0
    solubility_mol_per_m3_pa: float = 0.0
    permeability_si: float = 0.0
    permeability_barrer: float = 0.0
    penetrant: str = ""


class MembraneGasTransportCalculator(PropertyCalculator):
    """Estimate penetrant D, one-point S, and P = D*S from an equilibrated cell.

    The Simulation must already contain the penetrant at its equilibrium
    loading for ``pressure_bar``. The pressure is the external gas pressure,
    not necessarily the barostat's total pressure. One state gives a secant
    solubility; glassy-polymer dual-mode sorption requires multiple pressures.
    """

    def __init__(
        self,
        penetrant: str | PenetrantPreset,
        pressure_bar: float,
        temperature: float,
        protocol: SamplingProtocol,
        residue_names: Sequence[str] | None = None,
        fit_fraction: tuple[float, float] = (0.3, 0.8),
    ) -> None:
        super().__init__()
        if isinstance(penetrant, str):
            try:
                preset = LITERATURE_PENETRANTS[penetrant.upper()]
            except KeyError as exc:
                raise ValueError(
                    f"unknown penetrant {penetrant!r}; choose from {sorted(LITERATURE_PENETRANTS)}"
                ) from exc
        else:
            preset = penetrant
        if not np.isfinite(pressure_bar) or pressure_bar <= 0:
            raise ValueError("pressure_bar must be a positive finite reservoir pressure")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive finite and in K")
        if len(fit_fraction) != 2 or not 0 <= fit_fraction[0] < fit_fraction[1] <= 1:
            raise ValueError("fit_fraction must satisfy 0 <= start < end <= 1")
        self.preset = preset
        self.pressure_bar = float(pressure_bar)
        self.temperature = float(temperature)
        self.protocol = protocol
        self.residue_names = tuple(name.upper() for name in (residue_names or preset.residue_names))
        self.fit_fraction = (float(fit_fraction[0]), float(fit_fraction[1]))
        self.study_preset: MembraneLiteraturePreset | None = None

    @classmethod
    def from_literature(
        cls,
        simulation,
        penetrant: str | PenetrantPreset,
        *,
        study: MembraneLiteraturePreset = TRIPTYCENE_POLYIMIDE_N2_CH4_2023,
        residue_names: Sequence[str] | None = None,
        sample_interval_ps: float = 1.0,
        production_time_ns: float | None = None,
        pressure_bar: float | None = None,
        temperature: float | None = None,
        equilibration_steps: int = 0,
    ) -> "MembraneGasTransportCalculator":
        """Create a calculator with conditions and trajectory duration from a study.

        Gas loading in ``simulation`` must already be equilibrated at the
        chosen pressure and temperature. This factory does not build the
        force field or insert penetrants.
        """
        try:
            step_size_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        except Exception as exc:
            raise InvalidSimulationError(
                "from_literature needs an initialized Simulation with an integrator timestep"
            ) from exc
        if isinstance(penetrant, str):
            gas_name = penetrant.upper()
        else:
            gas_name = penetrant.name.upper()
        feed = dict(study.feed_mole_fractions)
        if gas_name not in feed:
            raise ValueError(f"{gas_name} is not part of literature preset {study.name!r}")
        pressure = study.mixed_feed_pressure_bar if pressure_bar is None else pressure_bar
        # Convert total mixed-feed pressure to the penetrant partial pressure.
        if pressure_bar is None:
            pressure *= feed[gas_name]
        calculator = cls(
            penetrant=penetrant,
            pressure_bar=pressure,
            temperature=study.temperature_kelvin if temperature is None else temperature,
            protocol=study.diffusion_protocol(
                step_size_ps=step_size_ps,
                sample_interval_ps=sample_interval_ps,
                production_time_ns=production_time_ns,
                equilibration_steps=equilibration_steps,
            ),
            residue_names=residue_names,
        )
        calculator.study_preset = study
        return calculator

    def _collect(self, simulation):
        topology = getattr(simulation, "topology", None)
        if topology is None:
            raise InvalidSimulationError("membrane transport requires Simulation.topology")
        residues = [
            residue for residue in topology.residues()
            if residue.name.upper() in self.residue_names
        ]
        if not residues:
            raise InvalidSimulationError(
                f"no penetrant residues named {self.residue_names} found in Simulation.topology"
            )
        groups = [[atom.index for atom in residue.atoms()] for residue in residues]
        masses = np.asarray(
            [
                simulation.system.getParticleMass(index).value_in_unit(unit.dalton)
                for index in range(simulation.system.getNumParticles())
            ],
            dtype=float,
        )
        if any(sum(masses[group]) <= 0 for group in groups):
            raise InvalidSimulationError("each penetrant residue must have positive total mass")

        if self.protocol.equilibration_steps:
            simulation.step(self.protocol.equilibration_steps)
        frames = np.empty((self.protocol.samples, len(groups), 3), dtype=float)
        boxes = np.empty((self.protocol.samples, 3, 3), dtype=float)
        for sample in range(self.protocol.samples):
            simulation.step(self.protocol.steps_per_sample)
            state = simulation.context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            positions = np.asarray(positions, dtype=float)
            box = np.asarray(box, dtype=float)
            if abs(float(np.linalg.det(box))) <= 0:
                raise InvalidSimulationError("membrane transport requires a valid periodic box")
            inverse_box = np.linalg.inv(box)
            fractional = positions @ inverse_box
            for molecule_index, atom_indices in enumerate(groups):
                atom_indices_array = np.asarray(atom_indices, dtype=int)
                atom_fractional = fractional[atom_indices_array]
                relative = atom_fractional - atom_fractional[0]
                relative -= np.rint(relative)
                unwrapped = (atom_fractional[0] + relative) @ box
                atom_masses = masses[atom_indices_array]
                frames[sample, molecule_index] = np.average(
                    unwrapped, axis=0, weights=atom_masses
                )
            boxes[sample] = box

        # Unwrap each molecule's center of mass through the periodic cell.
        fractional_com = np.einsum("tmi,tij->tmj", frames, np.linalg.inv(boxes))
        unwrapped_fractional = np.empty_like(fractional_com)
        unwrapped_fractional[0] = fractional_com[0]
        for sample in range(1, self.protocol.samples):
            delta = fractional_com[sample] - fractional_com[sample - 1]
            delta -= np.rint(delta)
            unwrapped_fractional[sample] = unwrapped_fractional[sample - 1] + delta
        unwrapped_positions = np.einsum("tmi,tij->tmj", unwrapped_fractional, boxes)
        step_size_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        return (
            {
                "penetrant_com_positions": unwrapped_positions,
                "volume": np.asarray([abs(np.linalg.det(box)) for box in boxes]),
            },
            {"penetrant_com_positions": "nm", "volume": "nm^3"},
            {
                "penetrant": self.preset.name,
                "penetrant_parameter_record": self.preset.model,
                "residue_names": self.residue_names,
                "molecule_count": len(groups),
                "pressure_bar": self.pressure_bar,
                "temperature_K": self.temperature,
                "protocol": self.protocol,
                "sample_interval_ps": float(step_size_ps * self.protocol.steps_per_sample),
                "parameter_record_reference": self.preset.reference,
                "literature_study": self.study_preset.name if self.study_preset else None,
                "study_reported_gas_model": (
                    dict(self.study_preset.gas_models).get(self.preset.name)
                    if self.study_preset else None
                ),
                "force_field_configured_by_calculator": False,
            },
        )

    def _analyze(self, raw_data: RawData) -> MembraneGasTransportResult:
        positions = raw_data.values["penetrant_com_positions"]
        if positions.shape[0] < 5:
            raise AnalysisError("at least five trajectory frames are needed for diffusion fitting")
        msd_nm2 = _mean_squared_displacement_fft(positions)
        dt_ps = raw_data.metadata["sample_interval_ps"]
        time_ps = np.arange(msd_nm2.size, dtype=float) * dt_ps
        first = max(1, int(np.floor((msd_nm2.size - 1) * self.fit_fraction[0])))
        last = min(msd_nm2.size, int(np.ceil((msd_nm2.size - 1) * self.fit_fraction[1])) + 1)
        if last - first < 3:
            raise AnalysisError("diffusion fit window must contain at least three MSD points")
        slope_nm2_ps, intercept = np.polyfit(time_ps[first:last], msd_nm2[first:last], 1)
        if slope_nm2_ps <= 0:
            raise AnalysisError("fitted MSD slope is not positive; check the Fickian fit window")
        diffusion_m2_s = float(slope_nm2_ps * 1e-6 / 6.0)
        volume_nm3 = float(np.mean(raw_data.values["volume"]))
        molecule_count = int(raw_data.metadata["molecule_count"])
        concentration_mol_m3 = molecule_count / volume_nm3 * 1e27 / AVOGADRO
        pressure_pa = self.pressure_bar * 1e5
        solubility = concentration_mol_m3 / pressure_pa
        permeability_si = diffusion_m2_s * solubility
        permeability_barrer = permeability_si / BARRER_IN_MOL_M_PER_M2_S_PA
        fitted = slope_nm2_ps * time_ps[first:last] + intercept
        return MembraneGasTransportResult(
            value=float(permeability_barrer),
            unit="Barrer",
            diffusion_coefficient_m2_per_s=diffusion_m2_s,
            solubility_mol_per_m3_pa=float(solubility),
            permeability_si=float(permeability_si),
            permeability_barrer=float(permeability_barrer),
            penetrant=self.preset.name,
            details={
                "diffusion_fit_r_squared": _r_squared(msd_nm2[first:last], fitted),
                "diffusion_fit_time_range_ps": [float(time_ps[first]), float(time_ps[last - 1])],
                "mean_squared_displacement_nm2": msd_nm2.tolist(),
                "lag_time_ps": time_ps.tolist(),
                "mean_volume_nm3": volume_nm3,
                "penetrant_molecule_count": molecule_count,
                "concentration_mol_m3": float(concentration_mol_m3),
                "solubility_basis": "single-pressure secant; assumes equilibrium loading",
                "permeability_model": "solution-diffusion P = D*S",
                "penetrant_parameter_record": self.preset.model,
                "parameter_record_reference": self.preset.reference,
                "literature_study": self.study_preset.name if self.study_preset else None,
                "study_reported_gas_model": (
                    dict(self.study_preset.gas_models).get(self.preset.name)
                    if self.study_preset else None
                ),
                "force_field_configured_by_calculator": False,
            },
        )


@dataclass(frozen=True)
class DualModeSorptionResult:
    """Dual-mode sorption fit for glassy-polymer loading data."""

    k_d_mol_m3_bar: float
    c_h_mol_m3: float
    b_per_bar: float
    r_squared: float

    def concentration_mol_m3(self, pressure_bar: float) -> float:
        p = np.asarray(pressure_bar, dtype=float)
        return float(self.k_d_mol_m3_bar * p + self.c_h_mol_m3 * self.b_per_bar * p / (1 + self.b_per_bar * p))

    def differential_solubility_mol_m3_pa(self, pressure_bar: float) -> float:
        p = float(pressure_bar)
        derivative_per_bar = self.k_d_mol_m3_bar + self.c_h_mol_m3 * self.b_per_bar / (
            1 + self.b_per_bar * p
        ) ** 2
        return float(derivative_per_bar / 1e5)


def fit_dual_mode_sorption(
    pressures_bar: Sequence[float], concentrations_mol_m3: Sequence[float]
) -> DualModeSorptionResult:
    """Fit ``C = kD*p + CH*b*p/(1+b*p)`` by a bounded grid over ``b``."""

    pressure = np.asarray(pressures_bar, dtype=float)
    concentration = np.asarray(concentrations_mol_m3, dtype=float)
    if (
        pressure.ndim != 1
        or concentration.shape != pressure.shape
        or pressure.size < 4
        or np.any(~np.isfinite(pressure))
        or np.any(~np.isfinite(concentration))
        or np.any(pressure < 0)
        or np.unique(pressure).size < 4
    ):
        raise ValueError("at least four finite concentration measurements at distinct nonnegative pressures are required")
    best: tuple[float, float, NDArray[np.float64]] | None = None
    for b in np.logspace(-6, 3, 2_000):
        design = np.column_stack((pressure, b * pressure / (1 + b * pressure)))
        coefficients, *_ = np.linalg.lstsq(design, concentration, rcond=None)
        if np.any(coefficients < 0):
            continue
        residual = float(np.sum((concentration - design @ coefficients) ** 2))
        if best is None or residual < best[0]:
            best = residual, float(b), coefficients
    if best is None:
        raise AnalysisError("dual-mode sorption fit did not find nonnegative coefficients")
    residual, b, coefficients = best
    total = float(np.sum((concentration - np.mean(concentration)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    return DualModeSorptionResult(
        k_d_mol_m3_bar=float(coefficients[0]),
        c_h_mol_m3=float(coefficients[1]),
        b_per_bar=b,
        r_squared=r_squared,
    )


def _mean_squared_displacement_fft(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """All-time-origin MSD, averaged over molecules, using FFT correlations."""

    values = np.asarray(positions, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("positions must have shape (frames, molecules, 3)")
    frame_count, molecule_count, _ = values.shape
    fft_size = 1 << (2 * frame_count - 1).bit_length()
    transformed = np.fft.rfft(values, n=fft_size, axis=0)
    correlations = np.fft.irfft(
        transformed * np.conjugate(transformed), n=fft_size, axis=0
    )[:frame_count]
    squared = np.sum(values * values, axis=2)
    prefix = np.concatenate((np.zeros((1, molecule_count)), np.cumsum(squared, axis=0)), axis=0)
    lag = np.arange(frame_count)
    left = prefix[frame_count - lag]
    right = prefix[frame_count] - prefix[lag]
    summed = left + right - 2.0 * np.sum(correlations, axis=2)
    return np.asarray(np.sum(summed, axis=1) / ((frame_count - lag) * molecule_count), dtype=float)


def _r_squared(observed: NDArray[np.float64], fitted: NDArray[np.float64]) -> float:
    residual = float(np.sum((observed - fitted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if np.isclose(total, 0.0):
        return 1.0 if np.isclose(residual, 0.0) else 0.0
    return 1.0 - residual / total
