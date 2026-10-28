# polymer-properties-calc

A Python library for estimating properties of polymer systems from an existing
`openmm.app.Simulation`. Mechanical calculations require OpenMM 8.3 or later
with barostat support for `computeCurrentPressure`.

## Current API

```python
from polymer_properties_calc import (
    MechanicalPropertiesCalculator,
    SamplingProtocol,
)

calculator = MechanicalPropertiesCalculator(
    axis="x",
    relax_transverse_axes=True,
    strain_rate=1e8,              # engineering strain / second (s^-1)
    stretch_interval_steps=100,
    max_strain=0.1,
    elastic_strain_range=(0.0, 0.02),
    yield_method="first_deviation",
    yield_tolerance=0.05,
    transverse_pressure=1.0,      # bar
)

calculator.run(simulation)  # Stretches the cell and advances the Simulation.
print(calculator.raw_data.values["axial_pressure"])
result = calculator.analyze()
print(result.value, result.unit, result.poisson_ratio)
print(result.yield_stress, result.yield_strain)
print(result.max_tensile_stress, result.strain_at_max_tensile_stress)
```

`strain_rate` is specified in s⁻¹. The calculator converts it to the
Simulation's time units internally. The default `1e8 s⁻¹` is equivalent to the
previous default of `1e-4 ps⁻¹`.

### Heat capacity from energy fluctuations

```python
from polymer_properties_calc import HeatCapacityCalculator, SamplingProtocol

heat_capacity = HeatCapacityCalculator(
    temperature=300,
    ensemble="NPT",               # Cp; use NVT and pressure=None for Cv
    pressure=1.0,                 # bar
    protocol=SamplingProtocol(equilibration_steps=10_000, samples=2_000, steps_per_sample=20),
)
heat_capacity.run(simulation)
result = heat_capacity.analyze()
print(result.value, result.unit)  # mass-specific heat capacity, J/(g K)
print(result.molar_heat_capacity, result.molar_unit)
```

For NVT, the calculator uses fluctuations in the total energy; for NPT, it uses
enthalpy, `H = U + P V`. The zero-lag energy/enthalpy autocorrelation gives the
variance used to calculate heat capacity. The full autocorrelation function and
an estimate of the correlation time are stored in `result.details`.

### Green–Kubo thermal conductivity

OpenMM does not expose a universal microscopic heat current for every force
field. The calculator therefore accepts a callback that returns the complete
vector heat current for the specific model, including its convective and
interaction/virial terms, in `kJ nm/(mol ps)`:

```python
from polymer_properties_calc import (
    GreenKuboThermalConductivityCalculator,
    SamplingProtocol,
)

# Implement this for the force field in use. Return the extensive total heat
# current, not a heat-current density.
def heat_current_provider(simulation):
    return heat_current_xyz_kj_nm_per_mol_ps(simulation)

conductivity = GreenKuboThermalConductivityCalculator(
    temperature=300,
    heat_current_provider=heat_current_provider,
    protocol=SamplingProtocol(equilibration_steps=10_000, samples=10_000, steps_per_sample=10),
    max_lag_fraction=0.5,
)
conductivity.run(simulation)
result = conductivity.analyze()
print(result.value, result.unit)
print(result.details["conductivity_by_lag_W_m_K"])
```

The Green–Kubo integral is calculated from the heat-current autocorrelation.
`value` corresponds to the selected integration cutoff; the result as a
function of cutoff is also available to help assess whether a plateau has been
reached. Accuracy depends on the `heat_current_provider` implementation and on
having a sufficiently long equilibrium trajectory.

### Continuous glass-transition calculation

```python
from polymer_properties_calc import GlassTransitionCalculator

tg = GlassTransitionCalculator(
    temperature_range=(250, 500),  # K
    rate_k_per_ns=100,
    mode="cycles",                 # cooling, heating, heating_cooling, cycles
    cycles=3,
    pressure=1.0,                   # bar
    equilibration_steps=10_000,
    temperature_update_interval_steps=100,
    sample_interval_steps=10_000,
)

tg.run(simulation)
result = tg.analyze()
print(result.value, result.unit)
print(result.details["per_pass"])

# Optionally estimate volumetric CTE below and above Tg, in 1/K.
result_with_cte = tg.analyze(include_cte=True)
print(result_with_cte.cte_below_tg, result_with_cte.cte_above_tg)
```

Temperature changes linearly at the requested rate during each pass. OpenMM's
temperature is updated every `temperature_update_interval_steps`, and density
is recorded every `sample_interval_steps`. In `cycles` mode, one cycle heats
from the minimum to the maximum temperature and then cools back down. The
result contains a Tg estimate for each pass and separate mean estimates for
heating and cooling. With `include_cte=True`, volumetric CTE is also estimated
from the density slopes below and above Tg:
`alpha_V = -(1/rho) d(rho)/dT`.

The mechanical protocol records axial pressure as a function of strain. Young's
modulus is the stress–strain slope fitted only over `elastic_strain_range`; the
Poisson ratio is calculated over the same interval as the negative slope of
transverse strain versus axial strain. With `yield_method="first_deviation"`,
yield is the first deviation from the extrapolated elastic segment by at least
`yield_tolerance` (a fraction of the stress at the end of the elastic interval).
The alternative `yield_method="stress_drop"` identifies a preceding local
maximum when stress subsequently drops by the specified fraction. If the
criterion is not reached before `max_strain`, the yield fields are `None`. The
maximum tensile stress and its corresponding strain are reported for the full
trajectory. Stresses are returned in GPa. OpenMM's
`MonteCarloAnisotropicBarostat.computeCurrentPressure` is used; this pressure
contains a kinetic contribution and a finite-difference virial estimate.

Available calculators include `MechanicalPropertiesCalculator`,
`GlassTransitionCalculator`, `DensityCalculator`, `HeatCapacityCalculator`,
`GreenKuboThermalConductivityCalculator`, and `CTECalculator`
(`ThermalExpansionCalculator` is an alias for volumetric CTE).

### Membrane gas-transport properties

For membranes, the library estimates penetrant diffusion from the MSD slope,
solubility with Widom insertion, and permeability using the solution–diffusion
relation `P = D*S`. `LITERATURE_PENETRANTS` contains TraPPE site parameters for
CH₄, CO₂, and N₂ from published models.

A protocol based on Tanis et al. (2023) for triptycene polyimide is included:
338.5 K, a 2:1 CH₄:N₂ mixture at 65.5 bar, and a 15–20 ns gas-diffusion
trajectory. The paper used Vrabec 2CLJQ for N₂ and the five-site Yin–MacKerell
model for CH₄; these differ from the TraPPE models in the catalog. To reproduce
the study, the supplied `Simulation` must already contain the matching
parameters and equilibrated gas loading. A runner is also provided to add gas
molecules:

```python
from polymer_properties_calc import MembraneGasTransportRunner, MembraneGasTransportCalculator

loaded_simulation = MembraneGasTransportRunner(
    penetrant="CH4",
    molecule_count=20,
    seed=7,
    equilibration_steps=1_000_000,
).prepare(simulation)

transport = MembraneGasTransportCalculator.from_literature(
    loaded_simulation,
    penetrant="CH4",
    pressure_bar=1.0,  # external partial pressure, bar
)
transport.run(loaded_simulation)
result = transport.analyze()
print(result.diffusion_coefficient_m2_per_s)
```

OpenMM cannot add particles to an existing `Context`. Therefore, `prepare()`
returns a new `Simulation` with a cloned source system, added molecules, and a
copy of the source state. The molecule count is explicit and is not inferred
from pressure. The loaded polymer needs relaxation after insertion; the
required duration depends on the material. Insertion supports systems with one
standard `NonbondedForce` and Lorentz–Berthelot mixing. Automatic mixing is not
supported for custom pair interactions.

The gas models use parameters from the literature. An arbitrary organic
penetrant can be parameterized with OpenFF (optional `openff-toolkit` and
`openff-interchange` dependencies):

```python
from polymer_properties_calc import PenetrantPreset, MembraneGasTransportRunner

benzene = PenetrantPreset.from_smiles("benzene", "c1ccccc1")
loaded_simulation = MembraneGasTransportRunner(benzene, molecule_count=10).prepare(simulation)
```

Install the optional packages with `pip install '.[openff]'`.

OpenFF LJ parameters and charges are transferred to `NonbondedForce`; internal
harmonic bonds, angles, and torsions are added to the system. This assumes
compatible mixing rules between OpenFF and the polymer force field. OpenFF
virtual sites are not yet supported. One generated molecular conformation is
currently retained, so Widom estimates for flexible penetrants correspond to
that rigid conformation and do not average over internal conformational
statistics.

For solubility, the Widom runner samples MD frames and computes the energy of
random test-molecule insertions in OpenMM, including the electrostatics method
configured in the source system:

```python
from polymer_properties_calc import WidomInsertionRunner

widom = WidomInsertionRunner.from_literature(
    simulation,
    penetrant="CO2",
    frame_count=100,
    seed=7,
)
widom.run(simulation)
solubility = widom.analyze()
print(solubility.excess_chemical_potential_kj_per_mol)
print(solubility.solubility_mol_per_m3_pa)
```

The Papchenko et al. literature profile uses the final 5 ns of a trajectory
and 10⁶ insertions per gas at 298 K. The runner distributes these insertions
across the selected frames (10,000 insertions per frame for 100 frames). In a
separate study of polymeric NBR, convergence was examined across many
independent configurations and up to 10⁸ insertions per frame. This illustrates
that the required sampling depends on the polymer and gas. Check convergence
with respect to both the number of frames and insertions; literature settings
are starting points, not universal guarantees of accuracy. The calculation
estimates infinite-dilution solubility. High loading and swelling require
iterative MD–Widom, GCMC, or an osmotic ensemble.

The literature profile specifies a sampling window, not a full reproduction of
the Papchenko et al. force field: that study used PCFF for the polymer and CH₄,
and COMPASS for CO₂. The TraPPE gas parameters in this library are a separate
literature model. To reproduce the paper numerically, provide the system and
gas parameters from its model.

To follow the MSD protocol from the triptycene polyimide paper, provide the
corresponding equilibrated state with gas molecules and use
`TRIPTYCENE_POLYIMIDE_N2_CH4_2023`; its reported gas-diffusion trajectory is
15–20 ns. The preset does not replace the force-field parameters.

A single pressure point is not enough to determine a full sorption isotherm
for a glassy polymer. Given concentrations at multiple pressures, fit a
dual-mode model:

```python
from polymer_properties_calc import fit_dual_mode_sorption

fit = fit_dual_mode_sorption(
    pressures_bar=[1, 5, 10, 30, 70],
    concentrations_mol_m3=[...],
)
print(fit.k_d_mol_m3_bar, fit.c_h_mol_m3, fit.b_per_bar)
```

For an already gas-loaded box, `MembraneGasTransportCalculator` can still
estimate the secant solubility `C/p` at a given external pressure. The Widom
runner estimates the Henry coefficient in the infinite-dilution limit.
Permeability is calculated as `D*S` and returned in Barrer.

Density, Tg, and CTE calculators expect a periodic system with an existing
`MonteCarloBarostat` or `MonteCarloAnisotropicBarostat`. The mechanical
calculator replaces the existing isotropic or anisotropic barostat with an
anisotropic one. With `relax_transverse_axes=True`, it maintains the requested
pressure along the two transverse axes while fixing the stretch axis. If this
option is disabled, the barostat measures pressure without changing box
lengths. Temperature is read from the integrator or existing barostat; pass it
explicitly if it cannot be determined.

### Pore geometry and PSD with Zeo++

`PorePropertiesCalculator` calls the external Zeo++ `network` executable for
periodically sampled OpenMM frames. It writes a P1 CIF for each complete unit
cell, then calculates the largest included sphere (LCD), largest free sphere
(PLD), largest included sphere along the free-sphere path (Dif), channel
dimensionality, accessible surface area and volume, and pore-size distribution
(PSD).

```python
from polymer_properties_calc import PorePropertiesCalculator

pores = PorePropertiesCalculator(
    zeo_executable="/path/to/zeo++/network",
    equilibration_steps=20_000,
    n_frames=20,
    sample_interval_steps=5_000,
    probe_radius_angstrom=0.0,
    monte_carlo_samples=50_000,
)
pores.run(simulation)
result = pores.analyze()
print(result.pore_limiting_diameter_angstrom)
print(result.largest_cavity_diameter_angstrom)
print(result.accessible_volume_fraction)
print(result.pore_diameter_bins_angstrom, result.pore_size_distribution)
```

In Zeo++, PLD is the largest spherical probe that can pass through a
periodic channel; LCD is the largest sphere that fits inside a cavity. PSD is a
histogram of local pore diameters. By default, the calculator returns Zeo++'s
derivative distribution (`Derivative_dist`), and `raw_data` stores values for
each frame. The default probe radius is zero for geometric void analysis. Set a
molecular probe radius, for example 1.86 Å for the N₂ probe, when measuring
accessibility to a specific probe. Zeo++ receives the same radius for channel
accessibility and the measurement probe, as its documentation recommends.
Increasing `monte_carlo_samples` improves sampling statistics at the cost of
runtime.

Zeo++ is installed separately from this Python package. It uses CCDC elemental
radii by default, and `-ha` enables its high-accuracy Voronoi scheme. For
polymers that require custom atomic radii, configure the Zeo++ radii for the
model before running. This is a geometric analysis of instantaneous
configurations: results depend on sampled frames, cell size, atomic radii, and
probe choice, and are not by themselves estimates of experimental sorption
porosity.
