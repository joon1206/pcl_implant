"""Governing equations and constitutive relations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import ModelParameters, SimulationConfig

GAS_CONSTANT_J_MOL_K = 8.314462618


@dataclass(frozen=True)
class FiniteVolumeGrid:
    faces_mm: np.ndarray
    centers_mm: np.ndarray
    cell_volumes: np.ndarray
    face_areas: np.ndarray


def make_grid(config: SimulationConfig) -> FiniteVolumeGrid:
    """Cell-centred grid with radial metric factors; angular constants cancel."""
    q = config.geometry.radial_power
    faces = np.linspace(0.0, config.characteristic_length_mm, config.cells + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    volumes = (faces[1:] ** (q + 1) - faces[:-1] ** (q + 1)) / (q + 1)
    areas = faces**q
    if q == 0:
        areas[0] = 1.0
    return FiniteVolumeGrid(faces, centers, volumes, areas)


def arrhenius_factor(parameters: ModelParameters) -> float:
    reference_k = parameters.reference_temperature_c + 273.15
    actual_k = parameters.temperature_c + 273.15
    exponent = -1000.0 * parameters.activation_energy_kj_mol / GAS_CONSTANT_J_MOL_K
    return float(np.exp(exponent * (1.0 / actual_k - 1.0 / reference_k)))


def effective_scission_constant(parameters: ModelParameters) -> float:
    return (
        parameters.scission_rate_inv_kda_day
        * parameters.ph_rate_multiplier
        * arrhenius_factor(parameters)
    )


def unpack_state(state: np.ndarray, cells: int) -> tuple[np.ndarray, ...]:
    return tuple(state.reshape(4, cells))


def pack_state(inv_mn: np.ndarray, acid: np.ndarray, xc: np.ndarray, solid: np.ndarray) -> np.ndarray:
    return np.concatenate([inv_mn, acid, xc, solid])


def initial_state(parameters: ModelParameters, config: SimulationConfig) -> np.ndarray:
    n = config.cells
    return pack_state(
        np.full(n, 1.0 / parameters.mn0_kda),
        np.full(n, parameters.external_acid),
        np.full(n, parameters.crystallinity0),
        np.ones(n),
    )


def acid_diffusivity(xc: np.ndarray, solid: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    crystal_factor = np.exp(
        -parameters.crystal_transport_barrier * (xc - parameters.crystallinity0)
    )
    pore_factor = 1.0 + parameters.porosity_transport_gain * np.maximum(1.0 - solid, 0.0)
    return parameters.acid_diffusivity_mm2_day * crystal_factor * pore_factor


def finite_volume_diffusion(
    concentration: np.ndarray,
    diffusivity: np.ndarray,
    grid: FiniteVolumeGrid,
    mass_transfer_mm_day: float,
    external_concentration: float,
) -> tuple[np.ndarray, float]:
    """Conservative diffusion with symmetry and outward Robin clearance.

    Flux is positive in the outward radial direction.  The centre flux is zero.
    At the exposed surface, F = h (C_surface - C_external).
    """
    n = concentration.size
    flux = np.zeros(n + 1)
    distances = np.diff(grid.centers_mm)
    interface_d = 2.0 * diffusivity[:-1] * diffusivity[1:] / np.maximum(
        diffusivity[:-1] + diffusivity[1:], 1e-300
    )
    flux[1:n] = -interface_d * np.diff(concentration) / distances
    flux[n] = mass_transfer_mm_day * (concentration[-1] - external_concentration)
    net_out = grid.face_areas[1:] * flux[1:] - grid.face_areas[:-1] * flux[:-1]
    return -net_out / grid.cell_volumes, float(flux[n])


def right_hand_side(
    _time_days: float,
    state: np.ndarray,
    parameters: ModelParameters,
    config: SimulationConfig,
    grid: FiniteVolumeGrid,
) -> np.ndarray:
    inv_mn, acid, xc, solid = unpack_state(state, config.cells)
    safe_inv_mn = np.maximum(inv_mn, 1.0 / parameters.mn0_kda)
    safe_acid = np.maximum(acid, 0.0)
    safe_xc = np.clip(xc, 0.0, 0.999)
    safe_solid = np.clip(solid, 0.0, 1.0)
    mn = 1.0 / safe_inv_mn

    amorphous_access = np.clip(
        (1.0 - safe_xc) / (1.0 - parameters.crystallinity0), 0.0, 2.0
    )
    d_inv_mn = (
        effective_scission_constant(parameters)
        * amorphous_access
        * (1.0 + parameters.autocatalysis_per_acid * safe_acid)
    )

    diffusivity = acid_diffusivity(safe_xc, safe_solid, parameters)
    diffusion, _ = finite_volume_diffusion(
        safe_acid,
        diffusivity,
        grid,
        parameters.mass_transfer_mm_day,
        parameters.external_acid,
    )
    d_acid = (
        diffusion
        + parameters.acid_yield_kda * d_inv_mn
        - parameters.bulk_clearance_per_day * safe_acid
    )

    degradation_fraction = np.clip(1.0 - mn / parameters.mn0_kda, 0.0, 1.0)
    target_xc = np.minimum(
        parameters.crystallinity0
        + parameters.chemicrystallization_gain * degradation_fraction,
        0.98,
    )
    scission_frequency_per_day = d_inv_mn * parameters.mn0_kda
    d_xc = (
        parameters.chemicrystallization_rate_per_day * (target_xc - safe_xc)
        - parameters.crystal_hydrolysis_per_day * scission_frequency_per_day * safe_xc
    )

    activation = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (mn - parameters.soluble_mn_kda) / parameters.dissolution_width_kda,
                -60.0,
                60.0,
            )
        )
    )
    d_solid = -parameters.dissolution_rate_per_day * activation * safe_solid
    return pack_state(d_inv_mn, d_acid, d_xc, d_solid)


def local_mechanics(
    mn_kda: np.ndarray,
    crystallinity: np.ndarray,
    solid_fraction: np.ndarray,
    parameters: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local normalized tensile modulus and strength.

    The modulus mixture captures competing chemicrystallization stiffening, while
    entanglement/tie-chain loss and porosity reduce both properties.  Strength is
    intentionally more molecular-weight-sensitive than small-strain modulus.
    """
    ratio = parameters.crystal_to_amorphous_modulus_ratio
    mix0 = (1.0 - parameters.crystallinity0) + ratio * parameters.crystallinity0
    mix = (1.0 - crystallinity) + ratio * crystallinity
    denominator = parameters.mn0_kda - parameters.critical_entanglement_mn_kda
    tie = np.clip(
        (mn_kda - parameters.critical_entanglement_mn_kda) / denominator,
        0.0,
        1.0,
    )
    modulus = (
        (mix / mix0)
        * tie**parameters.modulus_tie_exponent
        * solid_fraction**parameters.porosity_modulus_exponent
    )
    strength = (
        tie**parameters.strength_tie_exponent
        * solid_fraction**parameters.porosity_strength_exponent
    )
    return modulus, strength


def dimensionless_groups(parameters: ModelParameters, config: SimulationConfig) -> dict[str, float | str]:
    reaction_rate = effective_scission_constant(parameters) * parameters.mn0_kda
    diffusion_time = config.characteristic_length_mm**2 / parameters.acid_diffusivity_mm2_day
    damkohler = reaction_rate * diffusion_time
    biot = parameters.mass_transfer_mm_day * config.characteristic_length_mm / parameters.acid_diffusivity_mm2_day
    if damkohler < 0.1:
        regime = "reaction-limited / nearly spatially uniform"
    elif damkohler > 10.0:
        regime = "transport-influenced; spatial resolution required"
    else:
        regime = "mixed reaction-transport"
    return {
        "damkohler_scission": damkohler,
        "biot_clearance": biot,
        "diffusion_time_days": diffusion_time,
        "regime": regime,
    }

