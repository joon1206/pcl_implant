from dataclasses import replace

import numpy as np

from pcl_model.model import finite_volume_diffusion, make_grid
from pcl_model.parameters import Geometry, ModelParameters, SimulationConfig
from pcl_model.solver import simulate


def test_constant_field_and_equal_external_concentration_has_zero_diffusion() -> None:
    config = SimulationConfig(cells=12)
    grid = make_grid(config)
    concentration = np.full(config.cells, 0.3)
    diffusion, surface_flux = finite_volume_diffusion(
        concentration, np.ones(config.cells), grid, 2.0, 0.3
    )
    assert np.allclose(diffusion, 0.0)
    assert surface_flux == 0.0


def test_discrete_robin_balance_is_conservative_for_all_geometries() -> None:
    for geometry in Geometry:
        config = SimulationConfig(geometry=geometry, cells=17)
        grid = make_grid(config)
        concentration = np.linspace(0.1, 0.8, config.cells)
        derivative, surface_flux = finite_volume_diffusion(
            concentration, np.linspace(0.5, 1.0, config.cells), grid, 0.4, 0.0
        )
        inventory_rate = np.sum(derivative * grid.cell_volumes)
        boundary_loss = grid.face_areas[-1] * surface_flux
        assert np.isclose(inventory_rate, -boundary_loss, rtol=1e-12, atol=1e-12)


def test_vanishing_hydrolysis_preserves_initial_state() -> None:
    parameters = ModelParameters(scission_rate_inv_kda_day=1e-20)
    config = SimulationConfig(duration_days=50.0, output_interval_days=5.0, cells=20)
    result = simulate(parameters, config)
    assert np.allclose(result.mean_mn_kda, parameters.mn0_kda, rtol=1e-10)
    assert np.allclose(result.mass_retention, 1.0)


def test_solution_is_positive_and_refines_spatially() -> None:
    parameters = ModelParameters(acid_diffusivity_mm2_day=0.003, mass_transfer_mm_day=0.01)
    base = SimulationConfig(duration_days=80.0, output_interval_days=4.0, cells=20)
    coarse = simulate(parameters, base)
    medium = simulate(parameters, replace(base, cells=40))
    fine = simulate(parameters, replace(base, cells=80))
    assert np.min(fine.acid) >= 0
    assert np.min(fine.mn_kda) > 0
    assert abs(medium.mean_mn_kda[-1] - fine.mean_mn_kda[-1]) < abs(
        coarse.mean_mn_kda[-1] - fine.mean_mn_kda[-1]
    )


def test_strong_clearance_slows_autocatalytic_degradation() -> None:
    config = SimulationConfig(duration_days=150.0, output_interval_days=5.0, cells=30)
    retained = simulate(ModelParameters(mass_transfer_mm_day=0.0), config)
    cleared = simulate(ModelParameters(mass_transfer_mm_day=5.0), config)
    assert cleared.mean_mn_kda[-1] > retained.mean_mn_kda[-1]

