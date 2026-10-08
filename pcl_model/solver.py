"""Stable method-of-lines integration and derived observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import (
    dimensionless_groups,
    initial_state,
    local_mechanics,
    make_grid,
    right_hand_side,
    unpack_state,
)
from .parameters import ModelParameters, SimulationConfig


@dataclass(frozen=True)
class SimulationResult:
    time_days: np.ndarray
    radius_mm: np.ndarray
    mn_kda: np.ndarray
    acid: np.ndarray
    crystallinity: np.ndarray
    solid_fraction: np.ndarray
    mean_mn_kda: np.ndarray
    mass_retention: np.ndarray
    modulus_parallel_fraction: np.ndarray
    modulus_series_fraction: np.ndarray
    strength_weakest_fraction: np.ndarray
    failure_time_days: float | None
    diagnostics: dict[str, float | str]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[None, :], axis=1) / np.sum(weights)


def simulate(parameters: ModelParameters, config: SimulationConfig) -> SimulationResult:
    parameters.validate()
    config.validate()
    grid = make_grid(config)
    time = np.arange(0.0, config.duration_days, config.output_interval_days)
    if time.size == 0 or time[-1] < config.duration_days:
        time = np.append(time, config.duration_days)
    state0 = initial_state(parameters, config)

    solution = solve_ivp(
        lambda t, y: right_hand_side(t, y, parameters, config, grid),
        (0.0, config.duration_days),
        state0,
        method="BDF",
        t_eval=time,
        rtol=config.relative_tolerance,
        atol=config.absolute_tolerance,
        max_step=config.maximum_step_days,
    )
    if not solution.success:
        raise RuntimeError(f"integration failed: {solution.message}")
    if not np.all(np.isfinite(solution.y)):
        raise FloatingPointError("integration produced non-finite values")

    states = solution.y.T.reshape(time.size, 4, config.cells)
    inv_mn = states[:, 0]
    acid = states[:, 1]
    xc = states[:, 2]
    solid = states[:, 3]
    tolerance = 2e-7
    if np.min(acid) < -tolerance or np.min(solid) < -tolerance:
        raise FloatingPointError("integration violated positivity beyond solver tolerance")
    acid = np.maximum(acid, 0.0)
    xc = np.clip(xc, 0.0, 1.0)
    solid = np.clip(solid, 0.0, 1.0)
    mn = 1.0 / np.maximum(inv_mn, np.finfo(float).tiny)

    weights = grid.cell_volumes
    # Global Mn is total retained polymer mass divided by total chain count.
    # It is therefore a solid-mass-weighted harmonic, not arithmetic, mean of
    # local Mn values.
    mean_solid = _weighted_mean(solid, weights)
    mean_mn = mean_solid / np.maximum(
        _weighted_mean(solid / mn, weights), np.finfo(float).tiny
    )
    mass_retention = mean_solid
    modulus_local, strength_local = local_mechanics(mn, xc, solid, parameters)
    modulus_parallel = _weighted_mean(modulus_local, weights)
    modulus_series = 1.0 / _weighted_mean(
        1.0 / np.maximum(modulus_local, 1e-12), weights
    )
    strength_weakest = np.min(strength_local, axis=1)
    failed = np.flatnonzero(strength_weakest <= config.functional_strength_fraction)
    failure_time = float(time[failed[0]]) if failed.size else None

    diagnostics = dimensionless_groups(parameters, config)
    diagnostics.update(
        {
            "minimum_acid": float(np.min(acid)),
            "maximum_acid": float(np.max(acid)),
            "minimum_solid_fraction": float(np.min(solid)),
            "solver_function_evaluations": float(solution.nfev),
        }
    )
    return SimulationResult(
        time_days=time,
        radius_mm=grid.centers_mm,
        mn_kda=mn,
        acid=acid,
        crystallinity=xc,
        solid_fraction=solid,
        mean_mn_kda=mean_mn,
        mass_retention=mass_retention,
        modulus_parallel_fraction=modulus_parallel,
        modulus_series_fraction=modulus_series,
        strength_weakest_fraction=strength_weakest,
        failure_time_days=failure_time,
        diagnostics=diagnostics,
    )

