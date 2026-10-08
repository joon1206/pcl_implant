"""Closed-form molecular-weight baselines and calibration helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def random_scission_mn(time_days: np.ndarray, mn0_kda: float, k_inv_kda_day: float) -> np.ndarray:
    """Constant random scission: 1/Mn = 1/Mn0 + k t."""
    return 1.0 / (1.0 / mn0_kda + k_inv_kda_day * np.asarray(time_days))


def exponential_mn(time_days: np.ndarray, mn0_kda: float, rate_per_day: float) -> np.ndarray:
    return mn0_kda * np.exp(-rate_per_day * np.asarray(time_days))


def autocatalytic_random_scission_mn(
    time_days: np.ndarray,
    mn0_kda: float,
    base_rate_inv_kda_day: float,
    feedback_kda: float,
) -> np.ndarray:
    """Uniform, retained-acid limit of the PDE.

    If acid is proportional to newly created chains, dI/dt=k(1+g(I-I0)),
    where I=1/Mn and g has units kDa.  This expression is its exact solution.
    """
    time = np.asarray(time_days)
    inverse_mn = 1.0 / mn0_kda + np.expm1(base_rate_inv_kda_day * feedback_kda * time) / feedback_kda
    return 1.0 / inverse_mn


def fit_kinetic_models(time_days: np.ndarray, mn_kda: np.ndarray) -> dict[str, dict[str, object]]:
    time = np.asarray(time_days, dtype=float)
    observed = np.asarray(mn_kda, dtype=float)
    mn0 = float(observed[0])
    if time.ndim != 1 or observed.shape != time.shape or np.any(observed <= 0):
        raise ValueError("time and positive Mn observations must be matching 1D arrays")

    random_fit = least_squares(
        lambda z: random_scission_mn(time, mn0, np.exp(z[0])) - observed,
        np.log([1e-4]),
    )
    exponential_fit = least_squares(
        lambda z: exponential_mn(time, mn0, np.exp(z[0])) - observed,
        np.log([2e-3]),
    )
    auto_fit = least_squares(
        lambda z: autocatalytic_random_scission_mn(time, mn0, *np.exp(z)) - observed,
        np.log([5e-5, 100.0]),
        bounds=(np.log([1e-10, 1e-4]), np.log([1.0, 1e7])),
    )
    definitions = {
        "random_scission": (
            np.exp(random_fit.x),
            lambda t, p: random_scission_mn(t, mn0, p[0]),
            ["k_inv_kda_day"],
        ),
        "exponential": (
            np.exp(exponential_fit.x),
            lambda t, p: exponential_mn(t, mn0, p[0]),
            ["rate_per_day"],
        ),
        "autocatalytic_random_scission": (
            np.exp(auto_fit.x),
            lambda t, p: autocatalytic_random_scission_mn(t, mn0, p[0], p[1]),
            ["base_rate_inv_kda_day", "feedback_kda"],
        ),
    }
    result: dict[str, dict[str, object]] = {}
    for name, (parameters, function, labels) in definitions.items():
        prediction = function(time, parameters)
        result[name] = {
            "parameters": dict(zip(labels, map(float, parameters))),
            "prediction": prediction,
            "rmse_kda": float(np.sqrt(np.mean((prediction - observed) ** 2))),
            "mae_kda": float(np.mean(np.abs(prediction - observed))),
        }
    return result

