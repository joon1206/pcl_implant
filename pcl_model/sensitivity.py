"""Dependency-free Saltelli/Sobol utilities built on NumPy and SciPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ParameterRange:
    name: str
    lower: float
    upper: float
    logarithmic: bool = False

    def transform(self, unit_values: np.ndarray) -> np.ndarray:
        if self.lower <= 0 and self.logarithmic:
            raise ValueError("logarithmic ranges require positive bounds")
        if self.upper <= self.lower:
            raise ValueError("upper bound must exceed lower bound")
        if self.logarithmic:
            return np.exp(np.log(self.lower) + unit_values * np.log(self.upper / self.lower))
        return self.lower + unit_values * (self.upper - self.lower)


def saltelli_matrices(
    ranges: list[ParameterRange], base_sample_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Create A, B, and A-with-B-column matrices from a scrambled Sobol net."""

    if base_sample_size < 2 or base_sample_size & (base_sample_size - 1):
        raise ValueError("base_sample_size must be a power of two")
    dimension = len(ranges)
    if dimension == 0:
        raise ValueError("at least one parameter range is required")
    unit = qmc.Sobol(d=2 * dimension, scramble=True, seed=seed).random_base2(
        int(np.log2(base_sample_size))
    )
    a_unit, b_unit = unit[:, :dimension], unit[:, dimension:]
    a = np.column_stack([item.transform(a_unit[:, index]) for index, item in enumerate(ranges)])
    b = np.column_stack([item.transform(b_unit[:, index]) for index, item in enumerate(ranges)])
    hybrids = []
    for index in range(dimension):
        hybrid = a.copy()
        hybrid[:, index] = b[:, index]
        hybrids.append(hybrid)
    return a, b, hybrids


def sobol_indices(
    output_a: np.ndarray, output_b: np.ndarray, output_hybrids: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate first-order and total-order Sobol indices.

    First order uses the Saltelli covariance estimator and total order uses the
    Jansen squared-difference estimator.  Raw estimates are returned; small
    negative first-order values are useful diagnostics of finite-sample noise.
    """

    ya = np.asarray(output_a, dtype=float)
    yb = np.asarray(output_b, dtype=float)
    if ya.shape != yb.shape or ya.ndim != 1:
        raise ValueError("A and B outputs must be matching 1D arrays")
    variance = np.var(np.concatenate([ya, yb]), ddof=1)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("model output must have positive finite variance")
    first = []
    total = []
    for values in output_hybrids:
        hybrid = np.asarray(values, dtype=float)
        if hybrid.shape != ya.shape:
            raise ValueError("hybrid outputs must match A and B")
        first.append(np.mean(yb * (hybrid - ya)) / variance)
        total.append(0.5 * np.mean((ya - hybrid) ** 2) / variance)
    return np.asarray(first), np.asarray(total)
