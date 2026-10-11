"""Analytical molecular-weight distribution moments for ideal random scission.

The model assumes initially monodisperse linear chains and independent cleavage
of every backbone bond with probability ``p``.  It is deliberately small and
auditable: it adds distribution-level predictions without pretending to model
initial polydispersity, preferential cleavage, branching, or soluble-fragment
removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PCL_REPEAT_UNIT_KDA = 0.114142


@dataclass(frozen=True)
class DistributionMoments:
    """Number- and weight-average molecular weights for random fragments."""

    bond_break_probability: np.ndarray
    mn_kda: np.ndarray
    mw_kda: np.ndarray
    dispersity: np.ndarray
    initial_degree_of_polymerization: int


def bond_break_probability_from_mn(
    mn_kda: np.ndarray | float,
    mn0_kda: float,
    repeat_unit_kda: float = PCL_REPEAT_UNIT_KDA,
) -> np.ndarray:
    """Infer independent bond-cleavage probability from ``Mn``.

    For an initial chain with ``n`` repeat units, the expected fragment count is
    ``1 + (n - 1) p``.  Conservation of repeat units therefore gives
    ``Mn/Mn0 = 1 / (1 + (n - 1) p)``.
    """

    if mn0_kda <= 0 or repeat_unit_kda <= 0:
        raise ValueError("molecular weights must be positive")
    n = max(2, int(round(mn0_kda / repeat_unit_kda)))
    mn = np.asarray(mn_kda, dtype=float)
    if np.any(mn <= 0) or np.any(mn > mn0_kda * (1.0 + 1e-12)):
        raise ValueError("Mn must lie in (0, Mn0]")
    probability = (mn0_kda / mn - 1.0) / (n - 1)
    return np.clip(probability, 0.0, 1.0)


def random_scission_distribution_moments(
    mn_kda: np.ndarray | float,
    mn0_kda: float,
    repeat_unit_kda: float = PCL_REPEAT_UNIT_KDA,
) -> DistributionMoments:
    """Return exact expected ``Mn``, ``Mw``, and dispersity for ideal scission.

    The second moment follows by summing all ordered pairs of repeat units that
    remain connected after cleavage.  For separation ``d``, the connection
    probability is ``(1-p)**d``.
    """

    requested_mn = np.asarray(mn_kda, dtype=float)
    probability = bond_break_probability_from_mn(requested_mn, mn0_kda, repeat_unit_kda)
    n = max(2, int(round(mn0_kda / repeat_unit_kda)))
    effective_repeat_mass = mn0_kda / n
    distances = np.arange(1, n, dtype=float)
    multiplicities = n - distances
    flat_probability = np.atleast_1d(probability).reshape(-1)
    connected_pairs = n + 2.0 * np.sum(
        multiplicities[None, :] * (1.0 - flat_probability[:, None]) ** distances[None, :],
        axis=1,
    )
    expected_fragments = 1.0 + (n - 1) * flat_probability
    mn = mn0_kda / expected_fragments
    mw = effective_repeat_mass * connected_pairs / n
    dispersity = mw / mn
    shape = probability.shape
    return DistributionMoments(
        bond_break_probability=probability,
        mn_kda=mn.reshape(shape),
        mw_kda=mw.reshape(shape),
        dispersity=dispersity.reshape(shape),
        initial_degree_of_polymerization=n,
    )
