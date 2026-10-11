import numpy as np

from pcl_model.distribution import random_scission_distribution_moments


def test_unbroken_monodisperse_chain_has_unit_dispersity() -> None:
    moments = random_scission_distribution_moments(42.03, 42.03)
    assert np.isclose(moments.bond_break_probability, 0.0)
    assert np.isclose(moments.mn_kda, 42.03)
    assert np.isclose(moments.mw_kda, 42.03)
    assert np.isclose(moments.dispersity, 1.0)


def test_random_scission_moments_are_physically_ordered() -> None:
    requested = np.array([42.03, 30.0, 10.0, 3.0])
    moments = random_scission_distribution_moments(requested, 42.03)
    assert np.allclose(moments.mn_kda, requested)
    assert np.all(moments.mw_kda >= moments.mn_kda - 1e-12)
    assert np.all(np.diff(moments.bond_break_probability) > 0)
    assert np.all((moments.dispersity >= 1.0 - 1e-12) & (moments.dispersity < 2.0))
