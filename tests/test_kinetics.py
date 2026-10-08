import numpy as np

from pcl_model.kinetics import (
    autocatalytic_random_scission_mn,
    random_scission_mn,
)


def test_random_scission_inverse_mn_is_linear() -> None:
    time = np.array([0.0, 10.0, 30.0])
    mn = random_scission_mn(time, 50.0, 2e-4)
    assert np.allclose(1.0 / mn, 1.0 / 50.0 + 2e-4 * time)


def test_autocatalytic_limit_tends_to_random_scission() -> None:
    time = np.linspace(0, 100, 20)
    expected = random_scission_mn(time, 42.0, 5e-5)
    actual = autocatalytic_random_scission_mn(time, 42.0, 5e-5, 1e-7)
    assert np.allclose(actual, expected, rtol=2e-6)

