import numpy as np

from pcl_model.sensitivity import ParameterRange, saltelli_matrices, sobol_indices


def test_saltelli_sampling_is_deterministic_and_bounded() -> None:
    ranges = [ParameterRange("linear", 2.0, 4.0), ParameterRange("log", 1e-3, 1e1, True)]
    a1, b1, hybrids1 = saltelli_matrices(ranges, 16, seed=17)
    a2, b2, hybrids2 = saltelli_matrices(ranges, 16, seed=17)
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)
    assert all(np.array_equal(x, y) for x, y in zip(hybrids1, hybrids2))
    assert np.all((a1[:, 0] >= 2.0) & (a1[:, 0] <= 4.0))
    assert np.all((a1[:, 1] >= 1e-3) & (a1[:, 1] <= 1e1))


def test_sobol_indices_identify_dominant_additive_input() -> None:
    ranges = [ParameterRange("x", 0.0, 1.0), ParameterRange("y", 0.0, 1.0)]
    a, b, hybrids = saltelli_matrices(ranges, 1024, seed=4)
    evaluate = lambda values: values[:, 0] + 0.1 * values[:, 1]
    first, total = sobol_indices(evaluate(a), evaluate(b), [evaluate(item) for item in hybrids])
    assert first[0] > 0.95
    assert total[0] > 0.95
    assert first[1] < 0.02
    assert total[1] < 0.02
