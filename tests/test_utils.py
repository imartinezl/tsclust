import numpy as np
from numpy.testing import assert_allclose

import tsclust.utils


def test_arraylike_copy():
    X_npy = np.array([1, 2, 3])
    assert_allclose(tsclust.utils._arraylike_copy(X_npy), X_npy)
    assert_allclose(tsclust.utils._arraylike_copy(X_npy) is X_npy, False)
