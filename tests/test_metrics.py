#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial
import tsclust.metrics


def generate_data(n=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.ranf(n) * 100
    y = np.random.ranf(n) * 100
    w = np.random.ranf(n) * 100
    return x, y, w


def test_minkowski():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.minkowski(x, y, 2)
    d2 = scipy.spatial.distance.minkowski(x, y, 2)

    np.testing.assert_almost_equal(d1, d2)


def test_wminkowski():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.wminkowski(x, y, 2, w)
    d2 = scipy.spatial.distance.wminkowski(x, y, 2, w)
    np.testing.assert_almost_equal(d1, d2)


def test_euclidean():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.euclidean(x, y, w)
    d2 = scipy.spatial.distance.euclidean(x, y, w)
    np.testing.assert_almost_equal(d1, d2)


def test_sqeuclidean():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.sqeuclidean(x, y, w)
    d2 = scipy.spatial.distance.sqeuclidean(x, y, w)
    np.testing.assert_almost_equal(d1, d2)


def test_correlation():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.correlation(x, y, w, centered=True)
    d2 = scipy.spatial.distance.correlation(x, y, w, centered=True)
    np.testing.assert_almost_equal(d1, d2)


def test_cosine():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.cosine(x, y, w)
    d2 = scipy.spatial.distance.cosine(x, y, w)
    np.testing.assert_almost_equal(d1, d2)


def test_seuclidean():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.seuclidean(x, y, w)
    d2 = scipy.spatial.distance.seuclidean(x, y, w)
    np.testing.assert_almost_equal(d1, d2)


def test_cityblock():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.cityblock(x, y, w)
    d2 = scipy.spatial.distance.cityblock(x, y, w)
    np.testing.assert_almost_equal(d1, d2)


def test_mahalanobis():
    n = 1000
    x, y, w = generate_data(n)
    VI = np.random.ranf((n, n)) * 100

    d1 = tsclust.metrics.mahalanobis(x, y, VI)
    d2 = scipy.spatial.distance.mahalanobis(x, y, VI)
    np.testing.assert_almost_equal(d1, d2)


def test_chebychev():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.chebyshev(x, y, w)
    d2 = scipy.spatial.distance.chebyshev(x, y, w)
    np.testing.assert_almost_equal(d1, d2)


def test_braycurtis():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.braycurtis(x, y, w)
    d2 = scipy.spatial.distance.braycurtis(x, y, w)
    np.testing.assert_almost_equal(d1, d2)


def test_canberra():
    n = 1000
    x, y, w = generate_data(n)

    d1 = tsclust.metrics.canberra(x, y, w)
    d2 = scipy.spatial.distance.canberra(x, y, w)
    np.testing.assert_almost_equal(d1, d2)
