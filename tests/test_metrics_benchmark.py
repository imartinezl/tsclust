#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import scipy.spatial
import tsclust.metrics_scipy


def generate_data(n=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.ranf(n) * 100
    y = np.random.ranf(n) * 100
    w = np.random.ranf(n) * 100
    return x, y, w


def test_minkowski_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n, 0)

    benchmark.group = "minkowski"
    d1 = benchmark(tsclust.metrics_scipy.minkowski, x, y, 2)
    d2 = scipy.spatial.distance.minkowski(x, y, 2)
    assert d1 == pytest.approx(d2)


def test_minkowski_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n, 0)

    benchmark.group = "minkowski"
    d1 = tsclust.metrics_scipy.minkowski(x, y, 2)
    d2 = benchmark(scipy.spatial.distance.minkowski, x, y, 2)
    assert d1 == pytest.approx(d2)


def test_wminkowski_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "wminkowski"
    d1 = benchmark(tsclust.metrics_scipy.wminkowski, x, y, 2, w)
    d2 = scipy.spatial.distance.wminkowski(x, y, 2, w)
    assert d1 == pytest.approx(d2)


def test_wminkowski_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "wminkowski"
    d1 = tsclust.metrics_scipy.wminkowski(x, y, 2, w)
    d2 = benchmark(scipy.spatial.distance.wminkowski, x, y, 2, w)
    assert d1 == pytest.approx(d2)


def test_euclidean_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "euclidean"
    d1 = benchmark(tsclust.metrics_scipy.euclidean, x, y, w)
    d2 = scipy.spatial.distance.euclidean(x, y, w)
    assert d1 == pytest.approx(d2)


def test_euclidean_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "euclidean"
    d1 = tsclust.metrics_scipy.euclidean(x, y, w)
    d2 = benchmark(scipy.spatial.distance.euclidean, x, y, w)
    assert d1 == pytest.approx(d2)


def test_sqeuclidean_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "sqeuclidean"
    d1 = benchmark(tsclust.metrics_scipy.sqeuclidean, x, y, w)
    d2 = scipy.spatial.distance.sqeuclidean(x, y, w)
    assert d1 == pytest.approx(d2)


def test_sqeuclidean_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "sqeuclidean"
    d1 = tsclust.metrics_scipy.sqeuclidean(x, y, w)
    d2 = benchmark(scipy.spatial.distance.sqeuclidean, x, y, w)
    assert d1 == pytest.approx(d2)


def test_correlation_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "correlation"
    d1 = benchmark(tsclust.metrics_scipy.correlation, x, y, w, centered=True)
    d2 = scipy.spatial.distance.correlation(x, y, w, centered=True)
    assert d1 == pytest.approx(d2)


def test_correlation_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "correlation"
    d1 = tsclust.metrics_scipy.correlation(x, y, w, centered=True)
    d2 = benchmark(scipy.spatial.distance.correlation, x, y, w, centered=True)
    assert d1 == pytest.approx(d2)


def test_cosine_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "cosine"
    d1 = benchmark(tsclust.metrics_scipy.cosine, x, y, w)
    d2 = scipy.spatial.distance.cosine(x, y, w)
    assert d1 == pytest.approx(d2)


def test_cosine_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "cosine"
    d1 = tsclust.metrics_scipy.cosine(x, y, w)
    d2 = benchmark(scipy.spatial.distance.cosine, x, y, w)
    assert d1 == pytest.approx(d2)


def test_seuclidean_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "seuclidean"
    d1 = benchmark(tsclust.metrics_scipy.seuclidean, x, y, w)
    d2 = scipy.spatial.distance.seuclidean(x, y, w)
    assert d1 == pytest.approx(d2)


def test_seuclidean_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "seuclidean"
    d1 = tsclust.metrics_scipy.seuclidean(x, y, w)
    d2 = benchmark(scipy.spatial.distance.seuclidean, x, y, w)
    assert d1 == pytest.approx(d2)


def test_cityblock_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "cityblock"
    d1 = benchmark(tsclust.metrics_scipy.cityblock, x, y, w)
    d2 = scipy.spatial.distance.cityblock(x, y, w)
    assert d1 == pytest.approx(d2)


def test_cityblock_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "cityblock"
    d1 = tsclust.metrics_scipy.cityblock(x, y, w)
    d2 = benchmark(scipy.spatial.distance.cityblock, x, y, w)
    assert d1 == pytest.approx(d2)


def test_mahalanobis_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)
    VI = np.random.ranf((n, n)) * 100

    benchmark.group = "mahalanobis"
    d1 = benchmark(tsclust.metrics_scipy.mahalanobis, x, y, VI)
    d2 = scipy.spatial.distance.mahalanobis(x, y, VI)
    assert d1 == pytest.approx(d2)


def test_mahalanobis_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)
    VI = np.random.ranf((n, n)) * 100

    benchmark.group = "mahalanobis"
    d1 = tsclust.metrics_scipy.mahalanobis(x, y, VI)
    d2 = benchmark(scipy.spatial.distance.mahalanobis, x, y, VI)
    assert d1 == pytest.approx(d2)


def test_chebychev_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "chebyshev"
    d1 = benchmark(tsclust.metrics_scipy.chebyshev, x, y, w)
    d2 = scipy.spatial.distance.chebyshev(x, y, w)
    assert d1 == pytest.approx(d2)


def test_chebychev_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "chebyshev"
    d1 = tsclust.metrics_scipy.chebyshev(x, y, w)
    d2 = benchmark(scipy.spatial.distance.chebyshev, x, y, w)
    assert d1 == pytest.approx(d2)


def test_braycurtis_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "braycurtis"
    d1 = benchmark(tsclust.metrics_scipy.braycurtis, x, y, w)
    d2 = scipy.spatial.distance.braycurtis(x, y, w)
    assert d1 == pytest.approx(d2)


def test_braycurtis_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "braycurtis"
    d1 = tsclust.metrics_scipy.braycurtis(x, y, w)
    d2 = benchmark(scipy.spatial.distance.braycurtis, x, y, w)
    assert d1 == pytest.approx(d2)


def test_canberra_tsclust(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "canberra"
    d1 = benchmark(tsclust.metrics_scipy.canberra, x, y, w)
    d2 = scipy.spatial.distance.canberra(x, y, w)
    assert d1 == pytest.approx(d2)


def test_canberra_scipy(benchmark):
    n = 1000
    x, y, w = generate_data(n)

    benchmark.group = "canberra"
    d1 = tsclust.metrics_scipy.canberra(x, y, w)
    d2 = benchmark(scipy.spatial.distance.canberra, x, y, w)
    assert d1 == pytest.approx(d2)
