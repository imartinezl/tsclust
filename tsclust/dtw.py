#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
from scipy.spatial.distance import cdist

from metrics import get_metric
from step_pattern import get_pattern
from window import get_window
from utils import validate_time_series

# import metrics
# import step_pattern
# import window
# import utils

# Benchmark with:
# https://github.com/ricardodeazambuja/DTW
# https://github.com/statefb/dtwalign
# https://github.com/pierre-rouanet/dtw
# https://github.com/DynamicTimeWarping/dtw-python
# https://github.com/paul-freeman/dtw
# https://github.com/tslearn-team/tslearn/blob/775dadd/tslearn/metrics.py
# https://github.com/johannfaouzi/pyts/blob/1aa45589b91a12e8d55db86f1f97dca0b6e99984/pyts/metrics/dtw.py#L238
# https://github.com/alan-turing-institute/sktime/blob/b410de5e61aceb99c65e9ef5ac4e431c7d3d17f3/sktime/contrib/distance_based/ts_distance_measures.py#L11
# https://github.com/titu1994/dtw-numba
# https://github.com/klon/ucrdtw
# https://github.com/halachkin/cdtw
# https://github.com/wannesm/dtaidistance

jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": False,
    "error_model": "numpy",
    "fastmath": True,
    "debug": True,
}

# nb.config.THREADING_LAYER = 'omp'


@nb.jit(**jitkw)
def _compute_local_cost(x, y, dist, weight=None):
    n = len(x)
    m = len(y)
    local_cost = np.empty((n, m), dtype=np.float64)
    for i in nb.prange(n):
        for j in nb.prange(m):
            local_cost[i, j] = dist(x[i], y[j], weight)
    return local_cost


@nb.jit(**jitkw)
def _compute_global_cost(local_cost, pattern, window, window_size=None):
    n, m = local_cost.shape
    cost = np.ones((n, m), dtype=np.float64) * np.inf

    num_pattern = pattern.shape[0]
    max_pattern_len = pattern.shape[1]
    pattern_cost = np.zeros(num_pattern, dtype=np.float64)
    step_cost = np.zeros((num_pattern, max_pattern_len), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if window(i, j, n, m, window_size):
                if i == 0 and j == 0:
                    cost[i, j] = local_cost[ii, jj]
                    continue
                for p in range(num_pattern):
                    for s in range(max_pattern_len):
                        dx, dy, weight = pattern[p, s, :]
                        ii, jj = int(i + dx), int(j + dy)
                        if ii < 0 or jj < 0:
                            step_cost[p, s] = np.inf
                            continue
                        if np.isnana(weight):
                            step_cost[p, s] = cost[ii, jj]
                        else:
                            step_cost[p, s] = local_cost[ii, jj] * weight
                pattern_cost = step_cost.sum(axis=1)
                min_cost = pattern_cost.min()
                if min_cost != np.inf:
                    cost[i, j] = min_cost
    return cost


# @nb.jit(**jitkw)
def dtw(
    x,
    y,
    local_dist="sqeuclidean",
    step_pattern="asymmetricP05",
    window_name="none",
    window_size=None,
    compute_path=True,
):

    x = validate_time_series(x)
    y = validate_time_series(y)

    dist = get_metric(local_dist)  # jit method
    pattern = get_pattern(step_pattern)
    window = get_window(window_name)  # jit method

    cost, direction = _compute_cost(x, y, dist, pattern.array, window, window_size)

    dist, normalized_dist = _get_distance(cost, pattern)

    print(cost)
    print(direction)
    print(dist, normalized_dist)

    if compute_path:
        path = _backtrack(direction, pattern.array)
    else:
        path = None

    print(path)


def _get_distance(cost, pattern):
    dist = cost[-1, -1]
    normalized_dist = None

    if pattern.is_normalizable:
        n, m = cost.shape
        normalized_dist = pattern.do_normalize(dist, n, m)

    if dist == np.inf:
        raise ValueError("no alignment path found")
    return dist, normalized_dist


@nb.jit(**jitkw)
def _backtrack(direction, pattern):
    i, j = direction.shape
    i -= 1
    j -= 1
    path = np.array(((i, j),), dtype=np.int64)
    while i + j > 0:
        pattern_idx = int(direction[i, j])
        local_path, origin = _get_local_path(pattern, pattern_idx, i, j)
        path = np.vstack((path, local_path))
        i, j = origin

    return path[::-1]


@nb.jit(**jitkw)
def _get_local_path(pattern, pattern_idx, i, j):
    """Helper function to get local path."""
    weight_col = pattern[pattern_idx, :, 2]
    # note: starting point of pattern was already added
    num_steps = np.sum(weight_col != 0) - 1
    local_path = np.ones((num_steps, 2), dtype=np.int64) * -1
    origin = (0, 0)
    for s in range(num_steps):
        dx, dy, weight = pattern[pattern_idx, s, :]
        ii, jj = int(i + dx), int(j + dy)
        if np.isnan(weight):
            origin = (ii, jj)
        local_path[s, :] = (ii, jj)
    return local_path[::-1], origin


@nb.jit(**jitkw)
def _compute_cost(x, y, dist, pattern, window, window_size):
    n = x.shape[0]
    m = y.shape[0]
    cost = np.ones((n, m), dtype=np.float64) * np.inf
    direction = np.ones((n, m), dtype=np.float64) * (-1)

    num_pattern = pattern.shape[0]
    max_pattern_len = pattern.shape[1]
    pattern_cost = np.zeros(num_pattern, dtype=np.float64)
    step_cost = np.zeros((num_pattern, max_pattern_len), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if window(i, j, n, m, window_size):
                if i == 0 and j == 0:
                    cost[i, j] = dist(x[i], y[j])
                    continue
                for p in range(num_pattern):
                    for s in range(max_pattern_len):
                        dx, dy, weight = pattern[p, s, :]
                        ii, jj = int(i + dx), int(j + dy)
                        if ii < 0 or jj < 0:
                            step_cost[p, s] = np.inf
                            continue
                        if np.isnan(weight):
                            step_cost[p, s] = cost[ii, jj]
                        else:
                            step_cost[p, s] = dist(x[ii], y[jj]) * weight
                pattern_cost = step_cost.sum(axis=1)
                # TO-DO: review if this only happens at i==0 and j==0
                # if np.isinf(pattern_cost).sum() == num_pattern:
                #     pattern_cost = np.zeros(num_pattern, dtype=np.float64)
                #     for p in range(num_pattern):
                #         for s in range(max_pattern_len):
                #             if not step_cost[p, s] == np.inf:
                #                 pattern_cost[p] += step_cost[p, s]
                min_cost = pattern_cost.min()
                min_cost_dir = pattern_cost.argmin()
                if min_cost != np.inf:
                    cost[i, j] = min_cost
                    direction[i, j] = min_cost_dir
    return cost, direction


import timeit

# TO-DO
# a) variable time axis: imagine irregularly sampled time series (literature?)
# b) also, implement online dynamic time warping for large time series (bigger than 10000 points, for example)
# alternative to this: downsampled time series, or indexing?
# c) independent vs dependent multidimensional DTW:
# independent version (freedom wrap across all dimensions, simply summing DTW),
# dependent version (cumulative squared euclidean distance across all dimensions)


x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 3, 8])

x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([[2, 2], [3, 3], [3, 3], [8, 8]])

x = np.array([1, 2, 3, 4, 3, 2, 1, 1, 1, 2])
y = np.array([0, 1, 1, 2, 3, 4, 3, 2, 1, 1])

# dtw(x, y)
# print(nb.threading_layer())
# print(nb.get_num_threads())

t1 = timeit.timeit(lambda: dtw(x, y), number=1)
t2 = timeit.timeit(lambda: dtw(x, y), number=1)
print(t1, t2, t1 / t2)
