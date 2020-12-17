#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# from scipy.spatial.distance import cdist
import timeit

from tsclust.metrics import get_metric
from tsclust.step_pattern import get_pattern
from tsclust.window import get_window
from tsclust.utils import validate_time_series
from tsclust.result import DtwResult


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


def dtw(
    x,
    y,
    local_dist="euclidean",
    step_pattern="asymmetricP05",
    window_name="sakoechiba",
    window_size=19,
    compute_path=True,
):
    """
    Compute Dynamic Time Warping and find optimal alignment between two time series.

    **Details**

    The function performs Dynamic Time Warp (DTW) and computes the optimal
    alignment between two time series ``x`` and ``y``, given as numeric
    vectors. The “optimal” alignment minimizes the sum of distances between
    aligned elements. Lengths of ``x`` and ``y`` may differ.

    Several common variants of the DTW recursion are supported via the
    ``step_pattern`` argument, which defaults to ``symmetric2``. Step
    patterns are commonly used to *locally* constrain the slope of the
    alignment function.

    Windowing enforces a *global* constraint on the envelope of the warping
    path. It is selected by passing a string or function to the
    ``window_type`` argument. Commonly used windows are:

    -  ``"none"`` No windowing (default)
    -  ``"sakoechiba"`` A band around main diagonal
    -  ``"slantedband"`` A band around slanted diagonal
    -  ``"itakura"`` So-called Itakura parallelogram

    Some windowing functions may require parameters, such as the
    ``window_size`` argument.

    If the warping function is not required, computation can be sped up
    enabling the ``compute_path=False`` switch, which skips the backtracking
    step. The output object will then lack the ``path`` and
    ``path_origin`` fields.

    Parameters
    ----------
    x : 1D or 2D array (sample * feature)
        Query time series.
    y : 1D or 2D array (sample * feature)
        Reference time series.
    local_dist : string
        pointwise (local) distance function to use.
        Define how to calculate pair-wise distance between x and y.
    step_pattern :
        a stepPattern object describing the local warping steps
        allowed with their cost
    window_type :
        windowing function. Character: "none", "itakura",
        "sakoechiba", "slantedband".
    compute_path :
        Whether or not to obtain warping path. If False is selected,
        no backtrack will be computed and only alignment distance will be calculated (faster).


    Returns
    -------
    An object of class ``DtwResult``. See docs for the corresponding properties.


    Notes
    -----

    Cost matrices (both input and output) have query elements arranged
    row-wise (first index), and reference elements column-wise (second
    index). They print according to the usual convention, with indexes
    increasing down- and rightwards. Many DTW papers and tutorials show
    matrices according to plot-like conventions, i_e. reference index
    growing upwards. This may be confusing.

    A fast compiled version of the function is normally used. Should it be
    unavailable, the interpreted equivalent will be used as a fall-back with
    a warning.


    """

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

    path = None
    path_origin = None
    if compute_path:
        path, path_origin = _backtrack(direction, pattern.array)

    print(path, path_origin)
    dtw_result = DtwResult(x, y, cost, path, window, pattern, dist, normalized_dist)
    # dtw_result.plot_cost_matrix()
    # dtw_result.plot_pattern()
    # dtw_result.plot_path()
    # # dtw_result.plot_ts_subplot(x, "Query")
    # dtw_result.plot_ts_overlay(x, "Query")
    # dtw_result.plot_ts_overlay(y, "Reference")
    dtw_result.plot_summary()
    dtw_result.plot_warp()

    # plt.show()
    return dtw_result


@nb.jit(**jitkw)
def _compute_cost(x, y, dist, pattern, window, window_size):
    n = x.shape[0]
    m = y.shape[0]
    cost = np.ones((n, m), dtype=np.float64) * np.inf
    direction = np.ones((n, m), dtype=np.float64) * (-1)

    num_pattern = pattern.shape[0]
    max_pattern_len = pattern.shape[1]
    # pattern_cost = np.zeros(num_pattern, dtype=np.float64)
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
                        if weight == -1:
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


@nb.jit(**jitkw)
def _compute_cost_open_end(local_cost, pattern, window, window_size):
    n, m = local_cost.shape
    cost = np.ones((n, m), dtype=np.float64) * np.inf

    open_begin = True
    if open_begin:
        local_cost = np.vstack((np.zeros((1, local_cost.shape[1])), local_cost))
        cost = np.vstack((np.zeros((1, cost.shape[1])), cost))
        w_list[:, 0] += 1

    num_pattern = pattern.shape[0]
    max_pattern_len = pattern.shape[1]
    # pattern_cost = np.zeros(num_pattern, dtype=np.float64)
    step_cost = np.zeros((num_pattern, max_pattern_len), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if window(i, j, n, m, window_size):
                if i == 0 and j == 0:
                    cost[i, j] = local_cost[i, j]
                    continue
                for p in range(num_pattern):
                    for s in range(max_pattern_len):
                        dx, dy, weight = pattern[p, s, :]
                        ii, jj = int(i + dx), int(j + dy)
                        if ii < 0 or jj < 0:
                            step_cost[p, s] = np.inf
                            continue
                        if weight == -1:
                            step_cost[p, s] = cost[ii, jj]
                        else:
                            step_cost[p, s] = local_cost[ii, jj] * weight
                pattern_cost = step_cost.sum(axis=1)
                min_cost = pattern_cost.min()
                if min_cost != np.inf:
                    cost[i, j] = min_cost
    return cost


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
    # pattern_cost = np.zeros(num_pattern, dtype=np.float64)
    step_cost = np.zeros((num_pattern, max_pattern_len), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if window(i, j, n, m, window_size):
                if i == 0 and j == 0:
                    cost[i, j] = local_cost[i, j]
                    continue
                for p in range(num_pattern):
                    for s in range(max_pattern_len):
                        dx, dy, weight = pattern[p, s, :]
                        ii, jj = int(i + dx), int(j + dy)
                        if ii < 0 or jj < 0:
                            step_cost[p, s] = np.inf
                            continue
                        if weight == -1:
                            step_cost[p, s] = cost[ii, jj]
                        else:
                            step_cost[p, s] = local_cost[ii, jj] * weight
                pattern_cost = step_cost.sum(axis=1)
                min_cost = pattern_cost.min()
                if min_cost != np.inf:
                    cost[i, j] = min_cost
    return cost


def _get_distance(cost, pattern):
    dist = cost[-1, -1]
    normalized_dist = None

    if pattern.is_normalizable:
        n, m = cost.shape
        normalized_dist = pattern.do_normalize(dist, n, m)

    # if dist == np.inf:
    #     raise ValueError("no alignment path found")
    return dist, normalized_dist


@nb.jit(**jitkw)
def _backtrack(direction, pattern):
    i, j = direction.shape
    i -= 1
    j -= 1
    path = np.array(((i, j),), dtype=np.int64)
    path_origin = np.array(((i, j),), dtype=np.int64)

    num_pattern = pattern.shape[0]
    pattern_paths = [_get_local_path2(pattern, pattern_idx) for pattern_idx in range(num_pattern)]

    while i + j > 0:
        pattern_idx = int(direction[i, j])
        local_path = pattern_paths[pattern_idx] + np.array((i,j), dtype=np.int64)
        i, j = local_path[-1]
        path = np.vstack((path, local_path))
        path_origin = np.vstack((path_origin, np.array(((i,j),), dtype=np.int64)))
        # local_path, origin = _get_local_path(pattern, pattern_idx, i, j)
        # path_origin = np.vstack((path_origin, np.array((origin,), dtype=np.int64)))
        # i, j = origin
        # i, j = origin
    return path[::-1], path_origin[::-1]


# TO-DO: move this function to the step pattern? NO! BUT CALL ONLY ONCE
# the output should be the same for a given step pattern. only the i,j change
@nb.jit(**jitkw)
def _get_local_path(pattern, pattern_idx, i, j):
    """Helper function to get local path."""
    origin = (0, 0)
    max_pattern_len = pattern.shape[1]
    # # note: starting point of pattern was already added
    local_path = np.ones((max_pattern_len - 1, 2), dtype=np.int64) * -1
    for s in range(max_pattern_len):
        dx, dy, weight = pattern[pattern_idx, s, :]
        if dx == 0 and dy == 0:
            break  # condition that all step pattern end at the point (0,0)
        ii, jj = int(i + dx), int(j + dy)
        if weight == -1:
            origin = (ii, jj)
        local_path[s, :] = (ii, jj)
    local_path = local_path[:s]
    return local_path[::-1], origin


@nb.jit(**jitkw)
def _get_local_path2(pattern, pattern_idx):
    """Helper function to get local path."""
    max_pattern_len = pattern.shape[1]
    # # note: starting point of pattern was already added
    local_path = np.ones((max_pattern_len - 1, 2), dtype=np.int64) * -1
    for s in range(max_pattern_len):
        dx, dy, weight = pattern[pattern_idx, s, :]
        if dx == 0 and dy == 0:
            break  # condition that all step pattern end at the point (0,0)
        ii, jj = int(dx), int(dy)
        # if weight == -1:
        #     origin = (ii, jj)
        local_path[s, :] = (ii, jj)
    local_path = local_path[:s]
    return local_path[::-1]


# TO-DO
# a) variable time axis: imagine irregularly sampled time series (literature?)
# b) also, implement online dynamic time warping for large time series (bigger than 10000 points, for example)
# alternative to this: downsampled time series, or indexing?
# c) independent vs dependent multidimensional DTW:
# independent version (freedom wrap across all dimensions, simply summing DTW),
# dependent version (cumulative squared euclidean distance across all dimensions)


x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 3, 8])


x = np.array([[1, 1], [2, 2], [2, 3], [4, 4]])
y = np.array([[2, 2], [3, 3], [3, 3], [8, 8]])

x = np.array([1, 2, 3, 4, 3, 2, 1, 1, 1, 2])
y = np.array([0, 1, 1, 2, 3, 4, 3, 2, 1, 1])

# np.random.seed(1234)
# x = np.sin(2 * np.pi * 3.1 * np.linspace(0, 1, 101))
# x += np.random.rand(x.size)
# y = np.sin(2 * np.pi * 3 * np.linspace(0, 1, 120))
# y += np.random.rand(y.size)

result = dtw(x, y)

# print(nb.threading_layer())
# print(nb.get_num_threads())

# t1 = timeit.timeit(lambda: dtw(x, y), number=1)
# t2 = timeit.timeit(lambda: dtw(x, y), number=1)
# print(t1, t2, t1 / t2)
