#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import timeit

from utils import validate_time_series
from result import DtwResult


jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": False,
    "error_model": "numpy",
    "fastmath": True,
    "debug": False,
    "parallel": False,
}


@nb.jit(**jitkw)
def dtw( # symmetric1
    x,
    y,
    local_dist, # choose jit method
    window, # choose jit method
    window_size=None
):
    x = validate_time_series(x)
    y = validate_time_series(y)

    cost = get_cost(x, y, local_dist, window, window_size)
    path = get_warping_path(cost)
    dist = cost[-1, -1]
    normalized_dist = dist / (len(x) + len(y))
    return cost, path, dist, normalized_dist

@nb.jit(**jitkw)
def get_cost(x, y, local_dist, window, window_size):
    n = x.shape[0]
    m = y.shape[0]
    cost = np.full((n, m), np.inf, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if window(i, j, n, m, window_size):
                local_cost = local_dist(x[i], y[j])
                if (i == 0) and (j == 0):
                    cost[i, j] = local_cost
                elif (i == 0):
                    cost[i, j] = cost[i, j - 1] + local_cost
                elif (j == 0):
                    cost[i, j] = cost[i - 1, j] + local_cost
                else:
                    cost[i, j] = local_cost + min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j])
    return cost

@nb.jit(**jitkw)
def get_warping_path(cost):
    n, m = cost.shape
    i = n - 1
    j = m - 1
    path = []
    while i + j >= 0:
        path.append([i, j])
        c_diagonal = cost[i - 1, j - 1]
        c_vertical = cost[i - 1, j]
        c_horizontal = cost[i, j - 1]

        if c_diagonal <= c_horizontal:
            if c_diagonal <= c_vertical:
                i -= 1
                j -= 1
            else:
                i -= 1
        elif c_horizontal <= c_vertical:
            j -= 1
        else:
            i -= 1
    path = np.flipud(np.array(path))

    return path

# TO-DO
# a) variable time axis: imagine irregularly sampled time series (literature?)
# b) also, implement online dynamic time warping for large time series (bigger than 10000 points, for example)
# alternative to this: downsampled time series, or indexing?
# c) independent vs dependent multidimensional DTW:
# independent version (freedom wrap across all dimensions, simply summing DTW),
# dependent version (cumulative squared euclidean distance across all dimensions)

# print(nb.threading_layer())
# print(nb.get_num_threads())

x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 3, 8])

x = np.array([[1, 1], [2, 2], [2, 3], [4, 4]])
y = np.array([[2, 2], [3, 3], [3, 3], [8, 8]])

x = np.array([1, 2, 3, 4, 3, 2, 1, 1, 1, 2])
y = np.array([0, 1, 1, 2, 3, 4, 3, 2, 1, 1])

np.random.seed(1234)
n = 1000
x = np.sin(2 * np.pi * 3.1 * np.linspace(0, 1, n))
x += np.random.rand(x.size)
y = np.sin(2 * np.pi * 3 * np.linspace(0, 1, n))
y += np.random.rand(y.size)


from metrics import euclidean, sqeuclidean
from window import no_window, itakura_window

t1 = timeit.timeit(lambda: dtw(x, y, euclidean, no_window), number=1)
rep = 10
t2 = timeit.timeit(lambda: dtw(x, y, euclidean, no_window), number=rep)
print("DTW CLASSIC:", f"{t1:.4f}", f"{t2/rep:.6f}", f"{t1 / t2*rep:.4f}")

cost, path, dist, normalized_dist = dtw(x, y, euclidean, no_window)
print(dist, normalized_dist)

# from result import DtwResult
# cost, path, dist, normalized_dist = dtw(x, y, euclidean, no_window)
# result = DtwResult(x, y, cost, path, no_window, None, dist, None)
# result.plot_cost_matrix()
# result.plot_path()
# result.plot_warp()
# plt.show()

# Dynamic programming: recursion + memoization + guessing
# memoize and reuse solutions to subproblems that help solve the problem
# time = # subproblems x time_per_subproblem
# Bottom-Up DP algorithm: topological sort of subproblem dependency DAG (directed acyclic graph)
# practically faster, no recursion, and more obvious analysis