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

jitkwp = {
    "nopython": True,
    "nogil": True,
    "cache": False,
    "error_model": "numpy",
    "fastmath": True,
    "debug": False,
    "parallel": True,
}
nb.config.THREADING_LAYER = "omp"

@nb.jit(**jitkw)
def softdtw( # symmetric1
    x,
    y,
    gamma,
    local_dist, # choose jit method
    window, # choose jit method
    window_size=None
):
    x = validate_time_series(x)
    y = validate_time_series(y)

    local_cost = get_local_cost(x, y, local_dist, window, window_size)
    cost = get_cost(local_cost, gamma, window, window_size)
    gradient = get_gradient(local_cost, cost, gamma, window, window_size)

    return local_cost, cost, gradient
    # path = get_warping_path(cost)
    # dist = cost[-1, -1]
    # return cost, path, dist

@nb.jit(**jitkwp)
def get_local_cost(x, y, local_dist, window, window_size):
    n = len(x)
    m = len(y)
    local_cost = np.full((n, m), np.inf, dtype=np.float64)
    for i in nb.prange(n):
        for j in nb.prange(m):
            local_cost[i, j] = local_dist(x[i], y[j])
    return local_cost

@nb.jit(**jitkw)
def get_cost(local_cost, gamma, window, window_size):
    n = x.shape[0]
    m = y.shape[0]
    cost = np.full((n, m), np.inf, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if window(i, j, n, m, window_size):
                if (i == 0) and (j == 0):
                    cost[i, j] = local_cost[i,j]
                elif (i == 0):
                    cost[i, j] = cost[i, j - 1] + local_cost[i,j]
                elif (j == 0):
                    cost[i, j] = cost[i - 1, j] + local_cost[i,j]
                else:
                    cost[i, j] = local_cost[i,j] + softmin(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j], gamma)
    return cost

@nb.jit(**jitkw)
def softmin(a, b, c, gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    max_val = max(a,b,c)
    tmp = 0.0
    tmp += np.exp(a - max_val)
    tmp += np.exp(b - max_val)
    tmp += np.exp(c - max_val)

    return -gamma * (np.log(tmp) + max_val)


@nb.jit(**jitkw)
def get_gradient(local_cost, cost, gamma, window, window_size):
    # ONLY WORKS WITH EUCLIDEAN SQUARED DISTANCE
    m, n = local_cost.shape
    gradient = np.empty((m,n), dtype=np.float64)

    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            if window(i, j, n, m, window_size):
                if (i == n-1) and (j == m-1):
                    gradient[i,j] = 1
                elif (i == n-1):
                    b = np.exp((cost[i, j+1] - cost[i,j] - local_cost[i, j+1]) / gamma)
                    gradient[i, j] = gradient[i,j+1] * b
                elif (j == m-1):
                    a = np.exp((cost[i+1, j] - cost[i,j] - local_cost[i+1, j]) / gamma)
                    gradient[i, j] = gradient[i + 1, j] * a
                else:
                    a = np.exp((cost[i+1, j] - cost[i,j] - local_cost[i+1, j]) / gamma)
                    b = np.exp((cost[i, j+1] - cost[i,j] - local_cost[i, j+1]) / gamma)
                    c = np.exp((cost[i+1, j+1] - cost[i,j] - local_cost[i+1, j+1]) / gamma)
                    gradient[i, j] = gradient[i+1, j] * a + gradient[i, j+1] * b + gradient[i+1,j+1] * c
    return gradient

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

# x = np.array([[1, 1], [2, 2], [2, 3], [4, 4]])
# y = np.array([[2, 2], [3, 3], [3, 3], [8, 8]])
#
# x = np.array([1, 2, 3, 4, 3, 2, 1, 1, 1, 2])
# y = np.array([0, 1, 1, 2, 3, 4, 3, 2, 1, 1])
#
# np.random.seed(1234)
# n = 1000
# x = np.sin(2 * np.pi * 3.1 * np.linspace(0, 1, n))
# x += np.random.rand(x.size)
# y = np.sin(2 * np.pi * 3 * np.linspace(0, 1, n))
# y += np.random.rand(y.size)


from metrics import sqeuclidean
from window import no_window, itakura_window

t1 = timeit.timeit(lambda: softdtw(x, y, 1, sqeuclidean, no_window), number=1)
rep = 10
t2 = timeit.timeit(lambda: softdtw(x, y, 1, sqeuclidean, no_window), number=rep)
print("SOFTDTW:", f"{t1:.4f}", f"{t2/rep:.6f}", f"{t1 / t2*rep:.4f}")

# a, b, c = softdtw(x, y, 1, sqeuclidean, no_window)
# print(a)
# print(b)
# print(c)

# from result import DtwResult
# cost, path, dist = dtw(x, y, euclidean, no_window)
# result = DtwResult(x, y, cost, path, no_window, None, dist, None)
# result.plot_cost_matrix()
# result.plot_path()
# result.plot_warp()
# plt.show()
