#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
import timeit

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


def dtw(
    x,
    y,
    local_dist="euclidean",
    step_pattern="symmetric2",
    window_name="none",
    window_size=20,
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

    path = None
    path_aux = None
    if compute_path:
        path, path_aux = _backtrack(direction, pattern.array)

    print(path, path_aux)
    dtw_result = DtwResult(x, y, cost, path, window,
                           pattern, dist, normalized_dist)
    # dtw_result.plot_cost_matrix()
    # dtw_result.plot_pattern()
    # dtw_result.plot_path()
    # # dtw_result.plot_ts_subplot(x, "Query")
    # dtw_result.plot_ts_overlay(x, "Query")
    # dtw_result.plot_ts_overlay(y, "Reference")
    dtw_result.plot_summary()

    plt.show()
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
    path_aux = np.array(((i, j),), dtype=np.int64)
    while i + j > 0:
        pattern_idx = int(direction[i, j])
        local_path, origin = _get_local_path(pattern, pattern_idx, i, j)
        path = np.vstack((path, local_path))
        path_aux = np.vstack((path_aux,  np.array((origin,), dtype=np.int64)))
        i, j = origin
    return path[::-1], path_aux[::-1]


@nb.jit(**jitkw)
def _get_local_path(pattern, pattern_idx, i, j):
    """Helper function to get local path."""
    origin = (0, 0)
    max_pattern_len = pattern.shape[1]
    # # note: starting point of pattern was already added
    local_path = np.ones((max_pattern_len-1, 2), dtype=np.int64) * -1
    for s in range(max_pattern_len):
        dx, dy, weight = pattern[pattern_idx, s, :]
        if dx == 0 and dy == 0: # condition that all step pattern end at the point (0,0)
            break
        ii, jj = int(i + dx), int(j + dy)
        if weight == -1:
            origin = (ii, jj)
        local_path[s, :] = (ii, jj)
    local_path = local_path[:s]
    return local_path[::-1], origin


class DtwResult:
    """Result of DTW.

    Attributes
    ----------
    path : 2d array
        Alignment path.
        * First column: query path array
        * Second column: reference path array
    distance : float
        Alignment distance.
    normalized_distance : float
        Normalized alignment distance.
    """

    def __init__(self, x, y, cost, path, window, pattern, dist, normalized_dist):
        self.x = x
        self.y = y
        self.cost = cost

        if path is None:
            self.compute_path = False
        else:
            self.compute_path = True
            self.path = path

        self.window = window
        self.pattern = pattern
        self.dist = dist
        self.normalized_dist = normalized_dist

    def get_warping_path(self, target="query"):
        """Get warping path.

        Parameters
        ----------
        target : string, "query" or "reference"
            Specify the target to be warped.

        Returns
        -------
        warping_index : 1D array
            Warping index.

        """
        if target not in ("query", "reference"):
            raise ValueError("target argument must be 'query' or 'reference'")
        if target == "reference":
            xp = self.path[:, 0]  # query path
            yp = self.path[:, 1]  # reference path
        else:
            yp = self.path[:, 0]  # query path
            xp = self.path[:, 1]  # reference path
        interp_func = interp1d(xp, yp, kind="linear")
        # get warping index as float values and then convert to int
        # note: Ideally, the warped value should be calculated as mean.
        #       (in this implementation, just use value corresponds to rounded-up index)
        warping_index = interp_func(np.arange(xp.min(), xp.max() + 1)).astype(np.int64)
        # the most left side gives nan, so substitute first index of path
        warping_index[0] = yp.min()

        return warping_index

    def plot_cost_matrix(self, labels = True, ax=None, pax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        im = ax.imshow(self.cost.T, origin='lower', cmap='inferno', vmin=0, aspect='auto')
        if pax is None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            pax = divider.append_axes("bottom", size="5%", pad=0.05)
            plt.colorbar(im, cax=pax, orientation="horizontal")
        else:
            plt.colorbar(im, cax=pax)
        if self.compute_path:
            ax.plot(self.path[:, 0], self.path[:, 1], color="white", alpha=0.9)
            # ax.plot(self.path[:, 0], self.path[:, 1], c="red", marker='o', markersize=1)
        if labels:
            ax.set_xlabel("Query Index")
            ax.set_ylabel("Reference Index")
            ax.set_title("Cost matrix")
        else:
            ax.set_axis_off()
            pass

        return ax

    def plot_path(self, ax):
        if not self.compute_path:
            raise Exception("Alignment path not calculated.")
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(self.path[:, 0], self.path[:, 1])
        ax.set_title("Alignment Path")
        ax.set_xlabel("Query Index")
        ax.set_ylabel("Reference Index")
        return ax

    def plot_pattern(self, labels=True, ax=None):
        # if ax is None:
        #     fig, ax = plt.subplots(1)
        return self.pattern.plot(labels, ax)

    def plot_ts_subplot(self, data, title):
        d = data.shape[1]
        fig, ax = plt.subplots(nrows=d, ncols=1, sharex=True, constrained_layout=True)
        for i in range(d):
            ax[i].plot(data[:,i])
            ax[i].set_ylabel(f"dim.{i}")
        plt.xlabel("Time Index")
        fig.suptitle(title)


    def plot_ts_overlay(self, data, title, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)
        for i in range(data.shape[1]):
            ax.plot(data[:,i])
        ax.set_xlabel("Time Index")
        ax.set_title(title)

    def plot_ts_query(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)
        n, d = self.x.shape
        t = np.arange(n)
        for i in range(d):
            ax.plot(t, self.x[:,i], color="black", alpha=0.9)
        ax.set_xlabel("Query X")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


    def plot_ts_reference(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)
        n, d = self.y.shape
        t = np.arange(n)
        for i in range(d):
            ax.plot(self.y[:,i], t, color="black", alpha=0.9)
        ax.invert_xaxis()
        ax.set_ylabel("Reference Y")
        ax.yaxis.tick_right()
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis="y", which="both", right=False, labelright=False)

    def plot_summary(self):
        n = x.shape[0]
        m = y.shape[0]
        h = min(n, m)/2.5
        fig = plt.figure(constrained_layout=True, figsize=(10,10))
        spec = fig.add_gridspec(ncols=3, nrows=2,
                                 width_ratios=[h, n, n*0.05],
                                 height_ratios=[m, h])
        title = f"DTW: {self.dist:.2f} "
        if self.normalized_dist is not None:
            title += f"Normalized: {self.normalized_dist:.2f}"
        fig.suptitle(title)

        ax = fig.add_subplot(spec[0, 1])
        pax = fig.add_subplot(spec[0, 2])
        self.plot_cost_matrix(False, ax, pax)

        ax = fig.add_subplot(spec[1, 0])
        self.plot_pattern(False, ax)

        ax = fig.add_subplot(spec[1, 1])
        self.plot_ts_query(ax)

        ax = fig.add_subplot(spec[0, 0])
        self.plot_ts_reference(ax)






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
