#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numba as nb

jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": True,
    "error_model": "numpy",
    "fastmath": True,
    "debug": False,
    "parallel": False,
}

all = {
    "minkowski",
    "euclidean",
    "sqeuclidean",
    "correlation",
    "cosine",
    "seuclidean",
    "cityblock",
    "mahalanobis",
    "chebyshev",
    "braycurtis",
    "canberra",
}


@nb.jit(**jitkw)
def minkowski(u, v, p=2, w=None):
    s = 0.0
    if w is not None:
        if p == 1:
            root_w = w
        if p == 2:
            # better precision and speed
            root_w = np.sqrt(w)
        else:
            root_w = np.power(w, 1 / p)
    for k in nb.prange(len(u)):
        tmp = u[k] - v[k]
        if w is not None:
            tmp = tmp * root_w[k]
        s += np.power(tmp, p)
    s = np.power(s, 1 / p)
    return s


@nb.jit(**jitkw)
def euclidean(u, v):
    s = 0.0
    for k in nb.prange(len(u)):
        tmp = u[k] - v[k]
        s += tmp * tmp
    s = np.sqrt(s)
    return s


@nb.jit(**jitkw)
def sqeuclidean(u, v):
    s = 0.0
    for k in nb.prange(len(u)):
        tmp = u[k] - v[k]
        s += tmp * tmp
    return s


@nb.jit(**jitkw)
def correlation(u, v, w=None, centered=True):
    umu = 0
    vmu = 0
    N = len(u)
    if centered and w is not None:
        for k in nb.prange(N):
            umu += u[k] * w[k]
            vmu += v[k] * w[k]
        u = u - umu / N
        v = v - vmu / N
    elif centered:
        for k in nb.prange(N):
            umu += u[k]
            vmu += v[k]
        u = u - umu / N
        v = v - vmu / N
    uu = 0
    vv = 0
    s = 0
    for k in nb.prange(N):
        uu += u[k] ** 2
        vv += v[k] ** 2
        s += u[k] * v[k]
    uu = np.sqrt(uu)
    vv = np.sqrt(vv)
    s = s / (uu * vv)
    s = 1 - s
    return s


@nb.jit(**jitkw)
def cosine(u, v, w=None):
    return correlation(u, v, w=w, centered=False)


@nb.jit(**jitkw)
def seuclidean(u, v, V):
    if len(V) != len(u) or len(u) != len(v):
        raise TypeError("V must be a 1-D array of the same dimension as u and v.")
    # return euclidean(u, v, w=1 / V)
    return minkowski(u, v, p=2, w=1 / V)


@nb.jit(**jitkw)
def cityblock(u, v, w=None):
    s = 0.0
    for k in nb.prange(len(u)):
        tmp = abs(u[k] - v[k])
        if w is not None:
            tmp *= w[k]
        s += tmp
    return s


@nb.jit(**jitkw)
def mahalanobis(u, v, VI):
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)


@nb.jit(**jitkw)
def chebyshev(u, v, w=None):
    s_max = -np.inf
    for k in nb.prange(len(u)):
        s = abs(u[k] - v[k])
        if s > s_max:
            s_max = s
    return s


@nb.jit(**jitkw)
def braycurtis(u, v, w=None):
    l1_diff = 0.0
    l1_sum = 0.0
    for k in nb.prange(len(u)):
        tmp_diff = abs(u[k] - v[k])
        tmp_sum = abs(u[k] + v[k])
        if w is not None:
            tmp_diff *= w[k]
            tmp_sum *= w[k]
        l1_diff += tmp_diff
        l1_sum += tmp_sum
    return l1_diff / l1_sum


@nb.jit(**jitkw)
def canberra(u, v, w=None):
    d = 0.0
    for k in nb.prange(len(u)):
        abs_uv = abs(u[k] - v[k])
        abs_u = abs(u[k])
        abs_v = abs(v[k])
        tmp = abs_uv / (abs_u + abs_v)
        if w is not None:
            tmp *= w[k]
        d += tmp
    return d


def get_metric(metric_str):
    if metric_str == "minkowski":
        return minkowski
    elif metric_str == "euclidean":
        return euclidean
    elif metric_str == "sqeuclidean":
        return sqeuclidean
    elif metric_str == "correlation":
        return correlation
    elif metric_str == "cosine":
        return cosine
    elif metric_str == "seuclidean":
        return seuclidean
    elif metric_str == "cityblock":
        return cityblock
    elif metric_str == "mahalanobis":
        return mahalanobis
    elif metric_str == "chebyshev":
        return chebyshev
    elif metric_str == "braycurtis":
        return braycurtis
    elif metric_str == "canberra":
        return canberra
    else:
        raise NotImplementedError("given metric not supported")


@nb.jit(**jitkw)
def compute_metric(metric_str, u, v, w=None, p=2, centered=True):
    if metric_str == "minkowski":
        return minkowski(u, v, p, w)
    elif metric_str == "euclidean":
        return euclidean(u, v)
    elif metric_str == "sqeuclidean":
        return sqeuclidean(u, v)
    elif metric_str == "correlation":
        return correlation(u, v, w, centered)
    elif metric_str == "cosine":
        return cosine(u, v, w)
    # elif metric_str == "seuclidean":
    #     return seuclidean(u, v, V)
    elif metric_str == "cityblock":
        return cityblock(u, v, w)
    # elif metric_str == "mahalanobis":
    #     return mahalanobis(u, v, VI)
    elif metric_str == "chebyshev":
        return chebyshev(u, v, w)
    elif metric_str == "braycurtis":
        return braycurtis(u, v, w)
    elif metric_str == "canberra":
        return canberra(u, v, w)
    else:
        raise NotImplementedError("given metric not supported")
