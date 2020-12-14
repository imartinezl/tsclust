#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains code that was borrowed from numpy.

Copyright (c) 2005, NumPy Developers
All rights reserved.
"""


import numpy as np
import numba as nb
from numba.extending import overload

jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": True,
    "error_model": "numpy",
    "fastmath": True,
    "debug": True,
}

__all__ = {
    "minkowski",
    "wminkowski",
    "euclidean",
    "sqeuclidean",
    "correlation",
    "cosine",
    "hamming",
    "seuclidean",
    "cityblock",
    "mahalanobis",
    "chebyshev",
    "braycurtis",
    "canberra",
}


@overload(np.average)
def np_average(arr, axis=None, weights=None):
    # https://github.com/numba/numba/pull/3655
    if weights is None or isinstance(weights, nb.types.NoneType):

        def np_average_impl(arr, axis=None, weights=None):
            arr = np.asarray(arr)
            return np.mean(arr)

    else:
        if axis is None or isinstance(axis, nb.types.NoneType):

            def np_average_impl(arr, axis=None, weights=None):
                arr = np.asarray(arr)
                weights = np.asarray(weights)

                if arr.shape != weights.shape:
                    if axis is None:
                        raise TypeError(
                            "Numba does not support average when shapes of a and weights "
                            "differ."
                        )
                    if weights.ndim != 1:
                        raise TypeError(
                            "1D weights expected when shapes of a and weights differ."
                        )

                scl = np.sum(weights)
                if scl == 0.0:
                    raise ZeroDivisionError("Weights sum to zero, can't be normalized.")

                avg = np.sum(np.multiply(arr, weights)) / scl
                return avg

        else:

            def np_average_impl(arr, axis=None, weights=None):
                raise TypeError("Numba does not support average with axis.")

    return np_average_impl


@nb.jit(**jitkw)
def _validate_weights(w=None, dtype=nb.float64):
    w = _validate_vector(w, dtype=dtype)
    if np.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


@nb.jit(**jitkw)
def _validate_vector(u=None, dtype=nb.float64):
    u = np.asarray(u, dtype=dtype).flatten()  # .squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


@nb.jit(**jitkw)
def minkowski(u, v, p=2, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    if p < 1:
        raise ValueError("p must be at least 1")
    u_v = u - v
    # if w is not None:
    if w is not None:
        # w = _validate_weights(w) # gives error when w is None
        if p == 1:
            root_w = w
        if p == 2:
            # better precision and speed
            root_w = np.sqrt(w)
        else:
            root_w = np.power(w, 1 / p)
        u_v = root_w * u_v
    dist = np.linalg.norm(u_v, ord=p)  # exists implementation in numba
    return dist


@nb.jit(**jitkw)
def wminkowski(u, v, p, w):
    # w = _validate_weights(w)
    return minkowski(u, v, p=p, w=w ** p)


@nb.jit(**jitkw)
def euclidean(u, v, w=None):
    return minkowski(u, v, p=2, w=w)


@nb.jit(**jitkw)
def sqeuclidean(u, v, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    u_v = u - v
    u_v_w = u_v  # only want weights applied once
    if w is not None:
        # w = _validate_weights(w)
        u_v_w = w * u_v
    return np.dot(u_v, u_v_w)


@nb.jit(**jitkw)
def correlation(u, v, w=None, centered=True):
    u = _validate_vector(u)
    v = _validate_vector(v)
    # if w is not None:
    #     w = _validate_weights(w)
    if centered:
        umu = np.average(u, weights=w)
        vmu = np.average(v, weights=w)
        u = u - umu
        v = v - vmu
    uv = np.average(u * v, weights=w)
    uu = np.average(np.square(u), weights=w)
    vv = np.average(np.square(v), weights=w)
    dist = 1.0 - uv / np.sqrt(uu * vv)
    # Return absolute value to avoid small negative value due to rounding
    return np.abs(dist)


@nb.jit(**jitkw)
def cosine(u, v, w=None):
    return correlation(u, v, w=w, centered=False)


@nb.jit(**jitkw)
def hamming(u, v, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    if u.shape != v.shape:
        raise ValueError("The 1d arrays must have equal lengths.")
    u_ne_v = u != v
    # if w is not None:
    # w = _validate_weights(w)
    return np.average(u_ne_v, weights=w)


@nb.jit(**jitkw)
def seuclidean(u, v, V):
    u = _validate_vector(u)
    v = _validate_vector(v)
    V = _validate_vector(V, dtype=np.float64)
    if V.shape[0] != u.shape[0] or u.shape[0] != v.shape[0]:
        raise TypeError("V must be a 1-D array of the same dimension as u and v.")
    return euclidean(u, v, w=1 / V)


@nb.jit(**jitkw)
def cityblock(u, v, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = np.abs(u - v)
    if w is not None:
        # w = _validate_weights(w)
        l1_diff = w * l1_diff
    return l1_diff.sum()


@nb.jit(**jitkw)
def mahalanobis(u, v, VI):
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)


@nb.jit(**jitkw)
def chebyshev(u, v, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    if w is not None:
        # w = _validate_weights(w)
        has_weight = w > 0
        if has_weight.sum() < w.size:
            u = u[has_weight]
            v = v[has_weight]
    return np.max(np.abs(u - v))


@nb.jit(**jitkw)
def braycurtis(u, v, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = np.abs(u - v)
    l1_sum = np.abs(u + v)
    if w is not None:
        # w = _validate_weights(w)
        l1_diff = w * l1_diff
        l1_sum = w * l1_sum
    return l1_diff.sum() / l1_sum.sum()


@nb.jit(**jitkw)
def canberra(u, v, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    # if w is not None:
    #     w = _validate_weights(w)
    # with np.errstate(invalid='ignore'):
    abs_uv = np.abs(u - v)
    abs_u = np.abs(u)
    abs_v = np.abs(v)
    d = abs_uv / (abs_u + abs_v)
    if w is not None:
        d = w * d
    d = np.nansum(d)
    return d


def get_metric(metric_str):
    if metric_str == "minkowski":
        return minkowski
    elif metric_str == "wminkowski":
        return wminkowski
    elif metric_str == "euclidean":
        return euclidean
    elif metric_str == "sqeuclidean":
        return sqeuclidean
    elif metric_str == "correlation":
        return correlation
    elif metric_str == "cosine":
        return cosine
    elif metric_str == "hamming":
        return hamming
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
