#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains code that was borrowed from tslearn.

BSD 2-Clause License

Copyright (c) 2017, Romain Tavenard
All rights reserved.
"""

import numpy as np
import numba as nb
import warnings

try:
    from scipy.io import arff

    HAS_ARFF = True
except:
    HAS_ARFF = False


jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": True,
    "error_model": "numpy",
    "fastmath": True,
    "debug": True,
}


@nb.jit(**jitkw)
def validate_time_series(x):
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T
    return x


def _load_arff_uea(dataset_path):
    if not HAS_ARFF:
        raise ImportError(
            "scipy 1.3.0 or newer is required to load "
            "time series datasets from arff format."
        )
    data, meta = arff.loadarff(dataset_path)
    names = meta.names()  # ["input", "class"] for multi-variate

    # firstly get y_train
    y_ = data[names[-1]]  # data["class"]
    y = np.array(y_).astype("str")

    # get x_train
    if len(names) == 2:  # len=2 => multi-variate
        x_ = data[names[0]]
        x_ = np.asarray(x_.tolist())

        nb_example = x_.shape[0]
        nb_channel = x_.shape[1]
        length_one_channel = len(x_.dtype.descr)
        x = np.empty([nb_example, length_one_channel, nb_channel])

        for i in range(length_one_channel):
            # x_.dtype.descr: [('t1', '<f8'), ('t2', '<f8'), ('t3', '<f8')]
            time_stamp = x_.dtype.descr[i][0]  # ["t1", "t2", "t3"]
            x[:, i, :] = x_[time_stamp]

    else:  # uni-variate situation
        x_ = data[names[:-1]]
        x = np.asarray(x_.tolist(), dtype=np.float32)
        x = x.reshape(len(x), -1, 1)

    return x, y


def _load_txt_uea(dataset_path):
    try:
        data = np.loadtxt(dataset_path, delimiter=None)
        X = to_time_series_dataset(data[:, 1:])
        y = data[:, 0].astype("str")
        return X, y
    except:
        return None, None


def to_time_series_dataset(dataset, dtype=np.float):
    # try:
    #     import pandas as pd
    #     if isinstance(dataset, pd.DataFrame):
    #         return to_time_series_dataset(np.array(dataset))
    # except ImportError:
    #     pass
    if len(dataset) == 0:
        return np.zeros((0, 0, 0))
    if np.array(dataset[0]).ndim == 0:
        dataset = [dataset]
    n_ts = len(dataset)
    max_sz = max([ts_size(to_time_series(ts, remove_nans=True)) for ts in dataset])
    d = to_time_series(dataset[0]).shape[1]
    dataset_out = np.zeros((n_ts, max_sz, d), dtype=dtype) + np.nan
    for i in range(n_ts):
        ts = to_time_series(dataset[i], remove_nans=True)
        dataset_out[i, : ts.shape[0]] = ts
    return dataset_out.astype(dtype)


def to_time_series(ts, remove_nans=False):
    ts_out = _arraylike_copy(ts)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != np.float:
        ts_out = ts_out.astype(np.float)
    if remove_nans:
        ts_out = ts_out[: ts_size(ts_out)]
    return ts_out


def _arraylike_copy(arr):
    if type(arr) != np.ndarray:
        return np.array(arr)
    else:
        return arr.copy()


def ts_size(ts):
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and not np.any(np.isfinite(ts_[sz - 1])):
        sz -= 1
    return sz


def check_dims(X, X_fit_dims=None, extend=True, check_n_features_only=False):
    if X is None:
        raise ValueError("X is equal to None!")

    if extend and len(X.shape) == 2:
        warnings.warn(
            "2-Dimensional data passed. Assuming these are "
            "{} 1-dimensional timeseries".format(X.shape[0])
        )
        X = X.reshape((X.shape) + (1,))

    if X_fit_dims is not None:
        if check_n_features_only:
            if X_fit_dims[2] != X.shape[2]:
                raise ValueError(
                    "Number of features of the provided timeseries"
                    "(last dimension) must match the one of the fitted data!"
                    " ({} and {} are passed shapes)".format(X_fit_dims, X.shape)
                )
        else:
            if X_fit_dims[1:] != X.shape[1:]:
                raise ValueError(
                    "Dimensions of the provided timeseries"
                    "(except first) must match those of the fitted data!"
                    " ({} and {} are passed shapes)".format(X_fit_dims, X.shape)
                )
    return X
