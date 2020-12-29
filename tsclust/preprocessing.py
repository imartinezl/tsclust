#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# stuff of resampling, normalizing, etc.

import numpy as np
import numba as nb

from utils import check_equal_size, to_time_series_dataset, ts_size
from scipy.interpolate import interp1d


def aggregate(X):
    n_ts, max_sz, d = X.shape
    X_out = np.empty((n_ts, 1, d))
    for i in range(n_ts):
        X_out[i] = np.nanmean(X[i], axis=0, keepdims=True)
    return X_out

def resampler(X, sz=None):
    X_ = to_time_series_dataset(X)
    if sz is None:
        sz = X.shape[1]
    elif sz == 1:
        return aggregate(X)
    n_ts, max_sz, d = X_.shape
    equal_size = check_equal_size(X_)
    X_out = np.empty((n_ts, sz, d))
    for i in range(n_ts):
        xnew = np.linspace(0, 1, sz)
        if not equal_size:
            max_sz = ts_size(X_[i])
        for di in range(d):
            f = interp1d(np.linspace(0, 1, max_sz), X_[i, :max_sz, di], kind="slinear")
            X_out[i, :, di] = f(xnew)
    return X_out