#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numba as nb
from tsclust.dtw_classic import dtw
from metrics import euclidean, sqeuclidean
from window import no_window

import matplotlib.pyplot as plt

jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": False,
    "error_model": "numpy",
    "fastmath": True,
    "debug": False,
    "parallel": False,
}

# BATCH

# Euclidean barycenter
def euclidean_barycenter(X, weights=None):
    return np.average(X, axis=0, weights=weights)

# NonLinear Alignment and Averaging Filters (NLAAF): Tournament scheme
def nlaaf(D, W=None):
    if W is None:
        D = D.copy()
        # np.random.shuffle(D)
        W = [1] * len(D)
    pairs = []
    weights = []
    for i in range(0, len(D) - 1, 2):
        pairs.append((D[i], D[i + 1]))
        weights.append((W[i], W[i + 1]))

    if len(D) % 2 != 0:
        D = [D[-1]]
        W = [W[-1]]
    else:
        D = []
        W = []

    for i in range(len(pairs)):
        x, y = pairs[i]
        w1, w2 = weights[i]
        cost, path, dist, normalized_dist = dtw(x, y, euclidean, no_window)
        warp = [(x[p], y[q]) for p, q in path]
        mean = np.average(warp, axis=1, weights=[w1, w2])
        # std = np.std(warp, axis=1)
        D.append(mean)
        W.append(w1 + w2)

    if len(D) == 1:
        return D[-1]
    else:
        return nlaaf(D, W)


# Cross-Word Reference Template (CWRT)

# medoid estimation
@nb.jit(**jitkw)
def get_medoid(X):
    medoid = None
    min_dist = np.inf
    for x in X:
        sum_dist = 0
        for y in X:
            cost, path, dist, normalized_dist = dtw(x, y, euclidean, no_window)
            sum_dist += dist
        if sum_dist < min_dist:
            min_dist = sum_dist
            medoid = x
    return medoid

def cwrt(X, medoid=None):
    if medoid is None:
        medoid = get_medoid(X)
    medoid_assignments = [[] for i in range(len(medoid))]
    for x in X:
        cost, path, dist, normalized_dist = dtw(medoid, x, euclidean, no_window)
        for p, q in path:
            medoid_assignments[p].append(x[q])

    mean = np.array([np.mean(x) for x in medoid_assignments])
    std = np.array([np.std(x) for x in medoid_assignments])
    return mean, std, medoid


def inertia(X, mean):
    total = 0.0
    for x in X:
        cost, path, dist, normalized_dist = dtw(mean, x, euclidean, no_window)
        total += dist
    return total

def irr(X, centroid, medoid=None):
    if medoid is None:
        medoid = get_medoid(X)
    total_centroid = 0.0
    total_medoid = 0.0
    for x in X:
        cost, path, dist, normalized_dist = dtw(x, centroid, euclidean, no_window)
        # https://stats.stackexchange.com/questions/55083/dynamic-time-warping-and-normalization
        # total_centroid += dist / (len(x) + len(centroid)) #len(path)
        total_centroid += normalized_dist
        cost, path, dist, normalized_dist = dtw(x, medoid, euclidean, no_window)
        # total_medoid += dist / (len(x) + len(medoid)) #len(path)
        total_medoid += normalized_dist
    return 1 - (total_centroid / total_medoid)


def _set_weights(w, n):
    if w is None or len(w) != n:
        w = np.ones((n, ))
    return w


from softdtw import differentiate
def _softdtw_func(Z, X, weights, barycenter, gamma):
    # Compute objective value and grad at Z.
    Z = Z.reshape(barycenter.shape)
    G = np.zeros_like(Z)
    obj = 0

    for i in range(len(X)):
        # value, G_tmp = differentiate(Z, X[i], gamma)
        value, G_tmp = differentiate(Z, X[i], gamma, sqeuclidean, no_window, None)
        G += weights[i] * G_tmp
        obj += weights[i] * value

    return obj, G.ravel()

from utils import check_equal_size, to_time_series_dataset, to_time_series
from scipy.optimize import minimize
from preprocessing import resampler

def softdtw_barycenter(X, gamma=1.0, weights=None, method="L-BFGS-B", tol=1e-3,
                       max_iter=50, init=None):
    X_ = to_time_series_dataset(X)
    weights = _set_weights(weights, X_.shape[0])
    if init is None:
        if check_equal_size(X_):
            barycenter = euclidean_barycenter(X_, weights)
        else:
            resampled_X = resampler(X, sz=X_.shape[1])
            barycenter = euclidean_barycenter(resampled_X, weights)
    else:
        barycenter = init

    if max_iter > 0:
        X_ = np.array([to_time_series(d, remove_nans=True) for d in X_])

        def f(Z):
            return _softdtw_func(Z, X_, weights, barycenter, gamma)

        _softdtw_func(barycenter.ravel(), X_, weights, barycenter, gamma)
        # The function works with vectors so we need to vectorize barycenter.
        res = minimize(f, barycenter.ravel(), method=method, jac=True, tol=tol,
                       options=dict(maxiter=max_iter, disp=False))
        return res.x.reshape(barycenter.shape)
    else:
        return barycenter



medoid = get_medoid(np.array([[0],[0]]))

from tsclust.datasets import UCR_UEA_datasets
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("ShakeGestureWiimoteZ")
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("CBF")
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("Trace")
X = X_train[y_train == '2.0']
a = euclidean_barycenter(X)
# weights = _set_weights(None, X.shape[0])
# _softdtw_func(X[0], X, weights, X[2], 1.0)
# value, G_tmp = differentiate(X[0], X[1], 1.0, sqeuclidean, no_window, None)
b = softdtw_barycenter(X, max_iter=50)
plt.figure(figsize=(8,4))
for x in X:
    plt.plot(x, c='black', alpha=0.25)
plt.plot(a, c='red')
plt.plot(b, c='blue')
plt.show()

# X = np.vstack([X_train, X_test])
# y = np.hstack([y_train, y_test])

# X_train_ = [x[~np.isnan(x)] for x in X_train]
# X_train_ = [to_time_series(d, remove_nans=True) for d in X_train]
# X_test_ = [to_time_series(d, remove_nans=True) for d in X_test]
# X_ = np.array([X_train_, X_test_], dtype=object)

results = []
for y_label in np.unique(y):
    print(y_label)
    X_subset = X[y == y_label]
    medoid = get_medoid(X_subset)
    mean = nlaaf(X_subset)
    performance = irr(X_subset, mean, medoid)
    results.append([medoid, mean, performance])

[a[2] for a in results]
[len(a[1]) for a in results]

X_subset = X.copy()#[y == '1.0']
medoid = get_medoid(X_subset)

mean = nlaaf(X_subset)
print(irr(X_subset, mean, medoid))

fig, ax = plt.subplots(2, 1)
for x in X_subset:
    ax[0].plot(x, c="blue", alpha=0.1)
ax[1].plot(mean, c="red")

medoid = get_medoid(X_subset)
mean, std, medoid = cwrt(X_subset, medoid)
print(irr(X_subset, mean, medoid))

plt.show()



# STREAMING
