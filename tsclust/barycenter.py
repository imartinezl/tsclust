#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tsclust.dtw_classic import dtw
from metrics import euclidean
from window import no_window

import matplotlib.pyplot as plt
# BATCH

# NonLinear Alignment and Averaging Filters (NLAAF): Tournament scheme
def nlaaf(D, W=None):
    if W is None:
        D = D.copy()
        np.random.shuffle(D)
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
        cost, path, dist = dtw(x, y, euclidean, no_window)
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
def get_medoid(X):
    medoid = None
    min_dist = np.inf
    for x in X:
        sum_dist = 0
        for y in X:
            cost, path, dist = dtw(x, y, euclidean, no_window)
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
        cost, path, dist = dtw(medoid, x, euclidean, no_window)
        for p, q in path:
            medoid_assignments[p].append(x[q])

    mean = np.array([np.mean(x) for x in medoid_assignments])
    std = np.array([np.std(x) for x in medoid_assignments])
    return mean, std, medoid


def inertia(X, mean):
    total = 0.0
    for x in X:
        cost, path, dist = dtw(mean, x, euclidean, no_window)
        total += dist
    return total

def irr(X, centroid, medoid):
    total_centroid = 0.0
    total_medoid = 0.0
    for x in X:
        cost, path, dist = dtw(centroid, x, euclidean, no_window)
        total_centroid += dist
        cost, path, dist = dtw(medoid, x, euclidean, no_window)
        total_medoid += dist
    return 1 - (total_centroid / total_medoid)





from tsclust.datasets import UCR_UEA_datasets

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("Trace")
X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])
X_subset = X[y == '3.0']

mean = nlaaf(X_subset)

fig, ax = plt.subplots(2, 1)
for x in X_subset:
    ax[0].plot(x, c="blue", alpha=0.1)
ax[1].plot(mean, c="red")

medoid = get_medoid(X_subset)
mean, std, medoid = cwrt(X_subset, medoid)
print(irr(X_subset, mean, medoid))

plt.show()



# STREAMING
