#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# BATCH

# NonLinear Alignment and Averaging Filters (NLAAF): Tournament scheme
def nlaaf(D, W):
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
        dist, cost, path = dtw(x, y)
        warp = [(x[p], y[q]) for p, q in path]
        mean = np.average(warp, axis=1, weights=[w1, w2])
        # std = np.std(warp, axis=1)
        D.append(mean)
        W.append(w1 + w2)

    if len(D) == 1:
        return D[-1]
    else:
        return nlaaf(D, W)

import numpy as np
from tsclust.dtw import dtw
from tsclust.datasets import UCR_UEA_datasets

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('Trace')
X = np.vstack([X_train, X_test])

D = X.copy()
np.random.shuffle(D)
W = [1] * len(D)
mean = nlaaf(D, W)
plt.figure()
for x in X:
    plt.plot(x, c="blue", alpha=0.1)
plt.plot(mean, c="red")

# STREAMING