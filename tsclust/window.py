#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numba as nb
import matplotlib.pyplot as plt

jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": False,
    "error_model": "numpy",
    "fastmath": True,
    "debug": True,
}


@nb.jit(**jitkw)
def no_window(i, j, query_size, reference_size, window_size):
    return True


@nb.jit(**jitkw)
def sakoe_chiba_window(i, j, query_size, reference_size, window_size):
    ok = abs(j - i) <= window_size
    return ok


@nb.jit(**jitkw)
def itakura_window(i, j, query_size, reference_size, window_size):
    n = query_size
    m = reference_size
    ok = (
        (j < 2 * i)
        and (i <= 2 * j)
        and (i >= n - 1 - 2 * (m - j))
        and (j > m - 1 - 2 * (n - i))
    )
    return ok


@nb.jit(**jitkw)
def slanted_band_window(i, j, query_size, reference_size, window_size):
    diag = i * reference_size / query_size
    return abs(j - diag) <= window_size


@nb.jit(**jitkw)
def precompute(window, query_size, reference_size, window_size=None):
    mask = np.empty((query_size, reference_size))
    for i in range(query_size):
        for j in range(reference_size):
            mask[i, j] = window(i, j, query_size, reference_size, window_size)
    return mask

@nb.jit(**jitkw)
def get_position(window, query_size, reference_size, window_size=None):
    mask = precompute(window, query_size, reference_size, window_size)
    return np.argwhere(mask)


def display(mask):
    query_size, reference_size = mask.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.pcolormesh(mask, edgecolors='#BC3908', linewidth=0.1, cmap='gray', snap=True)
    ax.imshow(mask, origin="upper", cmap="cividis")
    ax.set_ylabel(f"query (size: {query_size})")
    ax.set_xlabel(f"reference (size: {reference_size})")
    # ax.grid(color='r', linestyle='-', linewidth=2)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.axis("equal")
    plt.show()