#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class DtwResult:
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

    def plot_cost_matrix(self, labels=True, ax=None, pax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        im = ax.imshow(
            self.cost.T, origin="lower", cmap="inferno", vmin=0, aspect="auto"
        )
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
        return self.pattern.plot(labels, ax)

    def plot_ts_subplot(self, data, title):
        d = data.shape[1]
        fig, ax = plt.subplots(nrows=d, ncols=1, sharex=True, constrained_layout=True)
        for i in range(d):
            ax[i].plot(data[:, i])
            ax[i].set_ylabel(f"dim.{i}")
        plt.xlabel("Time Index")
        fig.suptitle(title)

    def plot_ts_overlay(self, data, title, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)
        for i in range(data.shape[1]):
            ax.plot(data[:, i])
        ax.set_xlabel("Time Index")
        ax.set_title(title)

    def plot_ts_query(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)
        n, d = self.x.shape
        t = np.arange(n)
        for i in range(d):
            ax.plot(t, self.x[:, i], color="black", alpha=0.9)
        ax.set_xlabel("Query X")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    def plot_ts_reference(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)
        n, d = self.y.shape
        t = np.arange(n)
        for i in range(d):
            ax.plot(self.y[:, i], t, color="black", alpha=0.9)
        ax.invert_xaxis()
        ax.set_ylabel("Reference Y")
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", which="both", right=False, labelright=False)

    def plot_summary(self):
        n = self.x.shape[0]
        m = self.y.shape[0]
        h = min(n, m) / 2.5
        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        spec = fig.add_gridspec(
            ncols=3, nrows=2, width_ratios=[h, n, n * 0.05], height_ratios=[m, h]
        )
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

    def plot_warp(self, ax=None):
        h = (np.max(self.y) - np.min(self.x)) * 1
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)
        ax.plot(self.x + h, color="black")
        ax.plot(self.y - h, color="black")
        for p in self.path:
            i, j = p
            ax.plot(p, [self.x[i] + h, self.y[j] - h], c="#000000", alpha=0.1)
        ax.set_title("Warping Path")
