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

    def plot_cost_matrix(self, labels=True, ax=None, pax=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(5,5))
        im = ax.imshow(
            self.cost.T, origin="lower", cmap="viridis_r", vmin=0, aspect="auto", interpolation="none", alpha=0.9
        )

        n, m = len(self.x), len(self.y)
        if n<15 and m<15:
            for i in range(n):
                for j in range(m):
                    ax.text(i,j,int(self.cost[i,j]), ha="center", va="center")
        # co = ax.contour(self.cost.T, colors="grey", linewidths=1)
        # ax.clabel(co)
        if pax is None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            pax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im, cax=pax, orientation="vertical", label=r"Local Cost $C_{i,j}$")
            plt.colorbar(im, cax=pax, orientation="vertical", label=r"Accumulated Cost $D_{i,j}$")
        else:
            plt.colorbar(im, cax=pax)
        if self.compute_path:
            ax.plot(self.path[:, 0], self.path[:, 1], color="black", alpha=1.0, lw=2)
            # ax.plot(self.path[:, 0], self.path[:, 1], c="red", marker='o', markersize=1)
        if labels:
            ax.set_xlabel(r"$\mathbf{x}$ Index")
            ax.set_ylabel(r"$\mathbf{y}$ Index")
            # ax.set_title("Cost matrix")
        else:
            ax.set_axis_off()
            pass
        ax.axis("equal")

        return ax

    def plot_path(self, ax=None):
        if not self.compute_path:
            raise Exception("Alignment path not calculated.")
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(5,5))
        ax.plot(self.path[:, 0], self.path[:, 1])
        ax.set_title("Alignment Path")
        ax.set_xlabel(r"$\mathbf{x}$ Index")
        ax.set_ylabel(r"$\mathbf{y}$ Index")
        return ax

    def plot_pattern(self, labels=True, ax=None):
        return self.pattern.plot(labels, ax)

    def plot_ts_subplot(self, data, title):
        d = data.shape[1]
        fig, ax = plt.subplots(nrows=d, ncols=1, sharex=True, constrained_layout=True, figsize=(5,5))
        for i in range(d):
            ax[i].plot(data[:, i])
            ax[i].set_ylabel(f"dim.{i}")
        plt.xlabel("Time Index")
        fig.suptitle(title)

    def plot_ts_overlay(self, data, title, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True, figsize=(5,5))
        for i in range(data.shape[1]):
            ax.plot(data[:, i])
        ax.set_xlabel("Time Index")
        ax.set_title(title)

    def plot_ts_query(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True, figsize=(5,5))
        n, d = self.x.shape
        t = np.arange(n)
        for i in range(d):
            ax.plot(t, self.x[:, i], color="black", alpha=0.9)
        ax.set_xlabel(r"$\mathbf{x}$")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    def plot_ts_reference(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True, figsize=(5,5))
        n, d = self.y.shape
        t = np.arange(n)
        for i in range(d):
            ax.plot(self.y[:, i], t, color="black", alpha=0.9)
        ax.invert_xaxis()
        ax.set_ylabel(r"$\mathbf{y}$")
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
            fig, ax = plt.subplots(1, constrained_layout=True, figsize=(6,3))
        ax.plot(self.x + h, color="#EF476F", lw=2)
        ax.plot(self.y - h, color="#06D6A0", lw=2)
        ax.text(-1.2,6,r"$\mathbf{x}$",ha="right",va="center", color="#EF476F")
        ax.text(-1.2,-2.5,r"$\mathbf{y}$",ha="right",va="center", color="#06D6A0")
        for p in self.path:
            i, j = p
            ax.plot(p, [self.x[i] + h, self.y[j] - h], c="black", alpha=0.1)
        
        n, m = len(self.x), len(self.y)
        if n<15 and m<15:
            for i in range(n):
                ax.text(i, self.x[i]+h,int(self.x[i]),ha="center",va="bottom")
            for j in range(m):
                ax.text(j, self.y[j]-h,int(self.y[j]),ha="center",va="bottom")
        # ax.set_title("Warping Path")
        ax.axis("off")
