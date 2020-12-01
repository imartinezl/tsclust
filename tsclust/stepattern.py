#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 17 16:57 2020

@author: imartinez 
"""

"""
The :mod:`tslearn.stepattern` module delivers time-series specific metrics to be 
used at the core of machine learning algorithms.
**User guide:** See the :ref:`Dynamic Time Warping (DTW) <dtw>` section for 
further details.
"""

"""Code for Singular Spectrum Analysis."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import numba as nb

jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": False,
    "error_model": "numpy",
    "fastmath": True,
    "debug": True,
}


class Step:
    """
    Step pattern class: the documentation is being developed right now. I am going to extend this line so that I somehow a tool can split the line
    """

    def __init__(self, dx, dy, cost):
        self.dx = dx
        self.dy = dy
        self.cost = cost

    @property
    def dx(self) -> int:
        return self._dx

    @dx.setter
    def dx(self, dx: int) -> None:
        self._dx = dx

    @property
    def dy(self) -> int:
        return self._dy

    @dy.setter
    def dy(self, dy: int) -> None:
        self._dy = dy

    @property
    def cost(self) -> int:
        return self._cost

    @cost.setter
    def cost(self, cost: int) -> None:
        self._cost = cost

    def __str__(self):
        return f"Step dx:{self.dx} dy:{self.dy} cost:{self.cost}"

    def to_numpy(self):
        return np.array([self.dx, self.dy, self.cost])


import tabulate


class StepPattern:
    def __init__(self, name, steps=[], norm="NA"):
        self.name = name
        self.steps = steps
        self.norm = norm

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def steps(self) -> list:
        return self._steps

    @steps.setter
    def steps(self, steps: list) -> None:
        self._steps = steps

    @property
    def norm(self) -> str:
        return self._norm

    @norm.setter
    def norm(self, norm: str) -> None:
        self._norm = norm

    def add_step(self, step):
        self.steps.append(step)

    def del_step(self, step):
        self.steps.remove(step)

    def __str__(self):
        headers = ["Step", "dX", "dY", "Cost"]
        table = []
        for step in self.steps:
            table.append([step.dx, step.dy, step.cost])
        return (
            f"StepPattern: {self.name}\nNormalization: {self.norm}\n"
            + tabulate.tabulate(table, headers, tablefmt="github", showindex="always")
        )

    def to_numpy(self):
        return np.stack([step.to_numpy() for step in self.steps])


## Well-known step patterns
## White-Neely symmetric (default) aka Quasi-symmetric \cite{White1976}
symmetric1 = StepPattern("symmetric1", [Step(1, 1, 1), Step(0, 1, 1), Step(1, 0, 1)])
## Normal symmetric
symmetric2 = StepPattern(
    "symmetric2", [Step(1, 1, 2), Step(0, 1, 1), Step(1, 0, 1)], "N+M"
)
## classic asymmetric pattern: max slope 2, min slope 0
asymmetric = StepPattern(
    "asymmetric", [Step(1, 0, 1), Step(1, 1, 1), Step(1, 2, 1)], "N"
)

print(symmetric2)
print(asymmetric)

## Completely unflexible: fixed slope 1. Only makes sense with open.begin and open.end
rigid = StepPattern("rigid", [Step(1, 1, 1)], "N")
