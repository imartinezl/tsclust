#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import tabulate
import networkx as nx

jitkw = {
    "nopython": True,
    "nogil": True,
    "cache": False,
    "error_model": "numpy",
    "fastmath": True,
    "debug": True
}

def _num_to_str(num):
    if type(num) == int:
        return str(num)
    elif type(num) == float:
        return "{0:1.2f}".format(num)
    else:
        return str(num)

class BasePattern():
    def __init__(self):
        self.num_pattern = len(self.pattern) # number of patterns
        self.max_pattern_len = max([len(p["indices"]) for p in self.pattern]) # max length of pattern
        self._get_array()

    def _get_array(self):
        array = np.zeros([self.num_pattern, self.max_pattern_len, 3], dtype="float")
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            for j in range(pattern_len):
                array[i, j, 0:2] = self.pattern[i]["indices"][j]
                array[i, j, 2] = self.pattern[i]["weights"][j]
        self.array = array

    def _gen_graph(self):
        graph = nx.DiGraph()
        graph_layout = dict()
        edge_labels = dict()
        node_colors = []
        node_names = []
        # set node
        for i, pat in enumerate(self.pattern):
            step_len = len(pat["indices"])
            nn = []
            for j in range(step_len):
                node_name = str(i) + str(j)
                graph.add_node(node_name)
                graph_layout[node_name] = np.array(pat["indices"][j])
                if pat["weights"][j] == -1:
                    node_colors.append('r')
                else:
                    node_colors.append('b')
                nn.append(node_name)
            node_names.append(nn)
        # set edge
        for i, pat in enumerate(self.pattern):
            step_len = len(pat["indices"])
            for j in range(step_len-1):
                graph.add_edge(node_names[i][j], node_names[i][j+1])
                edge_labels[(node_names[i][j], node_names[i][j+1])] = str(pat["weights"][j+1])
        self._graph = graph
        self._graph_layout = graph_layout
        self._edge_labels = edge_labels
        self._node_colors = node_colors

    def __repr__(self):
        title = self.label + " pattern: \n\n"
        head = "g[i,j] = min(\n"

        body = ""
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            for j in range(pattern_len):
                dx, dy = self.pattern[i]["indices"][j]
                weight = self.pattern[i]["weights"][j]
                dx_str = "" if dx == 0 else f"{int(dx)}"
                dy_str = "" if dy == 0 else f"{int(dy)}"
                delta_str = f"i{dx_str:2},j{dy_str:2}"

                if weight == -1:
                    global_str = f"\tg[{delta_str}]"
                    body = body + " " + global_str
                else:
                    weight_str = f"{weight:2.2g} *"
                    local_str = f"+{weight_str} d[{delta_str}]"
                    body = body + " " + local_str
            body = body + " ,\n"

        tail = " ) \n\n"
        norm = f"normalization: {self.norm}\n"

        return title + head + body + tail + norm

    def table(self):
        headers = ['Pattern', 'dX', 'dY', 'Weight']
        table = []
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            for j in range(pattern_len):
                dx, dy = self.pattern[i]["indices"][j]
                weight = self.pattern[i]["weights"][j]
                table.append([i, dx, dy, weight])

        return f'{self.label} pattern: \n\n' + \
               tabulate.tabulate(table, headers, tablefmt="github") + \
               f'\n\nnormalization: {self.norm}\n'

    def plot(self):
        alpha = .5
        fudge = [0, 0]

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            px, py = zip(*self.pattern[i]["indices"])
            ax.plot(px, py, lw=2, color="tab:blue")
            ax.plot(px, py, color="black", marker="o", fillstyle="none")

            if pattern_len == 1:
                continue
            for j in range(pattern_len-1):
                weight = self.pattern[i]["weights"][j]
                if weight == -1:
                    ax.plot(px[j], py[j], 'o', color="black")
                else:
                    xh = alpha * px[j] + (1 - alpha) * px[j+1] + fudge[0]
                    yh = alpha * py[j] + (1 - alpha) * py[j+1] + fudge[1]
                    ax.annotate(str(weight), xy = (xh, yh), 
                                arrowprops=dict(arrowstyle="->", color="0.5",
                                                shrinkA=5, shrinkB=5,
                                                patchA=None, patchB=None,
                                                connectionstyle="arc3,rad=0.3",
                                                )
                                )


        ax.set_xlabel("Query index")
        ax.set_ylabel("Reference index")
        ax.set_xticks(np.unique(self.array[:, :, 0]))
        ax.set_yticks(np.unique(self.array[:, :, 1]))
        plt.show()
        return ax

    def plot_graph(self):
        """Show step pattern.
        """
        plt.figure(figsize=(6, 6))
        if not hasattr(self, "_graph"):
            self._gen_graph()
        nx.draw_networkx_nodes(self._graph, pos=self._graph_layout, node_color=self._node_colors, node_size=20, alpha=0.8)
        nx.draw_networkx_edges(self._graph, pos=self._graph_layout)
        nx.draw_networkx_edge_labels(self._graph,
                                     pos=self._graph_layout,
                                     edge_labels=self._edge_labels)
        min_x = np.min(self.array[:, :, 0])
        min_y = np.min(self.array[:, :, 1])
        plt.xlim([min_x - 0.5, 0.5])
        plt.ylim([min_y - 0.5, 0.5])
        plt.title(self.label + str(" pattern"))
        plt.xlabel("query index")
        plt.ylabel("reference index")
        plt.show()

    def __repr__depr(self):
        pattern = self.pattern
        s = self.label + " pattern: \n\n"
        for i in range(self.num_pattern):
            s += "pattern " + str(i) + ": "
            pattern_len = len(pattern[i]["indices"])
            p = str(pattern[i]["indices"][0])
            for j in range(1, pattern_len):
                p += " - ["
                p += str(pattern[i]["weights"][j])
                p += "] - "
                p += str(pattern[i]["indices"][j])
            s += p + "\n"
        s += "\nnormalization: " + self.norm
        return s

class Symmetric1(BasePattern):
    label = "symmetric1"
    pattern = [
        dict(
            indices=[(-1, 0), (0, 0)],
            weights=[-1, 1]
        ),
        dict(
            indices=[(-1, -1), (0, 0)],
            weights=[-1, 1]
        ),
        dict(
            indices=[(0, -1), (0, 0)],
            weights=[-1, 1]
        )
    ]
    norm = "none"

    def __init__(self):
        super().__init__()

class SymmetricP05(BasePattern):
    label = "symmetricP05"
    pattern = [
        dict(
            indices=[(-1, -3), (0, -2), (0, -1), (0, 0)],
            weights=[-1, 2, 1, 1]
        ),
        dict(
            indices=[(-1, -2), (0, -1), (0, 0)],
            weights=[-1, 2, 1]
        ),
        dict(
            indices=[(-1, -1), (0, 0)],
            weights=[-1, 2]
        ),
        dict(
            indices=[(-2, -1), (-1, 0), (0, 0)],
            weights=[-1, 2, 1]
        ),
        dict(
            indices=[(-3, -1), (-2, 0), (-1, 0), (0, 0)],
            weights=[-1, 2, 1, 1]
        )
    ]
    norm = "N+M"

    def __init__(self):
        super().__init__()

a = SymmetricP05()
print(a)
print(a.array)
a.plot()
print(a.table())
a.plot_graph()



class Step:
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
        return f'Step dx:{self.dx} dy:{self.dy} cost:{self.cost}'

    def to_numpy(self):
        return np.array([self.dx, self.dy, self.cost])

import tabulate
class StepPattern:
    def __init__(self, name, steps=[], norm='NA'):
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
        headers = ['Step', 'dX', 'dY', 'Cost']
        table = []
        for step in self.steps:
            table.append([step.dx, step.dy, step.cost])
        return  f'StepPattern: {self.name}\nNormalization: {self.norm}\n' + \
                tabulate.tabulate(table, headers, tablefmt="github", showindex="always")

    def to_numpy(self):
        return np.stack([step.to_numpy() for step in self.steps])

# ## Well-known step patterns
# ## White-Neely symmetric (default) aka Quasi-symmetric \cite{White1976}
# symmetric1 = StepPattern('symmetric1', [Step(1, 1, 1), Step(0, 1, 1), Step(1, 0, 1)])
# ## Normal symmetric
# symmetric2 = StepPattern('symmetric2', [Step(1, 1, 2), Step(0, 1, 1), Step(1, 0, 1)], 'N+M')
# ## classic asymmetric pattern: max slope 2, min slope 0
# asymmetric = StepPattern('asymmetric', [Step(1, 0, 1), Step(1, 1, 1), Step(1, 2, 1)], 'N')
#
# print(symmetric2)
# print(asymmetric)
#
# ## Completely unflexible: fixed slope 1. Only makes sense with open.begin and open.end
# rigid = StepPattern('rigid', [Step(1,1,1)], 'N')