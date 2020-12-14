#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains code that was borrowed from dtwalign.

https://github.com/statefb/dtwalign
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# import numba as nb
# import tabulate

# jitkw = {
#     "nopython": True,
#     "nogil": True,
#     "cache": False,
#     "error_model": "numpy",
#     "fastmath": True,
#     "debug": True,
# }

all = {
    "symmetric1",
    "symmetric2",
    "asymmetric",
    "symmetricP0",
    "asymmetricP0",
    "symmetricP05",
    "asymmetricP05",
    "symmetricP1",
    "asymmetricP1",
    "symmetricP2",
    "asymmetricP2",
    "typeIa",
    "typeIb",
    "typeIc",
    "typeId",
    "typeIas",
    "typeIbs",
    "typeIcs",
    "typeIds",
    "typeIIa",
    "typeIIb",
    "typeIIc",
    "typeIId",
    "typeIIIc",
    "typeIVc",
    "mori2006",
    "unitary",
}

def _num_to_str(num):
    if type(num) == int:
        return str(num)
    elif type(num) == float:
        return "{0:1.2f}".format(num)
    else:
        return str(num)



class BasePattern:
    label = ""
    pattern = []
    normalize = []

    def __init__(self):
        self.num_pattern = len(self.pattern)  # number of patterns
        self.max_pattern_len = max(
            [len(p["indices"]) for p in self.pattern]
        )  # max length of pattern
        self._get_array()

    def _get_array(self):
        array = np.zeros([self.num_pattern, self.max_pattern_len, 3], dtype="float")
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            weight_len = len(self.pattern[i]["weights"])
            if weight_len == pattern_len-1:
                self.pattern[i]["weights"].insert(0, -1)
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
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            nn = []
            for j in range(pattern_len):
                node_name = str(i) + str(j)
                graph.add_node(node_name)
                graph_layout[node_name] = np.array(self.pattern[i]["indices"][j])
                if self.pattern[i]["weights"][j] == -1:
                    node_colors.append("r")
                else:
                    node_colors.append("b")
                nn.append(node_name)
            node_names.append(nn)
        # set edge
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            for j in range(pattern_len - 1):
                graph.add_edge(node_names[i][j], node_names[i][j + 1])
                edge_labels[(node_names[i][j], node_names[i][j + 1])] = \
                    _num_to_str(self.pattern[i]["weights"][j + 1])
        self._graph = graph
        self._graph_layout = graph_layout
        self._edge_labels = edge_labels
        self._node_colors = node_colors

    @property
    def is_normalizable(self):
        return self.normalize != "none"

    def do_normalize(self, value, n, m):
        """Normalize
        row : 1D array
            expect last row of D
        n : int
            length of query (D.shape[0])
        m : int
            length of reference (D.shape[1])
        """
        if not self.is_normalizable:
            return None
        if self.normalize == "N+M":
            return value / (n + m + 1)
        elif self.normalize == "N":
            return value / n
        elif self.normalize == "M":
            return value / (m + 1)
        else:
            raise Exception()

    def __repr__(self):
        return self.repr_formula()

    def repr_formula(self):
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
                    local_str = f"+{weight:2.2g} * d[{delta_str}]"
                    body = body + " " + local_str
            body = body + " ,\n"

        tail = " ) \n\n"
        normalize = f"normalization: {self.normalize}\n"

        return title + head + body + tail + normalize

    def repr_graph(self):
        s = self.label + " pattern: \n\n"
        for i in range(self.num_pattern):
            s += "pattern " + str(i) + ": "
            pattern_len = len(self.pattern[i]["indices"])
            p = str(self.pattern[i]["indices"][0])
            for j in range(1, pattern_len):
                p += " - ["
                p += _num_to_str(self.pattern[i]["weights"][j])
                p += "] - "
                p += str(self.pattern[i]["indices"][j])
            s += p + "\n"
        s += "\nnormalization: " + str(self.normalize)
        return s

    # def table_tabulate(self):
    #     headers = ['Pattern', 'dX', 'dY', 'Weight']
    #     table = []
    #     for i in range(self.num_pattern):
    #         pattern_len = len(self.pattern[i]["indices"])
    #         for j in range(pattern_len):
    #             dx, dy = self.pattern[i]["indices"][j]
    #             weight = self.pattern[i]["weights"][j]
    #             table.append([i, dx, dy, weight])
    #
    #     title = f"{self.label} pattern: \n\n"
    #     normalize = f"\n\nnormalization: {self.normalize}\n"
    #     return title + tabulate.tabulate(table, headers, tablefmt="github") + normalize

    def table(self):
        title = f"{self.label} pattern: \n\n"
        headers = ["Pattern", "dX", "dY", "Weight"]
        width = [len(h) + 2 for h in headers]

        table = "|"
        for k in range(len(headers)):
            table += headers[k].center(width[k] + 2) + "|"
        table += "\n|"
        for k in range(len(headers)):
            table += "-" * (width[k] + 2) + "|"

        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            for j in range(pattern_len):
                dx, dy = self.pattern[i]["indices"][j]
                weight = self.pattern[i]["weights"][j]
                data = [i, dx, dy, weight]
                table += "\n|"
                for k in range(len(headers)):
                    table += _num_to_str(data[k]).rjust(width[k] + 1) + " |"

        normalize = f"\n\nnormalization: {self.normalize}\n"
        return title + table + normalize

    def plot_matplotlib(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(self.num_pattern):
            pattern_len = len(self.pattern[i]["indices"])
            if pattern_len == 1:
                continue

            px, py = zip(*self.pattern[i]["indices"])
            for j in range(pattern_len):
                weight = self.pattern[i]["weights"][j]
                if weight == -1:
                    ax.plot(px[j], py[j], "o", color="red")
                else:
                    alpha = 0.55
                    fudge = (
                        0.05 * abs(py[j] - py[j - 1]),
                        -0.12 * abs(px[j] - px[j - 1]),
                    )
                    xh = alpha * px[j - 1] + (1 - alpha) * px[j] + fudge[0]
                    yh = alpha * py[j - 1] + (1 - alpha) * py[j] + fudge[1]
                    ax.annotate(_num_to_str(weight), (xh, yh))
                    ax.plot(px[j], py[j], color="blue", marker="o", fillstyle="none")
                    ax.annotate(
                        "",
                        xy=(px[j], py[j]),
                        xytext=(px[j - 1], py[j - 1]),
                        arrowprops=dict(
                            arrowstyle="->", linewidth=1, shrinkA=10, shrinkB=10
                        ),
                    )
        x_ticks = np.unique(self.array[:, :, 0])
        y_ticks = np.unique(self.array[:, :, 1])
        ax.set_xlim([np.min(x_ticks) - 0.5, 0.5])
        ax.set_ylim([np.min(y_ticks) - 0.5, 0.5])
        ax.set_title(self.label + " pattern")
        ax.set_xlabel("Query index")
        ax.set_ylabel("Reference index")
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        plt.show()
        return ax

    def plot_graph(self):
        """Show step pattern."""
        fig, ax = plt.subplots(figsize=(6, 6))
        if not hasattr(self, "_graph"):
            self._gen_graph()
        nx.draw_networkx_nodes(
            self._graph,
            pos=self._graph_layout,
            node_color=self._node_colors,
            node_size=25,
            alpha=0.8,
        )
        nx.draw_networkx_edges(self._graph, pos=self._graph_layout)
        nx.draw_networkx_edge_labels(
            self._graph, pos=self._graph_layout, edge_labels=self._edge_labels
        )
        x_ticks = np.unique(self.array[:, :, 0])
        y_ticks = np.unique(self.array[:, :, 1])
        ax.set_xlim([np.min(x_ticks) - 0.5, 0.5])
        ax.set_ylim([np.min(y_ticks) - 0.5, 0.5])
        plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_title(self.label + " pattern")
        ax.set_xlabel("Query index")
        ax.set_ylabel("Reference index")
        plt.show()

        return ax

    def plot(self):
        return self.plot_graph()


class Symmetric1(BasePattern):
    label = "symmetric1"
    pattern = [
        dict(indices=[(-1, 0), (0, 0)], weights=[1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(0, -1), (0, 0)], weights=[1]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class Symmetric2(BasePattern):
    label = "symmetric2"
    pattern = [
        dict(indices=[(-1, 0), (0, 0)], weights=[1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[2]),
        dict(indices=[(0, -1), (0, 0)], weights=[1]),
    ]
    normalize = "N+M"

    def __init__(self):
        super().__init__()


class SymmetricP0(Symmetric2):
    """Same as symmetric2 pattern."""

    label = "symmetricP05"


class SymmetricP05(BasePattern):
    label = "symmetricP05"
    pattern = [
        dict(indices=[(-1, -3), (0, -2), (0, -1), (0, 0)], weights=[2, 1, 1]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[2, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[2]),
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[2, 1]),
        dict(indices=[(-3, -1), (-2, 0), (-1, 0), (0, 0)], weights=[2, 1, 1]),
    ]
    normalize = "N+M"

    def __init__(self):
        super().__init__()


class SymmetricP1(BasePattern):
    label = "symmetricP1"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[2, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[2]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[2, 1]),
    ]
    normalize = "N+M"

    def __init__(self):
        super().__init__()


class SymmetricP2(BasePattern):
    label = "symmetricP2"
    pattern = [
        dict(indices=[(-3, -2), (-2, -1), (-1, 0), (0, 0)], weights=[2, 2, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[2]),
        dict(indices=[(-2, -3), (-1, -2), (0, -1), (0, 0)], weights=[2, 2, 1]),
    ]
    normalize = "N+M"

    def __init__(self):
        super().__init__()


class Asymmetric(BasePattern):
    label = "asymmetric"
    pattern = [
        dict(indices=[(-1, 0), (0, 0)], weights=[1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, 0)], weights=[1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class AsymmetricP0(BasePattern):
    label = "asymmetricP0"
    pattern = [
        dict(indices=[(0, -1), (0, 0)], weights=[0]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, 0), (0, 0)], weights=[1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class AsymmetricP05(BasePattern):
    label = "asymmetricP05"
    pattern = [
        dict(
            indices=[(-1, -3), (0, -2), (0, -1), (0, 0)],
            weights=[1 / 3, 1 / 3, 1 / 3],
        ),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[0.5, 0.5]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-3, -1), (-2, 0), (-1, 0), (0, 0)], weights=[1, 1, 1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class AsymmetricP1(BasePattern):
    label = "asymmetricP1"
    pattern = [
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[0.5, 0.5]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class AsymmetricP2(BasePattern):
    label = "asymmetricP2"
    pattern = [
        dict(
            indices=[(-2, -3), (-1, -2), (0, -1), (0, 0)],
            weights=[2 / 3, 2 / 3, 2 / 3],
        ),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-3, -2), (-2, -1), (-1, 0), (0, 0)], weights=[1, 1, 1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class TypeIa(BasePattern):
    label = "typeIa"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 0]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[1, 0]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class TypeIb(BasePattern):
    label = "typeIb"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[1, 1]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class TypeIc(BasePattern):
    label = "typeIc"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[1, 0]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class TypeId(BasePattern):
    label = "typeId"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[2, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[2]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[2, 1]),
    ]
    normalize = "N+M"

    def __init__(self):
        super().__init__()


class TypeIas(BasePattern):
    label = "typeIas"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[0.5, 0.5]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[0.5, 0.5]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class TypeIbs(BasePattern):
    label = "typeIbs"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[1, 1]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class TypeIcs(BasePattern):
    label = "typeIcs"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[0.5, 0.5]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class TypeIds(BasePattern):
    label = "typeIds"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1.5, 1.5]),
        dict(indices=[(-1, -1), (0, 0)], weights=[2]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[1.5, 1.5]),
    ]
    normalize = "N+M"

    def __init__(self):
        super().__init__()


class TypeIIa(BasePattern):
    label = "typeIIa"
    pattern = [
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, 0)], weights=[1]),
        dict(indices=[(-2, -1), (0, 0)], weights=[1]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class TypeIIb(BasePattern):
    label = "typeIIb"
    pattern = [
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, 0)], weights=[2]),
        dict(indices=[(-2, -1), (0, 0)], weights=[2]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class TypeIIc(BasePattern):
    label = "typeIIc"
    pattern = [
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, 0)], weights=[1]),
        dict(indices=[(-2, -1), (0, 0)], weights=[2]),
    ]
    normalize = "none"

    def __init__(self):
        super().__init__()


class TypeIId(BasePattern):
    label = "typeIId"
    pattern = [
        dict(indices=[(-1, -1), (0, 0)], weights=[2]),
        dict(indices=[(-1, -2), (0, 0)], weights=[3]),
        dict(indices=[(-2, -1), (0, 0)], weights=[3]),
    ]
    normalize = "N+M"

    def __init__(self):
        super().__init__()


class TypeIIIc(BasePattern):
    label = "typeIIIc"
    pattern = [
        dict(indices=[(-1, -2), (0, 0)], weights=[1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-2, -2), (-1, 0), (0, 0)], weights=[1, 1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class TypeIVc(BasePattern):
    label = "typeIVc"
    pattern = [
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
        dict(indices=[(-1, -2), (0, 0)], weights=[1]),
        dict(indices=[(-1, -3), (0, 0)], weights=[1]),
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-2, -2), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-2, -3), (-1, 0), (0, 0)], weights=[1, 1]),
        dict(indices=[(-3, -1), (-2, 0), (-1, 0), (0, 0)], weights=[1, 1, 1]),
        dict(indices=[(-3, -2), (-2, 0), (-1, 0), (0, 0)], weights=[1, 1, 1]),
        dict(indices=[(-3, -3), (-2, 0), (-1, 0), (0, 0)], weights=[1, 1, 1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class Mori2006(BasePattern):
    label = "mori2006"
    pattern = [
        dict(indices=[(-2, -1), (-1, 0), (0, 0)], weights=[2, 1]),
        dict(indices=[(-1, -1), (0, 0)], weights=[3]),
        dict(indices=[(-1, -2), (0, -1), (0, 0)], weights=[3, 3]),
    ]
    normalize = "M"

    def __init__(self):
        super().__init__()


class Unitary(BasePattern):
    label = "unitary"
    pattern = [
        dict(indices=[(-1, -1), (0, 0)], weights=[1]),
    ]
    normalize = "N"

    def __init__(self):
        super().__init__()


class UserStepPattern(BasePattern):
    label = "user defined step pattern"

    def __init__(self, pattern, normalize):
        """User defined step pattern.

        Parameters
        ----------
        pattern : list
            list contains pattern information.
            ex) the case of symmetric2 pattern:

                .. code::

                    pattern = [
                        dict(
                            indices=[(-1,0),(0,0)],
                            weights=[1]
                        ),
                        dict(
                            indices=[(-1,-1),(0,0)],
                            weights=[2]
                        ),
                        dict(
                            indices=[(0,-1),(0,0)],
                            weights=[1]
                        )
                    ]

        normalize : string ('N','M','N+M','none')
            Guide to compute normalized distance.

        """
        # validation
        if normalize not in ("N", "M", "N+M", "none"):
            raise ValueError(
                "normalize argument must be \
                one of followings: 'N','M','N+M','none'"
            )

        self.pattern = pattern
        self.normalize = normalize
        # number of patterns
        self.num_pattern = len(self.pattern)
        # max length of pattern
        self.max_pattern_len = max([len(pi["indices"]) for pi in self.pattern])
        self._get_array()


def get_pattern(pattern_str):
    if pattern_str == "symmetric1":
        return Symmetric1()
    elif pattern_str == "symmetric2":
        return Symmetric2()
    elif pattern_str == "symmetricP05":
        return SymmetricP05()
    elif pattern_str == "symmetricP0":
        return SymmetricP0()
    elif pattern_str == "symmetricP1":
        return SymmetricP1()
    elif pattern_str == "symmetricP2":
        return SymmetricP2()
    elif pattern_str == "asymmetric":
        return Asymmetric()
    elif pattern_str == "asymmetricP0":
        return AsymmetricP0()
    elif pattern_str == "asymmetricP05":
        return AsymmetricP05()
    elif pattern_str == "asymmetricP1":
        return AsymmetricP1()
    elif pattern_str == "asymmetricP2":
        return AsymmetricP2()
    elif pattern_str == "typeIa":
        return TypeIa()
    elif pattern_str == "typeIb":
        return TypeIb()
    elif pattern_str == "typeIc":
        return TypeIc()
    elif pattern_str == "typeId":
        return TypeId()
    elif pattern_str == "typeIas":
        return TypeIas()
    elif pattern_str == "typeIbs":
        return TypeIbs()
    elif pattern_str == "typeIcs":
        return TypeIcs()
    elif pattern_str == "typeIds":
        return TypeIds()
    elif pattern_str == "typeIIa":
        return TypeIIa()
    elif pattern_str == "typeIIb":
        return TypeIIb()
    elif pattern_str == "typeIIc":
        return TypeIIc()
    elif pattern_str == "typeIId":
        return TypeIId()
    elif pattern_str == "typeIIIc":
        return TypeIIIc()
    elif pattern_str == "typeIVc":
        return TypeIVc()
    elif pattern_str == "mori2006":
        return Mori2006()
    elif pattern_str == "unitary":
        return Unitary()
    else:
        raise NotImplementedError("given step pattern not supported")
