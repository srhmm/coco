import networkx as nx
from networkx.exception import NetworkXNoCycle
import pandas as pd
# import pystan
import numpy as np
import scipy as sp
import timeit
import re
import random

from model import Causal, Confounded
import config


class Network(object):
    def __init__(self, N=0, D=0, M=0, rel_weight=0, x=None, G=None, **kwargs):
        self.G = G
        self.x = x
        self.M = M
        assert (D > 0) or (self.G is not None)
        if self.G is not None:
            self.D = len(G.nodes)
        else:
            self.D = D

        assert (N > 0) or (self.x is not None)
        if self.x is not None:
            self.N = self.x.shape[1]
        else:
            self.N = N

        # If the data is confounded, we want the weight of the edges in between the observed variables to me much smaller
        # than from the hidden variable
        # reel_weight is the fraction of the size of w_zx to w_xx (e.g. w_zx have std rel_weight and w_xx have std 1)
        if self.M > 0:
            self.rel_weight = rel_weight
        else:
            self.rel_weight = 0

        # This can be either a scalar or a vector
        self.gamma = kwargs.get('gamma', 0)

    def gen_acyclic_net(self, size=None):
        if not size:
            size = self.D
        # self.G = nx.scale_free_graph(self.D)
        self.G = nx.directed.scale_free_graph(size)
        self.G = self.G.to_directed()
        self._G = nx.DiGraph()
        for u, v, _ in self.G.edges:
            if (u, v) not in self._G.edges:
                self._G.add_edge(u, v)
        self.G = self._G
        nodes = list(self.G.nodes)
        try:
            cycle = nx.algorithms.cycles.find_cycle(self.G)
            while cycle:
                u, v = cycle[-1]
                self.G.remove_edge(u, v)
                cycle = nx.algorithms.cycles.find_cycle(self.G)
        except NetworkXNoCycle:
            pass
        return self.G

    def generate_data(self):
        z = np.random.normal(0, 1, [self.N, self.M])
        w = self.rel_weight * np.random.normal(0, 1, [self.M, self.D])
        self.x = z.dot(w) + np.random.normal(0, 1, [self.N, self.D])
        # Each variable's parents values will already have been updated by the time we come to a node so this works
        G_topo = nx.topological_sort(self.G)

        for node in G_topo:
            pred = list(self.G.predecessors(node))
            i = node
            if pred:
                indices = pred
                xi_pred = self.x[:, indices]
                wi = np.random.normal(0, 1, len(indices))
                self.x[:, i] += xi_pred.dot(wi)
            self.G.nodes[node]['data'] = self.x[:, i]
        self.x = pd.DataFrame(self.x, columns=list(self.G.nodes))
        return self.x

    def generate_data_alpha(self, confounded_sets=None):
        self.x = np.random.normal(0, 1, [self.N, self.D])
        if confounded_sets:
            z = []
            # For each set of confounded variables, they get their own set 
            for cs in confounded_sets:
                cs = list(cs)
                _z = np.random.normal(0, 1, [self.N, self.M])
                z.append(_z)
                _w = np.random.normal(0, 1, [self.M, len(cs)])
                self.x[:, cs] += self.alpha[cs] * _z.dot(_w)
        else:
            # This works independent of whether alpha is scalar or a D-dimensional vector of alphas for each of the xi
            # If alpha = 10, then the weights from z -> x will be on average 10x as large as those from xi -> xj
            z = np.random.normal(0, 1, [self.N, self.M])
            w = np.random.normal(0, 1, [self.M, self.D])
            self.x = self.alpha * z.dot(w) + np.random.normal(0, 1, [self.N, self.D])


        G_topo = nx.topological_sort(self.G)

        for node in G_topo:
            pred = list(self.G.predecessors(node))
            i = node
            if pred:
                indices = pred
                xi_pred = self.x[:, indices]
                wi = np.random.normal(0, 1, len(indices))
                # If it is a vector, use the correct entry; if not just multiply with the scalar
                self.x[:, i] += xi_pred.dot(wi)
            self.G.nodes[node]['data'] = self.x[:, i]
        self.x = pd.DataFrame(self.x, columns=list(self.G.nodes))
        if not confounded_sets:
            z = pd.DataFrame(z)
        return self.x, z
