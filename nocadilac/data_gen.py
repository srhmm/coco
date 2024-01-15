import scipy as sp
import networkx as nx
import math
import random
import numpy as np

class DataGen(object):
    def __init__(self, N, M, seed=0, frac_confounded=0, **kwargs):
        self.N = N
        self.M = M
        self.noise = kwargs.get('noise', 'normal')
        self.num_confounded = math.ceil(frac_confounded*self.M)
        self.alpha = kwargs.get('alpha', 1)
        self.Mz = kwargs.get('Mz', 1)
        self.gen_dict = {'lin': lambda x: x, 'quad': lambda x: x**2, 'cub': lambda x: x**3, 'exp': np.exp, 'log': sp.special.expit, 'sin': np.sin}
        self.gen = kwargs.get('gen', 'linear')

    def net(self):
        G = nx.directed.scale_free_graph(self.M)
        G = G.to_directed()
        _G = nx.DiGraph()
        for u, v, _ in G.edges:
            if (u, v) not in _G.edges:
                _G.add_edge(u, v)
        G = _G
        try:
            while True:
                cycle = nx.find_cycle(G)
                e = cycle.pop()
                G.remove_edge(*e)
        except nx.NetworkXNoCycle:
            pass
        self.G = G
        A = self.graph_to_adj()
        mask = np.random.uniform(0.1, 1, [self.M, self.M])
        W = A * mask
        self.W = W

    def graph_to_adj(self):
        A = np.zeros([self.M, self.M])
        for u, v in self.G.edges:
            A[u, v] = 1
        return A

    def data_init(self):
        if self.noise == 'normal':
            X = np.random.normal(0, 1, [self.N, self.M])
        elif self.noise == 'exp':
            X = np.random.exponential(1, [self.N, self.M])
        elif self.noise == 'gumbel':
            X = numpy.random.gumbel(0, 1, [self.N, self.M])

        return X

    def data(self):
        # Random parameters for nonlinear data generation
        u = np.random.uniform(0, 1, self.M)
        w = np.random.uniform(0, 1, self.M)
        s = self.gen_dict[self.gen]

        X = self.data_init()
        if self.num_confounded:
            self.conf_ind = sorted(random.sample(range(self.M), self.num_confounded))
            Z = np.random.normal(0, 1, [self.N, self.Mz])
            B = self.alpha*np.random.uniform(0, 1, [self.Mz, len(self.conf_ind)])
            X[:, self.conf_ind] += Z @ B
        else:
            self.conf_ind = []

        G_sort = nx.topological_sort(self.G)
        for i in G_sort:
            par = list(self.G.predecessors(i))
            X[:, i] += s(w[i]*X[:, par].dot(self.W[par, i])) + u[i]

        return X
