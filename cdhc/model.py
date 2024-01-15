# import pystan
# import numpy as np
# import scipy as sp
import logging

import numpy as np
import scipy as sp
import pymc3 as pm
import networkx as nx
from sklearn.preprocessing import StandardScaler
import theano.tensor as tt
import theano

import config

# TODO: how do you do this in a way that is more modular and will allow for network inference?

class Model(object):
    """Base class on top of which every concrete model is built."""
    def __init__(self, **kwargs):
        self.n_sample = kwargs.get('n_sample', 1000)
        self.inference = kwargs.get('inference', 'advi').lower()
        self.progressbar = kwargs.get('progressbar', True)
        self.logger = logging.getLogger("pymc3")
        if not kwargs.get('verbose', True):
            self.logger.setLevel(logging.ERROR)



    def add_data(self, X, Y=None):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.DX = X.shape[1]

    def create_model(self):
        raise NotImplementedError

    def fit_model(self, n=30000):
        """Based on a PyMC3 model, use ADVI to infer the posterior given the data."""
        with self.model:
            if self.inference == 'nuts':
                inference = pm.NUTS()
                self.trace = pm.sample(draws=self.n_sample, step=inference, random_seed=1)
            elif self.inference == 'advi':
                inference = pm.ADVI()
                approx = pm.fit(n=n, method=inference, progressbar=self.progressbar, random_seed=1)
                self.trace = approx.sample(draws=self.n_sample)
            elif self.inference == 'map':
                self.trace = pm.find_MAP()

    def eval_model(self):
        """Computes the log probabilities for each sample of the parameters from the posterior."""
        model = self.model
        trace = self.trace

        data_dict = {'X': self.X, 'Y': self.Y}

        variable_names = list(map(str, model.vars))

        logxp = model.X.logp
        if self.Y:
            logyp = model.Y.logp
        else:
            logyp = lambda point: 0
        if self.inference == 'map':
            logps = np.array([logxp(trace) + logyp(trace)])
        else:
            logps = np.zeros(self.n_sample)
            for i, _ in enumerate(trace):
                point = trace.point(i)
                point.update(data_dict)
                logps[i] = logxp(point) + logyp(point)

        return logps

    def eval_mean(self):
        logps = self.eval_model()
        mean = sp.special.logsumexp(logps) - np.log(logps.shape[0])
        return mean


class Causal(Model):
    """Model X as multivariate normally distributed and Y via a probabilistic linear regression from X to Y."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_data(self, X, Y=None):
        # If the target data is not specified, assume that it is in the last column of X.
        if Y:
            self.X = X
            self.Y = Y
        else:
            self.X = X.iloc[:, :-1]
            self.Y = X.iloc[:, -1]
        self.N = self.X.shape[0]
        self.DX = self.X.shape[1]

    def create_model(self):
        self.model = pm.Model()
        self.model_x()
        self.model_w()
        self.model_y()

    def model_x(self):
        """X is normally distributed."""
        with self.model:
            mux = pm.Normal('mux', 0, 1)
            sdx = pm.Lognormal('sdx', 0, 1)
            X = pm.Normal('X', mu=mux, sd=sdx, observed=self.X)

    def model_w(self):
        """Weights from X to Y are normally distributed."""
        with self.model:
            sdw = pm.Lognormal('sdw', 1)
            w = pm.Normal('w', 0, sdw, shape=[self.DX, 1])

    def model_y(self):
        """Model Y via regression onto X."""
        with self.model:
            muy = pm.Deterministic('muy',
                                   pm.math.dot(self.model.X, self.model.w))
            sdy = pm.Lognormal('sdy', 1)
            Y = pm.Normal('Y', mu=muy, sd=sdy, observed=self.Y)


class Confounded(Model):
    """Model (X, Y) by doing PPCA, i.e. jointly inferring a latent variable Z and doing linear regression from Z towards X,Y."""
    def __init__(self, DZ, **kwargs):
        self.DZ = DZ

        super().__init__(**kwargs)

    def create_model(self):
        self.model = pm.Model()
        self.model_z()
        self.model_w()
        self.model_x()
        if self.Y:
            self.model_y()

    def model_z(self):
        """Z is normally distributed."""
        with self.model:
            muz = pm.Normal('muz', 0, 10)
            sdz = pm.Lognormal('sdz', 0, 1)
            Z = pm.Normal('Z', mu=muz, sd=sdz, shape=[self.N, self.DZ])

    def model_w(self):
        """The weights from Z to (X,Y) are normally distributed."""
        with self.model:
            sdw = pm.Lognormal('sdw', 1)
            wx = pm.Normal('wx', 0, sdw, shape=[self.DZ, self.DX])
            wy = pm.Normal('wy', 0, sdw, shape=[self.DZ, 1])

    def model_x(self):
        """Model X via regression onto Z."""
        with self.model:
            mux = pm.Deterministic('mux',
                                   pm.math.dot(self.model.Z, self.model.wx))
            sdx = pm.Lognormal('sdx', 0, 1)
            X = pm.Normal('X', mu=mux, sd=sdx, observed=self.X)

    def model_y(self):
        """Model X via regression onto Z."""
        with self.model:
            muy = pm.Deterministic('muy',
                                   pm.math.dot(self.model.Z, self.model.wy))
            sdy = pm.Lognormal('sdy', 0, 1)
            Y = pm.Normal('Y', mu=muy, sd=sdy, observed=self.Y)


class GraphModel(Model):
    def __init__(self, G, **kwargs):
        super().__init__(**kwargs)
        self.G = G
        for node in self.G.nodes:
            self.G.nodes[node]['model'] = None
        try:
            assert not nx.algorithms.cycles.find_cycle(G), "Graph has a cycle!"
        except nx.exception.NetworkXNoCycle:
            pass

    def create_model(self):
        self.model = pm.Model()
        self.model_nodes()

    def model_nodes(self):
        self.model_params()
        self.model_leaves()
        self.model_interior()

    def model_params(self):
        with self.model:
            sdw = pm.Lognormal('sdw', 1)
            sdnode = pm.Lognormal('sdnode', 0, 1)

    def model_leaves(self):
        leaves = [node for node in self.G.nodes if not list(self.G.predecessors(node))]
        with self.model:
            for i, node in enumerate(leaves):
                self.G.nodes[node]['model'] = pm.normal(f'{node}', 0, self.model.sdnode,
                                                       observed = self.G.nodes[node]['data'])

    def model_interior(self):
        # Going through nodes in their topological order ensures that the parents always have their distributions already defined
        G_topo = nx.topological_sort(self.G)
        interior = [node for node in G_topo if list(self.G.predecessors(node))]
        for node in interior:
            preds = list(self.G.predecessors(node))
            if len(preds) == 1:
                P_node = tt.stack([self.G.nodes[preds[0]]['model']])
            elif len(preds) > 1:
                P_node = tt.stack([self.G.nodes[pred]['model'] for pred in preds])
            with self.model:
                self.G.nodes[node]['w'] = pm.Normal(f'w_{node}', 0, self.model.sdw, shape=[len(preds), 1])
                self.G.nodes[node]['mu'] = pm.Deterministic(f'mu_{node}',
                                                          pm.math.dot(self.G.nodes[node]['w'].T, P_node))
                self.G.nodes[node]['model'] = pm.Normal(f'{node}', self.G.nodes[node]['mu'], self.model.sdnode,
                                                       observed=self.G.nodes[node]['data'])

    def eval_model(self):
        """Computes the log probabilities for each sample of the parameters from the posterior."""
        model = self.model
        trace = self.trace


        data_dict = dict()
        logxp = {}
        for i, node in enumerate(self.G.nodes):
            data_dict[node] = self.G.nodes[node]['data']
            logxp[node] = getattr(model, node).logp
        logps = np.zeros(self.n_sample)

        if self.inference == 'map':
            trace.update(data_dict)
            logps = np.array([sum(logxp[node](trace) for node in self.G.nodes)])
        else:
            logps = np.zeros(self.n_sample)
            for i in range(self.n_sample):
                point = trace.point(i)
                point.update(data_dict)
                logps[i] = sum(logxp[node](point) for node in self.G.nodes)

        return logps

class MixedGraphModel(GraphModel):
    def __init__(self, G, **kwargs):
        super().__init__(G, **kwargs)
        self.M = kwargs.get('M', 3)
        self.N = 1000

        # How do I do this properly
        for node in self.G.nodes:
            self.N = (self.G.nodes[node]['data']).shape[0]
            break

    def model_nodes(self):
        self.model_params()
        self.model_z()
        self.model_leaves()
        self.model_interior()

    def model_z(self):
        with self.model:
            sdz = pm.Lognormal('sdz', 1)
            gamma = pm.Uniform('gamma', 0, 1)
            Z = pm.Normal('Z', 0, sdz, shape=[self.N, self.M])
            for node in self.G.nodes:
                self.G.nodes[node]['gamma'] = pm.Beta(f'gamma_{node}', 1/2, 1/2)
                self.G.nodes[node]['wz'] = pm.Normal(f'wz_{node}', 0, self.model.sdw, shape=self.M)
                self.G.nodes[node]['muz'] = pm.Deterministic(f'muz_{node}', pm.math.dot(Z, self.G.nodes[node]['wz']))

    def model_leaves(self):
        leaves = [node for node in self.G.nodes if not list(self.G.predecessors(node))]
        with self.model:
            for i, node in enumerate(leaves):
                self.G.nodes[node]['mu'] = self.G.nodes[node]['gamma'] * self.G.nodes[node]['muz']
                self.G.nodes[node]['model'] = pm.Normal(f'{node}', self.G.nodes[node]['mu'], self.model.sdnode,
                                                       observed = self.G.nodes[node]['data'])

    def model_interior(self):
        # Going through nodes in their topological order ensures that the parents always have their distributions already defined
        G_topo = nx.topological_sort(self.G)
        interior = [node for node in G_topo if list(self.G.predecessors(node))]
        for node in interior:
            preds = list(self.G.predecessors(node))
            if len(preds) == 1:
                P_node = tt.stack([self.G.nodes[preds[0]]['model']])
            elif len(preds) > 1:
                P_node = tt.stack([self.G.nodes[pred]['model'] for pred in preds])
            with self.model:
                self.G.nodes[node]['w'] = pm.Normal(f'w_{node}', 0, self.model.sdw, shape=[len(preds), 1])
                self.G.nodes[node]['mu'] = pm.Deterministic(f'mu_{node}', self.G.nodes[node]['gamma'] * self.G.nodes[node]['muz']
                                                           + (1-self.G.nodes[node]['gamma']) * pm.math.dot(self.G.nodes[node]['w'].T, P_node))
                self.G.nodes[node]['model'] = pm.Normal(f'{node}', self.G.nodes[node]['mu'], self.model.sdnode,
                                                       observed=self.G.nodes[node]['data'])
    def eval_gamma(self):
        mu_gammas = []
        std_gammas = []
        for node in self.G.nodes:
            mu_gammas.append(np.mean(self.trace[f'gamma_{node}']))
            std_gammas.append(np.std(self.trace[f'gamma_{node}']))
        return mu_gammas, std_gammas
