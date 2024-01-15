import operator
import numpy as np
import itertools
import random
import pickle
import os

import numpy as np
import pandas as pd
import networkx as nx
from networkx.exception import NetworkXNoCycle
from sklearn import linear_model, metrics

from network import Network
from model import Confounded
from ges import GES
from config import RESULTS_DIR

def generate_data(N, D, M, confounded_sets, alpha, num_datasets=1, store=False):
    """
    N: number of samples
    D: dimension of observed variables
    M: number of confounders
    num_confounded: number of variables out of the observed which are confounded
    alpha: List of either len=1 or len=D containing values for strength of confounding. If the list is of length 1
    then every confounded variable is assumed to use the same alpha. If it is of length D then each variable gets
    its own strength as specified in alpha.
    """
    for k in range(num_datasets):
        net = Network(N, D, M)
        net.gen_acyclic_net()
        nx.relabel_nodes(net.G, dict(zip(net.G.nodes, np.random.permutation(net.G.nodes))))
        remove = []
        for u, v in net.G.edges:
            for confounded_set in confounded_sets:
                if u in confounded_set and v in confounded_set:
                    remove.append((u, v))
        for edge in remove:
            net.G.remove_edge(*edge)
        # If only a single number alpha is specified, we share it across all variables that are confounded
        if len(alpha) == 1:
            _alpha = [alpha[0]] * D
            for i, _ in enumerate(_alpha):
                confounded_set_union = set()
                for cs in confounded_sets:
                    confounded_set_union = confounded_set_union.union(cs)
                if i not in confounded_set_union:
                    _alpha[i] = 0
        # If it's long enough to specify anything, we're good.
        elif len(alpha) == D:
            _alpha = alpha
        # Otherwise something is going wrong.
        else:
            raise NotImplemented("alpha has to be a list of either length 1 or of length equal to the number of variables to be generated.")
        net.alpha = np.array(_alpha)

        x, z = net.generate_data_alpha(confounded_sets)
        G = net.G
        if store:
            path = os.path.join(RESULTS_DIR, f'synthetic/alpha={alpha}')
            os.makedirs(path, exist_ok=True)
            x.to_csv(os.path.join(path, f'x{k}.csv'), index=None)
            z.to_csv(os.path.join(path, f'z{k}.csv'), index=None)
            with open(os.path.join(path, f'g{k}.csv'), 'wb') as g_file:
                pickle.dump(G, g_file)
            with open(os.path.join(path, f'conf{k}.csv'), 'wb') as conf_file:
                pickle.dump(confounded_set, conf_file)

    return x, z, G, confounded_set

if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    generate_data(5000, 10, 1, [range(3)], [1], 1)
    # for alpha in range(11):
    #     generate_data(5000, 10, 1, 3, alpha, 10)
