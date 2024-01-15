from collections import defaultdict, OrderedDict
import numpy as np
import networkx as nx
import itertools
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

from dag_gen import gen_random_directed_er, gen_partitions, gen_data_from_graph
from utils import data_sub, partition_to_map, partition_to_vector
from sparse_shift.kcd import KCD
# from sklearn.linear_model import Ridge
import datetime
from causallearn.utils.cit import CIT


# THIS
def oracle_validation():
    M = 10 # Num vars
    seed = 1
    G = gen_random_directed_er(M, seed)
    partitions = gen_partitions(100, 10, M, seed, G)
    MIs = []
    for i in np.random.permutation(G.nodes):
        if len(list(G.successors(i))) > 1:
            Z = i
            break
    for i, j in itertools.combinations(G.nodes(), 2):
        part_i = partition_to_vector(partitions[i])
        part_j = partition_to_vector(partitions[j])
        vec_i = [1 if x != y else 0 for k, x in enumerate(part_i) for y in part_i[k+1:]]
        vec_j = [1 if x != y else 0 for k, x in enumerate(part_j) for y in part_j[k+1:]]
        if i in G.successors(Z) and j in G.successors(Z):
            part_z = partition_to_vector(partitions[Z])
            vec_z = [1 if x != y else 0 for k, x in enumerate(part_z) for y in part_z[k+1:]]
            vec_i = np.maximum(vec_i, vec_z)
            vec_j = np.maximum(vec_j, vec_z)
        mi_ij = adjusted_mutual_info_score(vec_i, vec_j)
        MIs.append(mi_ij)
    for k, (i, j) in enumerate(itertools.combinations(G.nodes(), 2)):
        cf = (i in G.successors(Z) and j in G.successors(Z))
        print(i, j, MIs[k], G.has_edge(i,j), cf, MIs[k] > 0.005)


def data_gen(seed):
    num_env = 10
    num_vars = 30
    size_part = 3
    N = 10000
    G = gen_random_directed_er(num_vars, seed, .3)
    partitions = gen_partitions(num_env, size_part, num_vars, seed, G=G)
    X = gen_data_from_graph(N, G, num_env, partitions)
    for e in range(num_env):
        for i in G.nodes():
            m = np.min(X[e, i])
            M = np.max(X[e, i])
            D = M - m
            X[e, i] = (X[e,i] - m)/D
    # for i in G.nodes():
    #     pa_i = list(G.predecessors(i))
    #     BREAK = False
    #     if len(pa_i) == 1:
    #         j = pa_i[0]
    #         BREAK = True
    #         for part in partitions[i]:
    #             print(part)
    #             for e in part:
    #                 plt.scatter(X[e, j], X[e, i], label=f'{e}', alpha=.3)
    #         plt.legend()
    #     if BREAK == True:
    #         break
    return G, partitions, X

def test_mechanism():
    seed = 1
    G, partitions, X = data_gen(seed)
    E = X.shape[0]
    for m in np.random.permutation(G.nodes()):
        parents = list(G.predecessors(m))
        if len(parents) > 0:
            break
    partition = partitions[m]
    vec = partition_to_vector(partition)
    print(parents, partition, vec)
    pvals_same = []
    pvals_diff = []
    # param_same = []
    # param_diff = []
    k = 0
    KCI = CIT(X, 'kci')
    for e1, e2 in itertools.combinations(range(E), 2):
        k += 1
        print('{0} - {1}/{2}'.format(datetime.datetime.now().strftime('%H:%M:%S'), k, E*(E-1)/2))
        X1 = X[e1, parents].T
        X2 = X[e2, parents].T
        X_pa = np.vstack((X1[parents, :].T, X2[parents, :].T)) 
        X_m = np.concatenate((X1[m].T, X2[m].T))
        z = np.asarray([0] * X1.shape[1] + [1] * X2.shape[1])
        _, pvalue = KCD(n_jobs=8).test(
            X_pa,
            X_m,
            z,
            reps=1000,
            )
        if vec[e1] == vec[e2]:
            pvals_same.append(pvalue)
        else:
            pvals_diff.append(pvalue)
    print(np.mean(pvals_same), np.mean(pvals_diff))
