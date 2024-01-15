import itertools

import numpy as np
import pandas as pd

from co_shift_tests import co_pair_grouped
from co_test_type import CoShiftTestType
from dag_gen import gen_random_directed_er, gen_partitions
from invariance_confounding import gen_data_from_graph
from utils import data_check_soft,  partition_to_vector


def eval_gen_data(seed, fun_form):
    np.random.seed(seed)
    M = 5  # Num vars TODO
    N = 500
    C = 20  # 100

    partition_size = 2
    partition_size_cf = 2  # 19

    found_confounder = False

    data_check = True
    trial = 0

    # Set aside one node Z in the DAG that confounds at least two nodes.
    # If no such Z exists, skip this DAG.
    # If such Z exists, consider the children of Z as confounded nodes that should be tested, i.e. keep Z hidden and use only the causal parents of the nodes to estimate a pair(of contexts)wise discrepancy vector, then cmp. mutual info of this vector.
    # .. and consider all other nodes as nonconfounded nodes where we use ALL causal parents to get the pairwise discrepancy vector
    while (not found_confounder or not data_check) and trial < 1000:

        G = gen_random_directed_er(M, seed)
        for z in np.random.permutation(G.nodes):
            is_confounding = (len(list(G.successors(z))) > 1)
            if is_confounding:
                for i, j in itertools.combinations(G.nodes(), 2):
                    if (i == z) or (j == z):
                        continue

                    pa_i = list(G.predecessors(i))
                    pa_j = list(G.predecessors(j))
                    sub_pa_i = [p for p in pa_i if not (p == z) and not (p == j)]
                    sub_pa_j = [p for p in pa_j if not (p == z) and not (p == i)]
                    if (len(sub_pa_i) == 0) or (len(sub_pa_j) == 0):
                        continue
                    Z = z
                    found_confounder = True

        partitions = gen_partitions(C, partition_size, M, seed, G)
        X = gen_data_from_graph(N, G, C, partitions, seed, fun_form)

        # Make sure that data contains no nan/inf
        data_check = data_check_soft(X)
        seed = seed + 1
        trial = trial + 1

    if not found_confounder or not data_check:
        raise (ValueError(f"No confounders for this seed: {seed}"))

    return X, G, partitions, Z


class EvalShiftDiscovery():
    def __init__(self):
        self.record = {}

    def update(self, tp, tn, fp, fn, n_pa, caus, cfd, key):
        if not self.record.__contains__(key):
            self.record[key] = {}
            self.record[key] = \
                pd.DataFrame({'tp': [tp], 'tn': [tn], 'fp': [fp], 'fn': [fn], 'n_parents': [n_pa], 'caus': [caus], 'cfd': [cfd]})
        else:
            self.record[key].loc[len(self.record[key])] = [tp, tn, fp, fn, n_pa, caus, cfd]

def eval_rep_shift_discovery(reps, fun_form):

    test_types= [CoShiftTestType.PI_LINEAR, CoShiftTestType.PI_GP, CoShiftTestType.GRP_LINEAR]
    results = EvalShiftDiscovery()
    for test_type in test_types:
        seed = 0
        for it in range(reps):
            results = eval_shift_discovery( seed, fun_form , test_type, results)
            seed = seed + 1

def eval_shift_discovery(seed, fun_form , test_type, results):
    print(f'* TESTING: {test_type.value}, rep: {seed}')
    X, G, partitions, Z = eval_gen_data(seed, fun_form)
    for i in G.nodes:
        pa_i = list(G.predecessors(i))
        if len(pa_i) == 0:
            continue
        #if len(pa_i)>1: #DEBUG remove
        #    continue
        #Xc, yc = data_sub(X, pa_i), data_sub(X, i)
        #plt.scatter(Xc, yc)
        estim_shifts = co_pair_grouped(X, i, pa_i, test_type)
        part = partition_to_vector(partitions[i])
        true_shifts = [1 if x != y else 0 for k, x in enumerate(part) for y in part[k + 1:]]
        tn = sum([1 if true_shifts[i] == 0 and estim_shifts[i] == 0 else 0 for i in range(len(true_shifts))])
        tp = sum([1 if true_shifts[i] == 1 and estim_shifts[i] == 1 else 0 for i in range(len(true_shifts))])
        fn = sum([1 if true_shifts[i] == 1 and estim_shifts[i] == 0 else 0 for i in range(len(true_shifts))])
        fp = sum([1 if true_shifts[i] == 0 and estim_shifts[i] == 1 else 0 for i in range(len(true_shifts))])

        print(f'\tNode {i} Parents {pa_i} \t TP {tp}, FP {fp}, TN {tn}, FN {fn}')
        results.update(tp, tn, fp, fn, True, False, len(pa_i), test_type)

        for j in pa_i:
            estim_shifts = co_pair_grouped(X, i,[j], test_type) #TODO jointly for all parents  pa_i ?
            part = partition_to_vector(partitions[i])
            true_shifts = [1 if x != y else 0 for k, x in enumerate(part) for y in part[k + 1:]]
            tn = sum([1 if true_shifts[i] == 0 and estim_shifts[i] == 0 else 0 for i in range(len(true_shifts))])
            tp = sum([1 if true_shifts[i] == 1 and estim_shifts[i] == 1 else 0 for i in range(len(true_shifts))])
            fn = sum([1 if true_shifts[i] == 1 and estim_shifts[i] == 0 else 0 for i in range(len(true_shifts))])
            fp = sum([1 if true_shifts[i] == 0 and estim_shifts[i] == 1 else 0 for i in range(len(true_shifts))])

            print(f'\t\tNode {i} Parent {j} \t TP {tp}, FP {fp}, TN {tn}, FN {fn}')
            results.update(tp, tn, fp, fn, True, False, 1, test_type)

        # Anticausal Direction
        for j in pa_i:
            estim_shifts = co_pair_grouped(X, j, [i], test_type)

            part_i = partition_to_vector(partitions[i])
            part_j = partition_to_vector(partitions[j])
            vec_i = [1 if x != y else 0 for k, x in enumerate(part_i) for y in part_i[k + 1:]]
            vec_j = [1 if x != y else 0 for k, x in enumerate(part_j) for y in part_j[k + 1:]]

            true_shifts = np.maximum(vec_i, vec_j)


            tn = sum([1 if true_shifts[i] == 0 and estim_shifts[i] == 0 else 0 for i in range(len(true_shifts))])
            tp = sum([1 if true_shifts[i] == 1 and estim_shifts[i] == 1 else 0 for i in range(len(true_shifts))])
            fn = sum([1 if true_shifts[i] == 1 and estim_shifts[i] == 0 else 0 for i in range(len(true_shifts))])
            fp = sum([1 if true_shifts[i] == 0 and estim_shifts[i] == 1 else 0 for i in range(len(true_shifts))])
            print(f'\t\tNode {i} Anticause {j} \t TP {tp}, FP {fp}, TN {tn}, FN {fn}')
            results.update(tp, tn, fp, fn, False, False, 1, test_type)
    return results