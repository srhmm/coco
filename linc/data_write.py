import csv
import os

import numpy as np

from linc.gen_context_data import gen_context_data
from linc.function_types import FunctionType
from linc.intervention_types import IvType
from linc.utils_pi import pi_group_map


def print_data():
    C_n = 5
    D_n = 500
    node_n = 5
    runs = 20
    seed = 42

    for fun_type in [FunctionType.LINEAR_GAUSS, FunctionType.GP_PRIOR]:
        for iv_type in [IvType.CONST, IvType.SCALE, IvType.PARAM_CHANGE]:
            for iv_per_node in [[0,0], [1,1], [1,2], [1,3]]:
                if iv_per_node[1]==1:
                    ivs = "iid"
                else:
                    ivs = "iv"+str(iv_per_node[1])

                dpath_concat = "/data_synth/concat/"+str(fun_type)+"_"+str(iv_type)+"_"+ivs+"/"

                os.mkdir(dpath_concat)

                for i in range(1, runs+1):
                    Dc, dag, Gc, \
                    _, _, _, _, partitions_X, observational_X = gen_context_data(C_n, D_n, node_n, seed, fun_type, iv_type,
                                                     iv_type_covariates=iv_type, iid_contexts=False,
                                                     iv_per_node=iv_per_node, node_density=0.5, scale=True,
                                                                                 iv_in_groups = True, #TODO
                                                     partition_search=False, overlap_observational=True)
                    seed = seed + 1

                    np.save(dpath_concat+"data"+str(i), Dc[0])

                    Dconcat = np.concatenate([Dc[c_i] for c_i in range(1,5)] )
                    np.save(dpath_concat+"data_interv"+str(i), Dconcat)

                    regime_concat = np.concatenate([[c_i for _ in range(len(Dc[c_i]))] for c_i in range(0,C_n-1)] )
                    regime_1 = np.concatenate([[0 for _ in range(len(Dc[c_i]))] for c_i in range(1)] )

                    interv_1 = np.concatenate([[[] for _ in range(len(Dc[c_i]))] for c_i in range(1)] )
                    interv_concat = np.concatenate([[[] for _ in range(len(Dc[c_i]))] for c_i in range(0,C_n-1)] )

                    intervened_nodes = [[] for c_i in range(0,C_n-1)]

                    interv_1 = np.concatenate([[[] for _ in range(len(Dc[c_i]))] for c_i in range(1)] )
                    interv_concat = [[intervened_nodes[c_i] for _ in range(len(Dc[c_i]))] for c_i in range(0,C_n-1)]
                    for c_i in range(0,C_n-1):
                        regime_index = c_i +1
                        # eg regime 1 (index 0) the first 500 rows: which nodes have been intervened
                        for node in range(node_n):
                            group_map = pi_group_map(partitions_X[node], C_n)
                            if group_map[regime_index] > 0: #the group is distinct from obs. one
                                intervened_nodes[c_i].append(node)
                    with open(dpath_concat+"intervention"+str(i)+".csv", "ab") as f:
                        for c_i in range(0,C_n-1):
                            for j in range(len(Dc[c_i])):
                                np.savetxt(f, [interv_concat[c_i][j]], newline="\n", fmt='%.18g', delimiter=",")

                    trueG = np.array([[(0 if n == 0 else 1) for n in dag.weight_mat[j, :]] for j in range(len(dag.weight_mat))])
                    np.save(dpath_concat+"DAG"+str(i), trueG)

                    np.savetxt(dpath_concat+"regime"+str(i)+".csv", regime_concat, fmt="%d", delimiter=",")