import json
import os

import numpy as np
import pandas as pd

from coco.co_test_type import CoCoTestType, CoShiftTestType
from coco.co_co import CoCo
from coco.dag_gen import _random_nonlinearity, _linearity
from coco.utils import data_to_jci, G_to_adj
from coco.dag_confounded import DAGConfounded
from coco.mi_sampling import Sampler
from experiments.exp_coco.results_coco import ResultsCoCo

BIVARIATE_IDENTIFIABILITY = [[[(5, 0, 1), (5, 0, 0), (5, 0, 2), (5, 0, 3),(5, 0, 4), (5, 1, 0), (5, 1, 1), (5, 1, 2), (5, 1, 3),(5, 1, 4), (5, 2, 1), (5, 2, 0), (5, 2, 2), (5, 2, 3),(5, 2, 4), (5, 3, 1),
           (5, 3, 2), (5, 3, 3)]]]
BIVARIATE_IDENTIFIABILITY_CONTEXTS = [[[(2, 1, 1), (3, 1, 1), (3, 1, 2), (3, 0, 2), #2, 1, 2 and 3,1,3 not possible
                        (4, 3, 2),(4, 2, 2), (4, 1, 2), (4, 0, 2),
                          (6, 4, 3),(6, 3, 3), (6, 1, 3), (6, 0, 3),
                        (8, 5, 4),(8, 4, 4), (8, 2, 4), (8, 0, 4),
                        (10, 6, 5), (10, 5, 5), (10, 2, 5),(10, 0, 5),
                        (12, 8, 6), (12, 6, 6), (12, 3, 6),(12, 0, 6),
                        (15, 9, 7), (15, 7, 7), (15, 3, 7), (15, 0, 7),
                        (20, 12, 10), (20, 10, 10), (20, 4, 10), (20, 0, 10), (50, 12, 10), (50, 10, 10), (50, 5, 10), (50, 0, 10)]]]
VARY_CONTEXTS = [
    [[(3, 0, 1), (4, 0, 1), (6, 0, 1), (8, 0, 1), (10, 0, 1), (12, 0, 1)  # (6, 0, 3), (8, 0, 4), (10, 0, 5), (12, 0, 6)
      ]]]

def plot_bivariate_identifiability(n_c, path):
    BIVARIATE_IDENTIFIABILITY = {'noXshifts': {str(i): (n_c, 0, i) for i in range(n_c)},
                                 'oneXshift': {str(i): (n_c, 1, i) for i in range(n_c)},
                                 'twoXshifts': {str(i): (n_c, 2, i) for i in range(n_c)},
                                 'threeXshifts': {str(i): (n_c, 3, i) for i in range(n_c)},
                                 'halfXshifts': {str(i): (n_c, int( np.floor(n_c/2)), i)  for i in range(n_c)},
                                 'denseXshifts': {str(i): (n_c, int( np.floor(2*n_c/3)), i)  for i in range(n_c)},
                                 'unidentifiable': {str(i): (n_c, n_c-1, i)  for i in range(n_c)}
                                 }
    test_cases= BIVARIATE_IDENTIFIABILITY
    identifier = f'oracle_bivariate_{n_c}'
    reps = 50
    tex_file=  open(f"{path}/tex_coco/{identifier}.csv", "w+")
    res = run_coco_oracle(reps, test_cases, BIVARIATE=True, SHIFT_TEST = CoShiftTestType.SKIP, path=path)
    res.write_tex(test_cases, tex_file, range(n_c))

def run_coco_oracle(reps, test_cases, BIVARIATE=True, SHIFT_TEST = CoShiftTestType.SKIP, FUN_FORMS = [ (_linearity(), "LIN")], path=''):

    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    METHODS = ['oracle' ] #, 'cocoracle', 'coco']
    KNOWN_COMPONENTS = True

    sampler = Sampler()
    D_n = 500
    n_nodes = 10
    n_confounders = 2

    for fun_form, fun_str in FUN_FORMS:
        fnm = str(SHIFT_TEST) + '_' + str(CONFOUNDING_TEST) + '_' + fun_str + '_' + str(D_n)
        write_file = open(f"{path}/out_coco/LOG_oracle_{fnm}.csv", "w+")
        write_final_file = open(f"{path}/out_coco/RES_oracle_{fnm}.csv", "w+")

        for nm, lst, biv in [
            ('oracle', test_cases, BIVARIATE),
        ]:
            BIVARIATE = biv

            for fl in [write_file, write_final_file]:
                fl.write("\n\nEXPERIMENT: " + nm)
                fl.write("\nMETHOD: " + str(CONFOUNDING_TEST) + " x " + str(SHIFT_TEST))
                fl.write("\nFUN_FORM: " + fun_str)
                fl.flush()
            res = ResultsCoCo()
            for case_key in lst:
                print('CASE:', case_key)
                for xnm in lst[case_key]:
                        (n_contexts, n_shifts_observed, n_shifts_confounders) = lst[case_key][xnm]

                        seed = 0
                        for rep in range(reps):
                            print("REP", rep,  xnm, res.get_key(n_contexts, n_shifts_observed, n_shifts_confounders))
                            seed += 1
                            np.random.seed(seed)

                            list_n_shifts_observed = [0, n_shifts_observed]
                            list_n_shifts_confounders = [n_shifts_confounders]
                            if BIVARIATE:
                                n_components = 1
                                list_n_confounded_nodes = [2]
                                if rep % 2 == 0:  # np.random.uniform(0, 1) < 0.5:
                                    list_n_confounded_nodes = [0]  # or empty
                                dag = DAGConfounded(seed, n_contexts, 2, 1, list_n_confounded_nodes,
                                                    list_n_shifts_observed,
                                                    list_n_shifts_confounders, is_bivariate=True
                                                    )
                                D, Dobs = dag.gen_data(seed, D_n,
                                                       _functional_form=fun_form,
                                                       oracle_partition=False,
                                                       noise_iv=False
                                                       )
                                D_cocoracle, Dobs_cocoracle = dag.gen_data(seed, D_n,
                                                                           _functional_form=fun_form,
                                                                           oracle_partition=True,
                                                                           noise_iv=False
                                                                           )
                            else:
                                if KNOWN_COMPONENTS:
                                    n_components = n_confounders
                                else:
                                    raise NotImplementedError()
                                remaining = n_nodes - n_confounders  # + 1
                                list_n_confounded_nodes = []
                                for _ in range(n_confounders):
                                    if remaining == 0:
                                        size = 1
                                    else:
                                        size = np.random.choice([i for i in range(2, remaining + 1)], 1)[0]
                                    list_n_confounded_nodes.append(size)
                                    remaining = remaining - size

                                list_n_shifts_observed = [0, n_shifts_observed]
                                list_n_shifts_confounders = [n_shifts_confounders]

                                dag = DAGConfounded(seed, n_contexts, n_nodes, n_confounders, list_n_confounded_nodes,
                                                    list_n_shifts_observed,
                                                    list_n_shifts_confounders, is_bivariate=False
                                                    )
                                D, Dobs = dag.gen_data(seed, D_n,
                                                       _functional_form=fun_form,
                                                       oracle_partition=False,
                                                       noise_iv=False
                                                       )
                                D_cocoracle, Dobs_cocoracle = dag.gen_data(seed, D_n,
                                                                           _functional_form=fun_form,
                                                                           oracle_partition=True,
                                                                           noise_iv=False
                                                                           )

                            coco = CoCo(dag.G, D,
                                        CONFOUNDING_TEST,
                                        SHIFT_TEST,
                                        sampler, n_components, dag=dag
                                        )
                            cocoracle = CoCo(dag.G, D_cocoracle,
                                             CONFOUNDING_TEST,
                                             SHIFT_TEST,
                                             sampler, n_components, dag=dag
                                             )

                            # TODO select n_components.
                            # TODO the best dag acc. to sparse shift:

                            # mec, node_order = dag_to_mec(dag.G)
                            # cocos = [None for _ in range(len(mec))]
                            # for i, dag_i in enumerate(mec):
                            #    results_i = coco_mi.score_dag(dag_i, node_order)
                            #    sim_mi, sim_01, sim_pval, sim_cent, sim_causal_01, sim_causal_pval = results_i

                            #    cocos[i] = results_i
                            # print()

                            ##coco_dag= CoCo(dag, D,  CONFOUNDING_TEST,  SHIFT_TEST,  sampler  )
                            # coco_dag.score ...

                            coco_results = [('mi', coco, cocoracle)
                                            # , ('unknown_dag', ...) ->before using additional ones, double check
                                            ]
                            res.update(dag, n_contexts, n_shifts_observed, n_shifts_confounders, coco_results, METHODS)

                        res.write(write_file, n_contexts, n_shifts_observed, n_shifts_confounders, ['mi'], METHODS)
            res.write_final(write_final_file, ['mi'], METHODS)

        write_file.close()
        write_final_file.close()
        return res

