import json
import os

import numpy as np
import pandas as pd

from coco.co_test_type import CoCoTestType, CoShiftTestType
from coco.co_co import CoCo
from coco.dag_gen import _random_nonlinearity
from coco.utils import data_to_jci, G_to_adj
from coco.dag_confounded import DAGConfounded
from coco.mi_sampling import Sampler
from experiments.exp_coco.results_coco import ResultsCoCo


def run_coco2():
    SKIP_COCORACLE = True
    SHIFT_TEST = CoShiftTestType.PI_KCI
    KNOWN_COMPONENTS = True
    n_confounders = 2
    n_nodes = 10


    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    METHODS = ['oracle', 'cocoracle', 'coco']
    FUN_FORMS = [#(_linearity(), "LIN"),
                 (_random_nonlinearity(), "NLIN")
                 ]

    sampler = Sampler()
    D_n = 500

    reps = 20
    VARY_SHIFTS = [[[(5, 0, 1),(5, 0, 2), (5, 0, 3), (5, 1, 2), (5, 1, 3), (5, 2, 3), (5, 2, 2),(5, 3, 3) ]]]
    VARY_CONTEXTS = [[[(3, 0, 1), (4,0,1), (6, 0, 1), (8, 0, 1), (10, 0, 1), (12, 0, 1) # (6, 0, 3), (8, 0, 4), (10, 0, 5), (12, 0, 6)
                        ]]]

    CONFIRM = [[[(5, 0, 1), (5, 0, 0), (5, 0, 2), (5, 0, 3),(5, 0, 4), (5, 1, 0), (5, 1, 1), (5, 1, 2), (5, 1, 3),(5, 1, 4), (5, 2, 1), (5, 2, 0), (5, 2, 2), (5, 2, 3),(5, 2, 4), (5, 3, 1),
           (5, 3, 2), (5, 3, 3)]]]
    NEXT = [[[]]]

    EXP_IDENTIF = [
        [[(N, i, k)
         for i in range(min(k,max(2,np.int64(np.floor(N / 2)))))]
         for k in range(N-1)]for N in [2, 4, 6, 8]]

    EXP_NIDENTIF = [
        [[(N, i, k) for i in range(k, N)]
        for k in range(N)] for N
        in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]

    dataset_id = 0

    for fun_form, fun_str in FUN_FORMS:

        fnm = str(SHIFT_TEST) + '_' +str(CONFOUNDING_TEST) + '_' + fun_str + '_' + str(D_n) + '_' + str(n_nodes)+ '_' + str(n_confounders)
        path = "../experiments/out_coco/"
        write_file = open(f"{path}new_log_{fnm}.csv", "w+")
        write_final_file = open(f"{path}new_res_{fnm}.csv", "w+")

        for nm, lst, biv in [
            ('BIVARIATE, inc. confd shifts', CONFIRM, True),
           ('CONFOUNDED COMPONENTS,  inc. confd shifts', CONFIRM, False),
           #('BIVARIATE, inc. confd shifts', VARY_SHIFTS, True),
           #('BIVARIATE, inc. contexts', VARY_CONTEXTS, True),
           #('CONFOUNDED COMPONENTS,  inc. confd shifts', VARY_SHIFTS, False),
           #('CONFOUNDED COMPONENTS,  inc. contexts', VARY_CONTEXTS, False)
            ]:
            BIVARIATE = biv

            for fl in [write_file, write_final_file]:
                fl.write("\n\nEXPERIMENT: " + nm)
                fl.write("\nMETHOD: " + str(CONFOUNDING_TEST) + " x " + str(SHIFT_TEST))
                fl.write("\nFUN_FORM: " + fun_str)
                fl.flush()
            res = ResultsCoCo()
            for n in lst:
                for i in n:
                    for (n_contexts, n_shifts_observed, n_shifts_confounders) in i:
                        seed = 10
                        jci_folder =  f"../experiments/data_jci/{CONFOUNDING_TEST}_{SHIFT_TEST}_{fun_str}_{D_n}/"
                        if not os.path.exists(jci_folder):
                            os.mkdir(jci_folder)
                        jci_folder =  f"../experiments/data_jci/{CONFOUNDING_TEST}_{SHIFT_TEST}_{fun_str}_{D_n}/{n_contexts}_{n_shifts_observed}_{n_shifts_confounders}/"
                        if not os.path.exists(jci_folder):
                            os.mkdir(jci_folder)
                        for rep in range(reps):
                            print("REP", rep, res.get_key(n_contexts, n_shifts_observed, n_shifts_confounders))
                            seed = seed + 1
                            np.random.seed(seed)


                            list_n_shifts_observed = [0, n_shifts_observed]
                            list_n_shifts_confounders = [n_shifts_confounders]
                            if BIVARIATE:
                                n_components = 1
                                list_n_confounded_nodes = [2]
                                if  rep % 2 == 0: #np.random.uniform(0, 1) < 0.5:
                                    list_n_confounded_nodes = [0] # or empty
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
                                remaining = n_nodes - n_confounders# + 1
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
                                D_cocoracle, Dobs_cocoracle  = dag.gen_data(seed, D_n,
                                                       _functional_form=fun_form,
                                                       oracle_partition=True,
                                                       noise_iv=False
                                                       )

                            #write_data = open(f"../jci_data/D_{dataset_id}.csv", "w+")

                            #todo Dobs_cocoracle instead, if used.
                            A_jci, A_gaps_jci, A_edges_jci, A_true_jci = G_to_adj(dag.G, dag.G_true, n_contexts-1)
                            D_jci, _, json_args = data_to_jci(Dobs, seed, n_confounders)
                            # use index for evaluation!!


                            path = jci_folder #"../experiments/data_jci/"
                            D_df = pd.DataFrame(D_jci)
                            nms =   ["X"+str(i) for i in range(len(dag.G.nodes))] #["X"+str(i) for i in dag.G.nodes]
                            for ci in range(dag.n_c-1):
                                nms.append("X"+str(len(nms)+ci))
                            D_df.columns = nms

                            A_edges_jci = pd.DataFrame(A_edges_jci)

                            A_gaps_jci = pd.DataFrame(A_gaps_jci)

                            A_df = pd.DataFrame(A_jci)
                            A_df.columns = ["X"+str(i) for i in range(len(dag.G.nodes))]# ["X"+str(i) for i in dag.G.nodes]
                            A_true_df = pd.DataFrame(A_true_jci)
                            A_true_df.columns =  ["X"+str(i) for i in range(len(dag.G_true.nodes))]#["X"+str(i) for i in dag.G_true.nodes]
                            D_df.to_csv(f"{path}D_{dataset_id}.csv", header=True, index=False)
                            A_df.to_csv(f"{path}A_{dataset_id}.csv", header=True, index=False)
                            A_true_df.to_csv(f"{path}A_true_{dataset_id}.csv",  header=True, index=False)
                            A_edges_jci.to_csv(f"{path}A_edges_{dataset_id}.csv",  header=False, index=False)
                            A_gaps_jci.to_csv(f"{path}A_gaps_{dataset_id}.csv",  header=False, index=False)

                            with open(f"{path}D_{dataset_id}.json", 'w') as f:
                                json.dump(json_args, f)
                            dataset_id += 1

                            coco = CoCo(dag.G, D,
                                           CONFOUNDING_TEST,
                                           SHIFT_TEST,
                                           sampler, n_components, dag = dag
                                        )
                            if SKIP_COCORACLE:
                                cocoracle = coco
                            else: CoCo(dag.G, D_cocoracle,
                                           CONFOUNDING_TEST,
                                           SHIFT_TEST,
                                           sampler, n_components, dag = dag
                                        )

                            #TODO select n_components.
                            #TODO the best dag acc. to sparse shift:

                            #mec, node_order = dag_to_mec(dag.G)
                            #cocos = [None for _ in range(len(mec))]
                            #for i, dag_i in enumerate(mec):
                            #    results_i = coco_mi.score_dag(dag_i, node_order)
                            #    sim_mi, sim_01, sim_pval, sim_cent, sim_causal_01, sim_causal_pval = results_i

                            #    cocos[i] = results_i
                            #print()

                            ##coco_dag= CoCo(dag, D,  CONFOUNDING_TEST,  SHIFT_TEST,  sampler  )
                            #coco_dag.score ...

                            coco_results = [('mi', coco, cocoracle)
                                            #, ('unknown_dag', ...) ->before using additional ones, double check
                                            ]
                            res.update(dag, n_contexts, n_shifts_observed, n_shifts_confounders, coco_results, METHODS)

                        res.write(write_file, n_contexts, n_shifts_observed, n_shifts_confounders, ['mi'], METHODS)
            res.write_final(write_final_file, ['mi'], METHODS)

        write_file.close()
        write_final_file.close()

