import os

import numpy as np

from coco.co_co import CoCo
from coco.dag_confounded import DAGConfounded
from coco.dag_gen import _random_nonlinearity
from coco.mi_sampling import Sampler

from coco.co_test_type import CoCoTestType, CoShiftTestType, CoDAGType
from experiments.method_types import MethodType
from experiments.results_coco import ResultsCoCo
from experiments.run_coco import run_coco, run_fci


def example_coco():
    """ An example of generating a simple dataset with latent confounders and different contexts and running CoCo.
    """


def example_coco_and_oracles():
    """ An example of generating a simple dataset with latent confounders and different contexts and running CoCo and different oracle versions.
    """
    n_nodes, n_confounders, n_contexts, n_shifts_observed, n_shifts_confounders, D_n = (5, 1, 5, 1, 1, 500)
    fun_form = _random_nonlinearity()

    # hyperparameters
    seed = 1
    FCI_INDEP_TEST = 'fisherz'

    # Data Generation
    sampler = Sampler()
    dag = DAGConfounded(seed, n_contexts, n_nodes, n_confounders,  n_shifts_observed, n_shifts_confounders,
                        is_bivariate=False)

    D, Dobs = dag.gen_data(seed, D_n, _functional_form=fun_form, oracle_partition=True, noise_iv=False)

    # Methods
    fci_jci, D, Dobs, seed = run_fci(dag,D,  Dobs, D_n, fun_form, MethodType.FCI_JCI, FCI_INDEP_TEST, seed)

    coco = CoCo(D, dag.G.nodes,
                co_test=CoCoTestType.MI_ZTEST, shift_test=CoShiftTestType.PI_KCI,
                dag_discovery=CoDAGType.MSS,
                sampler=sampler, n_components=None, dag=dag)
    coco_oZ = CoCo(D, dag.G.nodes,
                co_test=CoCoTestType.MI_ZTEST, shift_test=CoShiftTestType.PI_KCI,
                dag_discovery=CoDAGType.MSS,
                sampler=sampler, n_components=n_confounders, dag=dag)
    coco_oG_oZ = CoCo(D, dag.G.nodes,
                     co_test=CoCoTestType.MI_ZTEST, shift_test=CoShiftTestType.PI_KCI,
                     dag_discovery=CoDAGType.SKIP,
                     sampler=sampler, n_components=n_confounders, dag=dag)
    coco_oG = CoCo(D, dag.G.nodes,
                     co_test=CoCoTestType.MI_ZTEST, shift_test=CoShiftTestType.PI_KCI,
                     dag_discovery=CoDAGType.SKIP,
                     sampler=sampler, n_components=None, dag=dag)
    coco_oPi_oG_oZ = CoCo(D, dag.G.nodes,
                  co_test=CoCoTestType.MI_ZTEST, shift_test=CoShiftTestType.SKIP,
                  dag_discovery=CoDAGType.SKIP,
                  sampler=sampler, n_components=n_confounders, dag=dag)
    coco_oPi_oG = CoCo(D, dag.G.nodes,
                  co_test=CoCoTestType.MI_ZTEST, shift_test=CoShiftTestType.SKIP,
                  dag_discovery=CoDAGType.SKIP,
                  sampler=sampler, n_components=None, dag=dag)

    # Eval

    tp, fp, tn, fn,  f1  = fci_jci.eval_confounded(dag, MethodType.FCI_JCI)
    print(f'\tJFCI-cfd: \t(f1={np.round(f1,2)})\t(tp={tp}, tn={tn}, fp={fp}, fn={fn})')
    tp, fp, tn, fn,  f1  = fci_jci.eval_causal(dag, MethodType.FCI_JCI)
    print(f'\tJFCI-caus: \t(f1={np.round(f1,2)})\t(tp={tp}, tn={tn}, fp={fp}, fn={fn})')
    #
    def illu_res(res, nm):
        tp, fp, tn, fn, f1, tpr, fpr = res
        print(f'\t{nm}: \t(f1={np.round(f1, 2)})'
              f'\t(tp={tp}, tn={tn}, fp={fp}, fn={fn})'
              f'\t(tpr={np.round(tpr)}, fpr={np.round(fpr)})')
    def illu_sets(res, nm):
        jacc, ari, ami, tp, fp, tn, fn, f1, tp_adjusted, fp_adjusted, tn_adjusted, fn_adjusted, f1_adjusted = res
        print(
            f'\t{nm}: \t(f1={np.round(f1, 2)})\t(tp={tp}, tn={tn}, fp={fp}, fn={fn}))')
    def illu_caus(res, nm):
        tp, fp, tn, fn, f1, tpr, fpr , _, _, _, _,_, _, _, _,_, _, _, _,_, _, _, _, = res
        print(
            f'\t{nm}: \t(f1={np.round(f1, 2)})\t(tp={tp}, tn={tn}, fp={fp}, fn={fn}))')

    print(f'1. Discovering Confounded node pairs:')
    for res, nm in [(coco_oPi_oG_oZ.eval_oracle_edges(dag), 'CoCo, O(Pi, G, |Z|)'),
                    (coco_oPi_oG.eval_oracle_edges(dag), 'CoCo, O(Pi, G)'),
                    (coco_oG_oZ.eval_estimated_edges(dag), 'CoCo, O(G, |Z|)'),
                    (coco_oG.eval_estimated_edges(dag), 'CoCo, O(G)'),
                    (coco_oZ.eval_estimated_edges(dag), 'CoCo, O(|Z|)'),
                    (coco.eval_estimated_edges(dag), 'CoCo')
                    ]:
        illu_res(res, nm)


    print(f'2. Discovering the number of confounders:')
    for res, nm in [(coco_oPi_oG.eval_estimated_graph_cuts(dag), 'CoCo, O(Pi, G)'),
                    (coco_oG.eval_estimated_graph_cuts(dag), 'CoCo, O(G)'),
                    (coco.eval_estimated_graph_cuts(dag), 'CoCo')
                    ]:
        illu_sets(res, nm)
    print(f'3. Discovering causal edges:')
    for res, nm in [(coco_oG.eval_causal(dag), 'CoCo, O(G)'),
                    (coco_oZ.eval_causal(dag), 'CoCo, O(|Z|)'),
                    (coco.eval_causal(dag), 'CoCo')
                    ]:
        illu_caus(res, nm)




def example_run(path):
    """ An example of running example_coco() repeatedly for different parameters (in CASES) and aggregating the results.
    """
    METHODS = [MethodType.COCO]
    # In this example, vary the mechanism shifts of the confounder
    # pattern: (n_observed_vars, n_latent_vars, n_contexts, n_shifts_observed, n_shifts_latent)
    case1 = (5, 1, 6, 1, 1, 'n_Zshift_1')
    case2 = (5, 1, 6, 1, 2, 'n_Zshift_1')
    case3 = (5, 1, 6, 1, 3, 'n_Zshift_1')

    reps = 2
    CASES = {'5_1_6_1_i': {'1': case1, '2': case2, '3': case3}}

    res = run_coco(reps, CASES, METHODS, path=path)
    for identifier in CASES:
        res.write_methods_tex(identifier, path + "/tex_coco", CASES[identifier], METHODS)