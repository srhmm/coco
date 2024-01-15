import cdt
from causaldag import rand
from causaldag import unknown_target_igsp
import causaldag as cd
import numpy as np
import random
from causaldag import partial_correlation_suffstat, partial_correlation_test
from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester, MemoizedCI_Tester
from graphical_model_learning import gsp
from statistics import mean, stdev
from function_types import FunctionType
from gen_context_data import gen_context_data
from intervention_types import IvType
from out import Out


class UTIGSP:
    """
    UTIGSP on data from multiple contexts. (A wrapper for experiments)
    """

    def __init__(self, cpdag, dag):
        self.domains_ = []
        self.cpdag = cpdag  # adj matrix
        self.dag = dag
        self.maxenv_only = True

    def add_environment(self, interventions):
        self.domains_.append(interventions)

    def get_min_dags(self, soft):
        dag = self._utigsp(self.domains_)
        self.min_dags_ = dag
        return self.min_dags_

    def get_min_cpdag(self, soft):
        cpdag = self.min_dags_# (np.sum(self.min_dags_, axis=0) > 0).astype(int)
        return cpdag

    def _utigsp(self, Xs):
        C_n = len(Xs)
        node_n = len(Xs[0])
        obs_samples = Xs[0]
        iv_samples_list = [Xs[i] for i in range(1, C_n)]
        obs_suffstat = partial_correlation_suffstat(obs_samples)
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)

        # Create conditional independence tester and invariance tester
        alpha = 1e-3
        alpha_inv = 1e-3
        ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

        # Run UT-IGSP
        n_settings = C_n - 1
        nodes = set(range(node_n))
        setting_list = [dict(known_interventions=[]) for _ in range(n_settings)]
        est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)

        return est_dag
        #true_adj = arc_to_adj(dag.arcs, node_n)
        #est_adj = arc_to_adj(est_dag.arcs, node_n)

        #sid = cdt.metrics.SID(true_adj, est_adj)
        #sid_cpdag = cdt.metrics.SID_CPDAG(true_adj, est_adj)
        #shd = cdt.metrics.SHD(true_adj, est_adj)
        #shd_cpdag = cdt.metrics.SHD_CPDAG(true_adj, est_adj)
