import numpy as np
from pulp import PulpSolverError

import upq
from out import Out
from pi_dag_search import pi_dag_search
from pi_mec import PiMEC
from pi_mechanism_search import pi_mechanism_search
from sparse_shift import cpdag2dags
from pi_tree import PiTree
from tree_search import tree_search
from vsn import Vsn


class LINC:
    """
    LINC (A wrapper for experiments)
    """

    def __init__(self, cpdag, dag, rff, mdl_gain, pi_search, ILP, known_mec, clus):
        self.domains_ = []
        self.cpdag = cpdag  # adj matrix
        self.dag = dag
        self.maxenv_only = True
        self.known_mec=known_mec
        #self.tp_metric = tp_metric
        self.rff = rff
        self.pi_search=pi_search
        self.min_dags_ = np.zeros((len(cpdag), len(cpdag)))
        self.min_gains_, self.min_sig_ = None, None
        self.min_obj_ = None
        self.min_conf_ = None
        self.vsn = Vsn(rff=rff, ilp_partitions=True, mdl_gain=mdl_gain, ilp=ILP,pi_search=pi_search, clus=clus)

    def add_environment(self, interventions):
        self.domains_.append(interventions)

    def get_mechanisms(self, y, subsets):
        return self._linc_mechanismsearch(self.domains_,y, subsets)


    # LINC for DAG search
    def get_min_dags(self, soft):
        mec, dag, mdl, gains, sig = self._linc_safe(self.domains_)
        self.min_obj_ = mec
        self.min_dags_ = dag
        self.min_mdl_ = mdl
        self.min_gains_, self.min_sig_ = gains, sig
        #self.min_gains_, self.min_sig_ = mec.conf_mat(dag)
        return self.min_dags_

    #TODO should this return some undirected edges if not conf?
    def get_min_cpdag(self, soft):
        cpdag =self.min_dags_# (np.sum(self.min_dags_, axis=0) > 0).astype(int)
        return cpdag

    def _linc_safe(self, Xs):
        if not self.known_mec:
            return self._linc_dagsearch(Xs)
        try:
            return self._linc(Xs)
        except (PulpSolverError):
            print("(LINC) Switching from ILP to exhaustive")
            self.vsn.ilp=False
            return self._linc(Xs)

    # LINC for DAG search (unknown MEC)
    def _linc_dagsearch(self, Xs):
        dag = pi_dag_search(Xs, self.vsn)
        adj = dag.get_adj().T
        mdl = dag.get_graph_mdlcost()
        gain_mat, sig_mat =dag.conf_mat(adj)
        return dag, adj, mdl, gain_mat, sig_mat

    # LINC for a node in the DAG
    def _linc_mechanismsearch(self, Xs, y, subsets):
        pis, scores = pi_mechanism_search(Xs, y, subsets, self.vsn)
        imin = min(range(len(scores)), key=scores.__getitem__)
        pistar, pistar_score = pis[imin], scores[imin]
        return pistar, pistar_score, pis, scores

    # MEC search
    def _linc(self, Xs):
        C_n = len(Xs)
        D_n = Xs[0].shape[0]
        X_n = Xs[0].shape[1]

        # Search over all DAGs in true MEC
        dags = cpdag2dags(self.cpdag)

        mdl_min = np.inf
        dag_min = np.zeros((X_n, X_n))
        mec = PiMEC(C_n, X_n, Xs, self.vsn)

        for cand in dags:
            mdl = mec.eval_adj(cand, gain=self.vsn.mdl_gain)
            if mdl < mdl_min:
                mdl_min = mdl
                dag_min = cand

        gain_min,  sig_min = mec.conf_mat(dag_min, gain=self.vsn.mdl_gain)
        return mec, dag_min, mdl_min, gain_min, sig_min

    #The same logic but hacky code (passing score caches from each tree class to the next)
    def _linc_tree_for_mec(self, Xs, rff, gain):
        C_n = len(Xs)
        D_n = Xs[0].shape[0]
        X_n = Xs[0].shape[1]

        # Search over all DAGs in true MEC
        dags = cpdag2dags(self.cpdag)
        T = PiTree(C_n, X_n, Xs, self.vsn)
        score_cache, mdl_cache, pi_cache = T.score_cache, T.mdl_cache, T.pi_cache
        mdl_min = np.inf
        dag_min = np.zeros((X_n, X_n))
        tree_min = T

        for cand in dags:
            T = PiTree(C_n, X_n, Xs, self.vsn)
            _ = T.initial_edges(upq.UPQ(), Out("", vb=False, tofile=False), D_n,
                                cmp_score=False, score_cache = score_cache, mdl_cache=mdl_cache, pi_cache=pi_cache)  # skips computing initial scores for all node pairs

            for i in range(len(cand)):
                for j in range(len(cand[i, :])):
                    if cand[i, j] != 0:
                        gain, score, mdl, pi, _ = T.eval_edge(T.init_edges[j][i])
                        T.add_edge(i, j, score, mdl, pi)
            mdl = T.get_graph_mdlcost()
            if gain:
                mdl= T.get_graph_score()
            #cache = T.score_cache
            if mdl < mdl_min:
                mdl_min = mdl
                dag_min = cand
                tree_min = T

        gain_min, sig_min = dag_min, dag_min #TODO no signif computed here yet

        return tree_min, dag_min, mdl_min, gain_min, sig_min



