import itertools
from collections import defaultdict

import numpy as np
import sklearn
from sklearn.metrics import pair_confusion_matrix

from co_shift_tests import co_pair_grouped, co_shift_test
from co_test_type import CoCoTestType, CoShiftTestType, CoDAGType
from co_causal_tests import test_causal
from coco.eval import eval_confounded, eval_causal_confounded
from coco.utils_coco import to_affinity_mat, graph_cuts

from coco.utils_coco import node_similarities
from dag_confounded import DAGConfounded, clus_sim_spectral
from mi import mutual_info_scores
from mi_sampling import Sampler
from co_confounding_tests import test_confounded_ami, test_confounded_Z_mi_sampling, test_confounded_Z_ami_sampling, \
    test_confounded
from sparse_shift import MinChange, cpdag2dags, dag2cpdag
from utils import shifts_to_map, map_to_shifts, partition_to_map, f1_score


class CoCo:
    def __init__(self, D, observed_nodes,
                 co_test: CoCoTestType,
                 shift_test: CoShiftTestType,
                 dag_discovery: CoDAGType,
                 sampler:Sampler,
                 #oracle args
                 n_components=None,
                 dag=None,#Provide dag if oracle stuff should be computed
                 node_nms=None, #debug, remove
                 alpha_shift_test=0.5):
        self.node_nms = node_nms
        self.alpha_shift_test = alpha_shift_test
        self.nodes = observed_nodes #G.nodes
        self.n_contexts = D.shape[0]

        self.D = D
        self.maps_estimated = defaultdict(list) #partitions for each node, resp. to the true dag
        self.pval_estimated = defaultdict(list) #pvals of mechanism shift tests btw each pair of contexts
        self.maps_estimated_dag = defaultdict(list)  #for the last dag that _score_dag was called on
        self.co_test = co_test
        self.shift_test = shift_test
        self.sampler = sampler

        if dag_discovery.value == CoDAGType.SKIP.value:
            G_ = dag.G
            print(f"(COCO-dag) Oracle")
        else:
            G_ = self._discover_dag(dag.G, dag_discovery)
        self.G_ = G_
        self._score_DAG(G_)
        self._estimated_similarities()

        if dag is not None:
            self._oracle_similarities(dag)
            self._oracle_graph_cuts()
            if n_components is not None:
                self._oracle_graph_cuts_n(n_components)

        self._estimated_graph_cuts()
        if n_components is not None:
            self._estimated_graph_cuts_n(n_components)  # TODO select n. components


    def _discover_dag(self, G, dag_discovery, soft=True):
        adj = np.array([np.zeros(len(G.nodes)) for _ in range(len(G.nodes))])
        for nj in G.nodes:
            for ni in G.predecessors(nj):
                i = [n for n in G.nodes].index(ni)
                j = [n for n in G.nodes].index(nj)
                adj[i][j] = 1

        true_cpdag = dag2cpdag(adj)

        if dag_discovery.value != CoDAGType.MSS.value:
            raise NotImplementedError("Other methods for discovering causal DAGs not implemented.")

        mch_kci = MinChange(true_cpdag, alpha=0.05, scale_alpha=True, test='kci', test_kwargs={
                "KernelX": "GaussianKernel",
                "KernelY": "GaussianKernel",
                "KernelZ": "GaussianKernel",
            })
        print(f"(COCO-dag) Discovering a causal DAG w MSS (Perry et al. 2022)")
        for n_env, X in enumerate(self.D):
            mch_kci.add_environment(np.array(X.T)) #.T)

        min_cpdag = mch_kci.get_min_cpdag(soft)
        print(
            f"(COCO-dag)  MEC size: { len(cpdag2dags(true_cpdag))}, unoriented: { np.sum((true_cpdag + true_cpdag.T) == 2) // 2}, left unoriented: {np.sum((min_cpdag + min_cpdag.T) == 2) // 2}")

        import networkx as nx
        G_ = nx.DiGraph([])
        edges = []
        for ind, i in enumerate(G.nodes):
            for indj, j in enumerate(G.nodes):
                if min_cpdag[ind, indj] == 1:
                    edges.append((i, j))
        G_.add_nodes_from(G.nodes)
        G_.add_edges_from(edges)
        self.G_estimated = G_
        return G_

    def eval_causal(self, dag):
        return eval_causal_confounded(self.G_, dag, self.sim_01, dag.nodes_confounded)


    def _score_DAG(self, G):
        ''' Runs CoCo's mechanism shift test on each node given its parents in a DAG.
        self.maps_estimated_i: for node_i, partition of contexts that agrees with the shift tests (P_c1(V_i | pa_i)=/=P_c2(V_i | pa_i)).
        self.pval_estimated_i: for node_i, pvals of the shift test between context pairs, pval(P_c1(V_i | pa_i)=/=P_c2(V_i | pa_i)).

        :param G: true DAG or DAG inferred with MSS.
        :return: self.maps_estimated, self.pval_estimated: partitions showing mechanism shifts;
        '''
        print(f"(COCO-shift) Discovering causal mechanism shifts w {self.shift_test}")
        for node_i in self.nodes:
            pa_i = [n for n in G.predecessors(node_i)]
            map_i, pval_mat_i = co_shift_test(self.D, node_i, pa_i, self.shift_test, self.alpha_shift_test)

            self.maps_estimated[node_i] = map_i
            self.pval_estimated[node_i] = pval_mat_i

    def score_other_DAG(self, adj, node_order):
        maps_estimated_dag = defaultdict(list)

        for i, node_i in enumerate(node_order):
            pa_i = [n for j, n in enumerate(node_order) if adj[j][i]==1]
            map_i = co_shift_test(self.D, node_i, pa_i, self.shift_test)
            maps_estimated_dag[node_i] = map_i
        sim_mi, sim_01, sim_pval, sim_cent, sim_causal_01, sim_causal_pval =\
            node_similarities(maps_estimated_dag, self.nodes, self.co_test, self.sampler)
        return sim_mi, sim_01, sim_pval, sim_cent, sim_causal_01, sim_causal_pval

    def score_bivariate(self, D, n_i, n_j, prnt=True):
        pa_j = [n_i]
        map_j = co_shift_test(D, n_j, pa_j, self.shift_test)
        pa_i = [n_j]
        map_i = co_shift_test(D, n_i, pa_i, self.shift_test)

        decision_mi, mi, pval_mi, stdev_cent_sampling = test_confounded(map_i, map_j, self.co_test, self.sampler)

        mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)

        decision_causal, pval_causal = test_causal(h1, h2, mi, stdev_cent_sampling)  # sampler

        if prnt:
            print("EDGE", n_i, "-", n_j, "\t", map_i, map_j, decision_mi, decision_causal)

    def _estimated_similarities(self):

        print(f"(COCO-confounding) Discovering confounding w {self.co_test}")
        maps_estimated = self.maps_estimated
        self.sim_mi, self.sim_01, self.sim_pval, self.sim_cent, self.sim_causal_01, self.sim_causal_pval = \
            node_similarities(maps_estimated, self.nodes, self.co_test, self.sampler)

    def _oracle_similarities(self, dag):

        print(f"(COCO-dag) Oracle")
        print(f"(COCO-shift) Oracle")
        print(f"(COCO-confounding) Discovering confounding w {self.co_test}")
        maps_dependent = dag.maps_nodes
        self.o_sim_mi, self.o_sim_01, self.o_sim_pval, self.o_sim_cent, self.o_sim_causal_01, self.o_sim_causal_pval = \
            node_similarities(maps_dependent, self.nodes, self.co_test, self.sampler)
        maps_independent = dag.maps_nodes_star
        self.o_null_sim_mi, self.o_null_sim_01, self.o_null_sim_pval, _, _, _ = \
            node_similarities(maps_independent, self.nodes, self.co_test, self.sampler)

    def _oracle_graph_cuts(self):
        self.oracle_cuts = graph_cuts(self.nodes, self.o_sim_mi, self.o_sim_01, self.o_sim_pval, n_components=None)

    def _oracle_graph_cuts_n(self, n_components):
        self.oracle_cuts = graph_cuts(self.nodes, self.o_sim_mi, self.o_sim_01, self.o_sim_pval, n_components=n_components)

    def _estimated_graph_cuts(self):
        self.estimated_cuts = graph_cuts(self.nodes, self.sim_mi, self.sim_01, self.sim_pval, n_components=None)

    def _estimated_graph_cuts_n(self, n_components):
        self.estimated_cuts = graph_cuts(self.nodes, self.sim_mi, self.sim_01, self.sim_pval, n_components=n_components)

    def confounded_components_to_labels(self, confounded_components):
        ''' Converts a list of confounded subsets to labels

        :param confounded_components: [[n_i for n_i confounded by c_j] for c_j in confounders]
        :return: [0 if n_i not in confounded_components else #c_j]
        '''

        index = [i for i in self.nodes]
        labels = [0 for _ in self.nodes]

        if len(confounded_components)==0:
            return labels
        g_i = 1 #0 means unconfounded
        for g in confounded_components:
            for n_i in g:
                i = np.min(np.where(index==n_i))
                labels[i] = g_i
            g_i = g_i + 1
        return labels

    '''
    def oracle_graph_cut_n(self, n_components):
        return clus_sim_spectral(self.sim_01, n_components)

    def oracle_graph_cut(self):
        def disagreement(n_i, n_j, labels, sim):
            if labels[n_i-1]==labels[n_j-1]:
                return sim[n_i-1][n_j-1]
            else:
                return -sim[n_i - 1][n_j - 1]

        eval_n = [0 for _ in range(1,len(self.nodes)-2)]
        for n_components in range(1,len(self.nodes)-2):
            labels = clus_sim_spectral(self.sim_01, n_components)
            eval_n[n_components-1] = sum([disagreement(n_i, n_j, labels, self.sim_vi) for n_i, n_j in itertools.combinations(self.nodes, 2)])
        eval = [abs(i) for i in eval_n]
        n_star = eval.index(min(eval))+1
        return clus_sim_spectral(self.sim_01, n_star)
    '''
#### Evaluation
    def show_oracle_confounded_edges(self):
        return self._show_edges(self.o_sim_01, self.o_sim_mi, self.o_sim_pval)

    def show_estimated_confounded_edges(self):
        return self._show_edges(self.sim_01, self.sim_mi, self.sim_pval)

    def show_oracle_confounded_true_partitions(self):
        return self._show_edges(self.o_null_sim_01, self.o_null_sim_mi, self.o_null_sim_pval)

    def eval_oracle_edges(self, dag: DAGConfounded):
        return eval_confounded(self.o_sim_01, dag.nodes_confounded)

    def eval_estimated_edges(self, dag: DAGConfounded):
        return eval_confounded(self.sim_01, dag.nodes_confounded)

    def eval_oracle_edges_true_partitions(self, dag: DAGConfounded, s):
        tp, fp, tn, fn, _, _, _, _, _, _ = eval_confounded(self.o_null_sim_01, [])
        assert tp == 0 and fn == 0
        return fp, tn

    def eval_oracle_graph_cuts(self, dag):
        cuts = self.oracle_cuts
        true_cuts = dag.nodes_confounded
        return self._eval_cuts(true_cuts, cuts)

    def eval_estimated_graph_cuts(self, dag):
        cuts = self.estimated_cuts
        true_cuts = dag.nodes_confounded
        return self._eval_cuts(true_cuts, cuts)

    def _eval_cuts(self, true_cuts,  cuts):
        true_labels = self.confounded_components_to_labels(true_cuts)
        labels = self.confounded_components_to_labels(cuts)

        jacc = sklearn.metrics.jaccard_score(true_labels, labels, average='weighted') #todo average
        ari = sklearn.metrics.adjusted_rand_score(true_labels, labels)
        ami = sklearn.metrics.adjusted_mutual_info_score(true_labels, labels)
        (tn, fp), (fn, tp) = pair_confusion_matrix(true_labels, labels)

        # convert to Python integer types, to avoid overflow or underflow
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        tp_adjusted, fp_adjusted, fn_adjusted, tn_adjusted = 0, 0, 0, 0
        for i, j in itertools.combinations(range(len(true_labels)), 2):
            if true_labels[i] > 0 or true_labels[j] > 0 or labels[i] > 0 or labels[j] > 0:
                if labels[i]==labels[j] and labels[i]>0:
                    if true_labels[i]==true_labels[j] and true_labels[i]>0:
                        tp_adjusted += 1
                    else:
                        fp_adjusted += 1
                else:
                    if true_labels[i]==true_labels[j] and true_labels[i]>0:
                        fn_adjusted += 1
                    else:
                        tn_adjusted += 1

        f1 = f1_score(tp, fp, fn)
        f1_adjusted = f1_score(tp_adjusted, fp_adjusted, fn_adjusted)

        return jacc, ari, ami, tp, fp, tn, fn, f1, tp_adjusted, fp_adjusted, tn_adjusted, fn_adjusted, f1_adjusted

    @staticmethod
    def _show_edges(sim_01, sim_mi, sim_pval):
        cfd = []
        for node_i in sim_01:
            for node_j in sim_01[node_i]:
                if sim_01[node_i][node_j]:
                    cfd.append((node_i, node_j, sim_mi[node_i][node_j], sim_pval[node_i][node_j] ))
        return cfd

