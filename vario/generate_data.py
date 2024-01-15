import numpy as np
from causaldag import rand
from typing import List


from vario.linear_dag import LinearDAG
from vario.linear_regression import linear_regression
from vario.utils import cantor_pairing, scale_data
from vario.utils_arc_weights import gen_arc_weights_pi
from vario.utils_context_partition import random_context_partition, partition_to_groupmap


def generate_DAG(n_nodes, seed, node_density=.5):
    np.random.seed(seed)
    dag = rand.directed_erdos(nnodes=n_nodes, density=node_density)
    dag = rand.rand_weights(dag)
    return dag

def generate_data_from_DAG(n_nodes:int,n_contexts:int,
                           n_samples_per_context:int,
                           partition_Y:List[List[int]],
                           min_interventions:int,
                           max_interventions:int,
                           scale: bool,verbose: bool, seed=42):
    """
    Generates data in n contexts.

    :param n_nodes: variables, target Y and covariates
    :param n_contexts: contexts
    :param n_samples_per_context: samples in each context
    :param partition_Y: shows mechanism changes for target Y, e.g. [[0,1,2,3],[4]] means a change in the last context
    :param min_interventions: min number of mechanism changes on covariates
    :param max_interventions: max number of mechanism changes on covariates
    :param scale: whether to scale the data
    :param verbose: whether to print linear parameters
    :param seed: seed
    :return: data_each_context,
           target, dag, partitions_each_node
    """

    dag = generate_DAG(n_nodes, seed)
    rst = np.random.RandomState(seed)

    # Choose node for the target variable Y in the graph. If possible, this node has parents
    nodes_with_parents = [n for n in range(n_nodes) if n not in dag.sources()]
    if len(nodes_with_parents) == 0:
        nodes_with_parents = [n for n in range(n_nodes)] # choose an arbitrary Y

    target = nodes_with_parents[rst.randint(low=0, high=len(nodes_with_parents))]
    parents = dag.parents_of(target)
    children = dag.children_of(target)

    data_each_context = [None for _ in range(n_contexts)]
    dag_each_context = [None for _ in range(n_contexts)]


    partitions_X, _, arc_weights_X = initialize_DAG_nodes(n_contexts, n_nodes, min_interventions, max_interventions,
                        dag, target, partition_Y, seed)#for each variable X: chooses a random partition and generates edge weights

    # DAG and data per context ----------
    for c_i in range(n_contexts):
        lin_dag_c = LinearDAG(dag.nodes, dag.arcs)

        for node_X in dag.nodes:
            pi_X = partitions_X[node_X]
            pi_map = partition_to_groupmap(pi_X, n_contexts)
            pi_k = pi_map[c_i]

            # arc weight in the group of context i
            weights_to_p = arc_weights_X[node_X][pi_k]
            for (i, j) in dag.arcs:
                if j == node_X:
                    w_ij = weights_to_p[(i, j)]
                    lin_dag_c.set_arc_weight(i, j, w_ij)

        # Sample data in context c
        data_c = lin_dag_c.sample_linear(nsamples=n_samples_per_context)
        if scale:
            data_c = scale_data(data_c)
        data_each_context[c_i] = data_c
        dag_each_context[c_i] = lin_dag_c


    if verbose:
        print("\n--- Generating a DAG --- ")
        print("Target Y:", target)
        print("Partition:", partitions_X[target])
        print("Causal parents:", parents)
        print("Causal children:", children)
        print("Mechanism changes:")
        for node in dag.nodes:
            print("\t", "Node", node, ":", partitions_X[node])
        print("Edge, edge weights per context:")
        for node in dag.nodes:
            #if not p in parents and not p in children:
            #    continue

            for (i,j) in dag.arcs:
                if node==j:
                    #if #j == target and i == node or j == node:
                    w = [0 for _ in range(n_contexts)]
                    for c_i in range(n_contexts):
                        if (i, j) in dag_each_context[c_i].arc_weights:
                            w[c_i] = round(dag_each_context[c_i].arc_weights[(i, j)],2)

                    data_X = np.array(
                        [data_each_context[c_i][:, [i]] for c_i
                         in range(n_contexts)])
                    data_Y = np.array([data_each_context[c_i][:, j] for c_i in range(n_contexts)])
                    estim_w = linear_regression(data_X, data_Y, scale=False)

                    print("\t", i, "->", j, "\t", w, "\t", [round(w[0],2) for w in estim_w]) #, "\t", partitions_X[j])
    partitions_each_node = partitions_X

    return data_each_context, target, dag, partitions_each_node





def initialize_DAG_nodes(n_contexts, n_nodes, min_interventions, max_interventions,
                        dag, target, partition_Y, seed):


    #for each variable X: context partition, observational group, arc weights of groups ----------
    partitions_X = [None for _ in range(n_nodes)]
    observational_X = [0 for _ in range(n_nodes)]
    arc_weights_X = [None for _ in range(n_nodes)]
    iv_in_groups=True
    iid_contexts=False
    context_0_observational=False
    for node_X in dag.nodes:
        partition_X = random_context_partition(n_contexts, np.random.RandomState(cantor_pairing(node_X, seed)),  # the same for all c_i
                              permute=True,
                              k_min=min_interventions, k_max=max_interventions,
                              single_context_groups=not iv_in_groups)
        if iid_contexts:
            assert len(partition_X) == 1
        else:
            if min_interventions is not None:
                assert min_interventions <= len(partition_X) <= max_interventions

        if node_X == target:
            partition_X = partition_Y
        partitions_X[node_X] = partition_X

        if iv_in_groups:
            observational_X[node_X] = np.random.RandomState(cantor_pairing(node_X, seed)).choice(
                range(len(partition_X)))
        else:
            if len(partition_X) > 1:
                for interventional_group in range(len(partition_X) - 1):
                    assert len(partition_X[interventional_group]) == 1
            observational_X[node_X] = len(partition_X) - 1
        if context_0_observational:
            observational_X[node_X] = 0
        assert 0 <= observational_X[node_X] < len(partition_X)

        arc_weights_X[node_X] = gen_arc_weights_pi(partition_X, dag, seed, False)
    return partitions_X, observational_X, arc_weights_X

