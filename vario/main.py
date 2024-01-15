from vario.causal_mechanism_search import causal_mechanism_search
from vario.causal_variable_search import causal_variable_search
from vario.causal_dag_search import causal_dag_search
from vario.generate_data import generate_data_from_DAG
from vario.utils_eval import eval_causal_edges, eval_partition

if __name__ == '__main__':
    # Example
    n_nodes = 5
    n_contexts = 5
    n_samples = 1000
    n_context_groups = 3
    true_partition = [[0, 1, 2, 3],[4]]

    # Generating data
    data, index_Y, true_dag, true_partitions = generate_data_from_DAG(n_nodes, n_contexts, n_samples,
                                                                      partition_Y=true_partition,
                                                                      min_interventions=5,max_interventions=5,
                                                                      scale=False, verbose=True, seed=3)

    # Alg. VARIO_PI: Discovering mechanism changes for a single variable Y
    # - Oracle case: the causal parents X_S of Y are known
    # - we want to discover mechanism changes of the function f:X_S->Y
    indices_X = true_dag.parents_of(index_Y)
    estim_partition, estim_score, _ = causal_mechanism_search(data, index_Y, indices_X, greedy=False, verbose=True)
    eval_partition(estim_partition, true_partition)
    greedy_partition, greedy_score, _ = causal_mechanism_search(data, index_Y, indices_X, greedy=True, verbose=True)
    eval_partition(greedy_partition, true_partition)

    # Main Alg. VARIO: Discovering causal variables and mechanism changes for Y
    # - Case: causal parents of variable Y is unknown
    # - we want to discover the best set X_S and the mechanism changes of f:X_S->Y
    indices_X = [n for n in range(n_nodes) if not (n==index_Y)]
    causal_parents, causal_partition, score, _ = causal_variable_search(data, index_Y, indices_X,
                                                                        greedy_mechanism_search=False, verbose=True)


    # Extended Alg. VARIO_G: Discovering causal edges for all variables
    # - Case:  causal parents of all variables unknown
    causal_edges = causal_dag_search(data, greedy_mechanism_search=False, verbose=True)


    # Evaluation
    eval_causal_edges(causal_edges, true_partitions, true_dag)



