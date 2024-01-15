from typing import List

from vario.causal_variable_search import subset_search
from vario.partition_record import PartitionRecord


def causal_dag_search(data_each_context:List, greedy_mechanism_search=True, verbose=False):
    """ Discovers the causal variables and mechanism changes for all variables.

    :param data_each_context
    :return: causal edges and partitions for all targets
    """
    partition_record = PartitionRecord(data_each_context)
    all_nodes = range(data_each_context[0].shape[1])
    if verbose:
        print("\n--- Causal DAG search ---")
    for iy in all_nodes:
        ix = [x for x in all_nodes if not (x==iy)]
        partition_record = subset_search(partition_record, iy, ix, greedy_mechanism_search, verbose=False)
        partition_record.process_target(iy)

    if verbose:
        print("Mechanism changes discovered:")
        for iy in all_nodes:
            partition, score = partition_record.get_partition(iy)
            print("\t ", "Node", iy, ":", partition, round(score,2))
        print("Causal edges discovered:")
        for iy in all_nodes:
            parents, _ = partition_record.get_causal_variables(iy)
            if len(parents):
                print("\t ", "Node", iy, ":",  ((" -> "+str(iy)+", ").join([str(p) for p in parents])), "->", iy)



        #print("Estimated parents, partition, score:\n{", ','.join([str(x) for x in parents]), "}",  partition, round(score,2))
    return partition_record.causal_edges
