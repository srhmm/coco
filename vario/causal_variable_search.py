import more_itertools as mit
from typing import List

from vario.causal_mechanism_search import causal_mechanism_search
from vario.partition_record import PartitionRecord


def causal_variable_search(data_each_context: List, iy: int, ix: List, greedy_mechanism_search=True, verbose=False):
    """ Discovers the causal variables (and mechanism changes) for a target variable Y.

    :param data_each_context
    :param iy: target Y
    :param ix: covariates X
    :return: causal parents, partition
    """
    partition_record = PartitionRecord(data_each_context)
    partition_record = subset_search(partition_record, iy, ix, greedy_mechanism_search, verbose)
    partition_record.process_target(iy)

    partition, score = partition_record.get_partition(iy)
    parents, _ = partition_record.get_causal_variables(iy)

    print("Estimated parents, partition, score:\n{", ','.join([str(x) for x in parents]), "}",  partition, round(score,2))
    return parents, partition, score, partition_record


def subset_search(partition_record, iy, ix, greedy, verbose):
    """  Evaluates each subset {X1, ... Xn} of covariates X of a target Y.
    For each, it discovers mechanism changes of the function f : {X1, ... Xn} -> Y, i.e. a context partition.

    :param partition_record: PartitionRecord
    :param iy: target Y
    :param ix: covariates X
    :return: partition_record
    """
    if verbose:
        print("\n--- Subset Search for Causal Variables ---")
        print("Target: Y =", iy)
    subsets = mit.powerset(ix)
    for subset in subsets:
        if len(subset) == 0:
            continue
        candidate_parents = [s for s in subset]
        _, _, partition_record = causal_mechanism_search(partition_record.data_each_context, iy, candidate_parents,
                                                         greedy=greedy, partition_record=partition_record)

    return partition_record