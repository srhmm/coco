from vario.linear_projection import LinearProjection
from vario.utils_confidence import conf_intervals_gaussian, confident_score


class PartitionRecord:
    """ Stores each multi-edge: {X1, ... Xn} -> Y and a context partition."""

    def __init__(self, data_each_context):
        """ Initializes PartitionRecord, which depends on one input dataset (with data from multiple contexts)

        :param data_each_context: list [np array over all variables]
        :return: PartitionRecord
        """
        self.data_each_context = data_each_context
        self.all_nodes = data_each_context[0].shape[1]
        self.covariates = self.all_nodes - 1
        self.CUTOFF = 3
        """ top x partitions considered for each variable """
        self.DEBUG = False
        """unless DEBUG=True, intermediate results are deleted when they arent needed anymore"""
        self.causal_edges = {}
        """final causal parents and partition and weighted scores (for each considered target Y)"""
        self._causal_edges={}
        conf_intervals, _ = conf_intervals_gaussian(n_contexts=len(data_each_context))
        self.conf_intervals = conf_intervals
        """confidence intervals which can be used to judge whether the score for a partition is significant"""
        self._linmap_cache = {}
        """regression results.  For all subsets of variables X, we have a function f: {X1, ... Xn} -> Y. Regression info about f is in a class of type LinearProjection."""
        self._partition_cache = {}
        """partitions seen during search. For all subsets of variables X, e.g. for {X1, ... Xn} -> Y, we store context partitions such as [[0,1],[2,3,4]]."""

        # Intermediate results during search are stored in the following attributes:
        self._candidate_edges = {}
        """scores for single variables and a partition, e.g. for X1 we score how well the partition [[0,1],[2,3,4]] explains a target Y"""
        self._candidate_multiedges = {}
        """ scores for set of variables and a partition, e.g. for {X1, ... Xn} and [[0,1],[2,3,4]]"""
        #weighted scores
        self._admissible_edges = {}
        self._admissible_partitions = {}
        self._subsets_seen = {}
        self._subset_num = {}

    def eval(self, index_Y, indices_X, partition):
        """ During causal mechanism search, evaluate a child-parent-partition combination,
        e.g. {X1, ... Xn} -> Y : [[0,1],[2,3,4]].

        :param index_Y: target/child
        :param indices_X: covariates/parents
        :param partition: candidate context partition
        :return: score
        """
        linmap = self.eval_regression_function(index_Y, indices_X)
        score = self._eval_multiedge(index_Y, indices_X, partition, linmap)

        return score

    def eval_regression_function(self, index_Y, indices_X):
        """ Evaluate a child-parentset combination

        :param index_Y: target/child
        :param indices_X: covariates/parents
        :return: LinearProjection with linear parameters of f:X->Y
        """

        hash_XY = f'f:X_{str(indices_X)}->Y_{str(index_Y)}'
        if self._linmap_cache.__contains__(hash_XY):
            linmap = self._linmap_cache[hash_XY]
            assert self._subset_num.__contains__(f'Y_{index_Y}')
        else:
            linmap = LinearProjection(self.data_each_context, index_Y, indices_X)
            self._linmap_cache[hash_XY] = linmap
            if not self._subset_num.__contains__(f'Y_{index_Y}'):
                self._subset_num[f'Y_{index_Y}'] = 1
            else:
                self._subset_num[f'Y_{index_Y}'] = self._subset_num[f'Y_{index_Y}'] + 1

        return linmap

    def _eval_multiedge(self, index_Y, indices_X, partition, linmap):
        """ Evaluate a child-parentset-partition combination

        :param index_Y: target/child
        :param indices_X: covariates/parents
        :param partition: candidate context partition
        :param linmap: LinearProjection with linear parameters of f:X->Y
        :return:
        """

        hash_multiedge = f'f:X_{str(indices_X)}->Y_{str(index_Y)}:pi_{str(partition)}'
        hash_XY = f'f:X_{str(indices_X)}->Y_{str(index_Y)}'

        if self._candidate_multiedges.__contains__(hash_multiedge):
            return self._candidate_multiedges[hash_multiedge]


        score = linmap.score(partition)
        if not self._partition_cache.__contains__(hash_XY):
            self._partition_cache[hash_XY] = {}
        self._partition_cache[hash_XY][str(partition)] = (score, partition)
        self._candidate_multiedges[hash_multiedge] = score

        for index_X in indices_X:
            hash_pair = f'pair:X_{str(index_X)},Y_{str(index_Y)}'
            if not self._candidate_edges.__contains__(hash_pair):
                self._candidate_edges[hash_pair] = {str(partition): (partition, score, 1, index_X, index_Y)}
            else:
                if not self._candidate_edges[hash_pair].__contains__(str(partition)):
                    self._candidate_edges[hash_pair][str(partition)] = (partition, score, 1, index_X, index_Y)
                pi, score_sum, ct, ix, iy = self._candidate_edges[hash_pair][str(partition)]
                assert (ix == index_X and iy == index_Y)
                self._candidate_edges[hash_pair][str(partition)] = (pi, score_sum + score, ct + 1, index_X, index_Y)

            # DEBUG
            # if not self._subsets_seen.__contains__(str(partition)):
            #     self._subsets_seen[str(partition)] = {str(index_X)  : 1}
            # else:
            #     if not self._subsets_seen[str(partition)].__contains__(str(index_X)):
            #         self._subsets_seen[str(partition)][str(index_X)]  = 1
            #     else:
            #         self._subsets_seen[str(partition)][str(index_X)]  =  self._subsets_seen[str(partition)][str(index_X)] + 1

        return score

    def _sorted_partitions(self, index_Y, indices_X):
        # returns all partitions found for X and Y during search, although not in an efficient way.
        hash_XY = f'f:X_{str(indices_X)}->Y_{str(index_Y)}'
        if not self._partition_cache.__contains__(hash_XY):
            return {}
        else:
            return sorted(self._partition_cache[hash_XY].items(), key=lambda item: item[1][0], reverse=False)

    def process_target(self, iy, cut=True):
        """ Call after causal variable search for one target is complete to process the results.

        :return:
        """
        self._weight_edges(iy, cut)
        self._discover_causal_edges(iy)
        # Remove all intermediate results for the target as these are not needed anymore (except for debug)
        if not (self.DEBUG):
            self.clear_target()


    def _weight_edges(self, target, cut):
        """ When called, we have evaluated {X1,...Xn}->Y for a fixed target Y and all subsets of parents X, and discovered candidate partitions and scores for them.

            Here we now traverse all variables Xi. For the candidate partitions, we find an average score telling us how "good" the partition is for Xi and Y.

            "Good" means: 1. the partition occurs in many subsets containing Xi, 2. the partition has a good score in most subsets.
            To balance these two goals we use weighted sorting.

        :return:
        """
        for hash_pair in self._candidate_edges:
            if not self._admissible_edges.__contains__(hash_pair):
                self._admissible_edges[hash_pair] = {}
            max_ct = 0
            avg_score = 0
            num_score = 0

            # Weighted sort: For each variable Xi, sort the candidate partitions for Xi and Y as follows:
            # - score for sorting: average score that the partition has in all supersets {X1, .. Xi, ..Xn} -> Y, compared to other scores
            # - weight for sorting: number of occurences of the partition in supersets of Xi, compared to other partitions
            #  (for example, if a partition occurs with a high score in many subsets, good; if it occurs with a high score but only in one subset, bad)

            for partition_str in self._candidate_edges[hash_pair]:
                _, score_sum, ct, _, iy = self._candidate_edges[hash_pair][partition_str]
                if not (iy == target): #only consider edges to Y
                    continue
                avg_score = avg_score + score_sum / ct
                num_score = num_score + 1
                if ct > max_ct:
                    max_ct = ct
            avg_score = avg_score / num_score

            for partition_str in self._candidate_edges[hash_pair]:
                pi, score_sum, ct, ix, iy = self._candidate_edges[hash_pair][partition_str]
                score = (ct / max_ct * (score_sum / ct)) + ((1 - ct / max_ct) * avg_score)
                self._admissible_edges[hash_pair][partition_str] = (pi, score, ct, ix, iy) #(score, score_sum / ct, pi, ct)



            self._admissible_edges[hash_pair] = sorted(self._admissible_edges[hash_pair].items(),
                                                      key=lambda item: item[1][1], reverse=True)
            if cut:
                if len(self._admissible_edges[hash_pair]) > self.CUTOFF:
                    self._admissible_edges[hash_pair] = self._admissible_edges[hash_pair][0:self.CUTOFF-1]

            # Store such that each partition points to score and X, Y
            for partition_str, val in self._admissible_edges[hash_pair]:
                pi, score, ct, ix, iy = val
                if not self._admissible_partitions.__contains__(partition_str):
                    self._admissible_partitions[partition_str] = {hash_pair: (pi, score, ix, iy)}
                else:
                    self._admissible_partitions[partition_str][hash_pair] = (pi, score, ix, iy)

    def _discover_causal_edges(self, target):
        """When called, we evaluated each pair (Xi,Y) and collected candidate partitions (and scores) for them.

            Here we now aggregate these results to find a single best partition and parent set {X1, ... Xn} that explain Y best.
        """
        for partition_str in self._admissible_partitions:
            for hash_pair in self._admissible_partitions[partition_str]:
                pi, score, ix, iy = self._admissible_partitions[partition_str][hash_pair]
                if not (iy == target):
                    continue
                hashY = f'Y_{iy}'
                is_significant = confident_score(self.conf_intervals, pi, score, len(self.data_each_context))

                if not is_significant:
                    continue
                if not self._causal_edges.__contains__(hashY):
                    self._causal_edges[hashY] = {str(pi): (pi, score, 1, [ix], iy)}
                else:
                    if not self._causal_edges[hashY].__contains__(str(pi)):
                        self._causal_edges[hashY][str(pi)] = (pi, score, 1, [ix], iy)
                    else:
                        _, cur_score, cur_ct, parents, iyy = self._causal_edges[hashY][str(pi)]
                        assert (iyy == iy)
                        self._causal_edges[hashY][str(pi)] = (pi, cur_score + score, cur_ct + 1, [p for p in range(self.all_nodes) if p in parents or p==ix], iy)

        # Aggregate the scores and keep the best causal edge and partition for Y
        for hashY in self._causal_edges:

            for partition_str in self._causal_edges[hashY]:
                pi, score, ct, ix, iy = self._causal_edges[hashY][partition_str]
                if not (iy == target):
                    continue
                self._causal_edges[hashY][partition_str] = pi, score / ct, ct, ix, iy
            self._causal_edges[hashY] = sorted(self._causal_edges[hashY].items(),
                                                      key=lambda item: item[1][1], reverse=True)
            #best edge and partition:
            pi, score, ct, pa, iy = self._causal_edges[hashY][0][1]
            # Reject the edges if the partition has no invariance:
            n_contexts = len(self.data_each_context)
            no_invariance = (len(pi)==n_contexts)
            if no_invariance:
                pa = []
            self.causal_edges[hashY] = pi, score, pa, iy


    def clear_target(self):
        self._partition_cache = {}
        self._candidate_edges = {}
        self._admissible_edges = {}
        self._admissible_partitions = {}
        self._candidate_multiedges = {}
        self._causal_edges = {}

    def get_causal_variables(self, iy):
        _, score, parents, _ = self.causal_edges[f'Y_{iy}']
        return parents, score

    def get_partition(self, iy):
        pi, score, _, _ = self.causal_edges[f'Y_{iy}']
        return pi, score
