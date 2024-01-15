import numpy as np
import scipy.stats

from vario.linear_projection import LinearProjection
from vario.utils_context_partition import random_context_partition, pi_valid


def conf_intervals_gaussian(n_contexts, n_samples = 1000, reps = 100):
    """ Returns, for each number of groups k, a confidence interval of the scores that we likely obtained at random for a partition with that number of groups (assuming scores Gaussian distributed, and "null data" is Gaussian distributed)

    :param n_contexts:
    :return:
    """
    scores = [[None for _ in range(reps)] for _ in range(n_contexts-1)] # k=2,...k=n_contexts.
    for i in range(reps):
        sample_0 = np.array([np.random.normal(size=(n_samples,2)) for _ in range(n_contexts)])
        linmap_0 = LinearProjection(sample_0, 0, [1])
        for k in range(2, n_contexts + 1):
            partition_k_groups = random_context_partition(n_contexts, permute=False, k_min=k, k_max=k)
            scores[(k-2)][i] = linmap_0.score(partition_k_groups)
    description =  ["(mean mu, mu+conf, mu-conf) for K="+str(k+2) for k in range(n_contexts-1)] # k=2,...k=n_contexts.
    return [mean_confidence_interval(scores_k) for scores_k in scores], description


def confident_score(conf_intervals, partition, score, n_contexts):
    k = len(partition)
    if k == 1:
        return False  # as our score is  a "gain" over the partition that has one group, the partition with one group is insignificant by design.
    indx = k - 2
    pi_valid(partition, n_contexts)
    assert 0 <= indx <= n_contexts - 1
    low_thresh = conf_intervals[indx][1]
    up_thresh = conf_intervals[indx][2]
    return score <= low_thresh or score >= up_thresh


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h