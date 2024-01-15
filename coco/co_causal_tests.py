import numpy as np#
import scipy

from mi import mutual_info_scores

def test_causal(h1, h2, mi, stdev_cent_sampling, t=1.96):

    if stdev_cent_sampling == 0:
        z = h1 - mi
    else:
        z = (h1 - mi) / stdev_cent_sampling  # TODO emi or mean_emi_sampling?

    pval = scipy.stats.norm.sf(abs(z))

    causal = z < t and z > -t# (z > t or z < -t)
    return causal, pval


def test_causal_(map_i, map_j, sampler):
    N = len(map_i)
    assert (len(map_j) == N)
    n_shifts_i, n_shifts_j = len(np.unique(map_i)) - 1, len(np.unique(map_j)) - 1

    mi, ami, emi, h1, h2 = mutual_info_scores(map_i, map_j)

    top_mi_sampling, _, _, _, _, mean_emi_sampling, \
    var_mi_sampling, var_ami_sampling, stdev_mi_sampling, stdev_ami_sampling,\
    stdev_cent_sampling, stdev_vi_sampling = sampler.sample(N, n_shifts_i, n_shifts_j)
    return test_causal(h1, h2, mi, stdev_cent_sampling)



