# Based on sklearn.metrics.cluster, modifies: expected_mutual_information_fast,
# adds: variance_mutual_information, mutual_info_zscore


from scipy.special import gammaln
import numpy as np
from math import exp, lgamma #TODO check against  from libc.math cimport exp, lgamma
#cimport numpy as cnp
from sklearn.metrics.cluster._supervised import check_clusterings, contingency_matrix, mutual_info_score, entropy

'''
def expected_mutual_information_confounded(contingency, n_samples):
    emi, e2mi = 0, 0
    n_rows, n_cols = contingency.shape
    a = np.ravel(contingency.sum(axis=1).astype(np.int64, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int64, copy=False))
    a_view = a
    b_view = b

    # any labelling with zero entropy implies EMI = 0
    if a.size == 1 or b.size == 1:
        return 0.0, 0.0

    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    nijs[0] = 1  # Stops divide by zero warnings. As its not used, no issue.
    # term1 is nij / N
    term1 = nijs / n_samples
    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    log_a = np.log(a)
    log_b = np.log(b)
    # term2 uses log(N * nij) = log(N) + log(nij)
    log_Nnij = np.log(n_samples) + np.log(nijs)
    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    gln_Na = gammaln(n_samples - a + 1)
    gln_Nb = gammaln(n_samples - b + 1)
    gln_Nnij = gammaln(nijs + 1) + gammaln(n_samples + 1)

    # emi itself is a summation over the various values.
    for i in range(n_rows):
        for j in range(n_cols):
            start = max(1, a_view[i] - n_samples + b_view[j])
            end = min(a_view[i], b_view[j]) + 1
            for nij in range(start, end):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                       - gln_Nnij[nij] - lgamma(a_view[i] - nij + 1)
                       - lgamma(b_view[j] - nij + 1)
                       - lgamma(n_samples - a_view[i] - b_view[j] + nij + 1))
                term3 = exp(gln)
                emi += (term1[nij] * term2 * term3)
                e2mi += (term1[nij] * term2)**2 * term3
    return emi, e2mi



'''