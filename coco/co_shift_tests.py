import itertools

import numpy as np
from numpy.linalg import LinAlgError
from scipy.stats import wasserstein_distance, ks_2samp, stats, norm
import scipy
import scipy.special as scsp
from sklearn.linear_model import LinearRegression

from linc.vsn import Vsn
from utils import data_sub, logg, partition_to_map, pval_to_map
from co_test_type import CoShiftTestType
from gaussian_process_regression import gp_regression, lin_regression
from utils import data_group

from sparse_shift.testing import test_mechanism, test_dag_shifts
from scipy.stats import t
import statsmodels.api as sm
from co_test_type import CoShiftTestType
from linc.pi_mechanism_search import pi_partition_search
from vario.causal_mechanism_search import causal_mechanism_search


def co_shift_test(Dc: list, node_i:int, pa_i: list, shift_test: CoShiftTestType, alpha=0.5):
    ''' Discovers a grouping of contexts

    :param Dc: data
    :param node_i: target node
    :param pa_i: causal parents of target
    :param shift_test: test type
    :return:
    '''

    n_c = Dc.shape[0]
    #n_nodes = Dc.shape[2] - 1

    D_up, pa_up_i = Dc, pa_i

    n_pairs = len([i for i in itertools.combinations(range(n_c), 2)])
    pval_mat = np.ones((n_pairs, n_pairs))
    #if len(pa_i) == 0:
    #    D_up, pa_up_i = _augment(Dc)

    if shift_test.value == CoShiftTestType.SKIP.value:
        map = [0 for _ in range(n_c)]

    elif shift_test.value == CoShiftTestType.VARIO.value:
        pistar, _, _ = causal_mechanism_search(D_up, node_i, pa_up_i, greedy=False)
        map = partition_to_map(pistar)

    elif shift_test.value == CoShiftTestType.VARIO_GREEDY.value:
        pistar, _, _ = causal_mechanism_search(D_up, node_i, pa_up_i, greedy=True)
        map = partition_to_map(pistar)
        
    elif shift_test.value == CoShiftTestType.LINC.value:
        vsn = Vsn(rff=True, ilp=True, clus=False)
        pistar, _ = pi_partition_search(D_up, node_i, pa_up_i, vsn)
        map = partition_to_map(pistar)
        
    elif shift_test.value == CoShiftTestType.PI_KCI.value:
        parents = [1 if node_i in pa_i else 0 for node_i in range(Dc.shape[2])]
        pval_mat = test_mechanism(Dc, node_i, parents, 'kci', {})
        map = pval_to_map(pval_mat, alpha=alpha)
    else:
        raise ValueError()
    return map, pval_mat

def _augment(Dc):
    n = Dc.shape[2]+1
    D = np.random.normal(size=(Dc.shape[0], Dc.shape[1], n ))
    D[:,:,range(n-1)] = Dc
    return D, [n]

#TODO update
def co_pair_grouped(Dc: list, j: int, pa: list, test: CoShiftTestType):
    ''' Tests for each context pair whether it is in the same group or not

    :param Dc: data per context
    :param j: target node
    :param pa: causal parents
    :param test: test type
    :return:
    '''
    contexts = range(len(Dc))

    # Kernel Conditional Independence
    if test == CoShiftTestType.SOFT_KCI or test == CoShiftTestType.PI_KCI:
        parent_ind = [1 if i in pa else 0 for i in range(Dc[0].shape[1])]
        pval_pair_mat = test_mechanism(Dc, j, parent_ind, 'kci')

        # Use p values or indicator vectors
        if not test.is_soft():
            CO = [1 if pval_pair_mat[k][l] <= 0.05 else 0 for ind, k in enumerate(contexts) for l in contexts[ind + 1:]]
        else:
            assert(test == CoShiftTestType.SOFT_KCI)
            CO = [-np.log(pval_pair_mat[k][l]) for ind, k in enumerate(contexts) for l in contexts[ind + 1:]]


    # Gaussian Process Model Comparison
    elif test == CoShiftTestType.PI_GP or test == CoShiftTestType.SOFT_GP:
        Xc, yc = data_sub(Dc, pa), data_sub(Dc, j)
        gains = [co_pair_compression(Xc, yc, k, l, rand_fourier_features=True) for ind, k in enumerate(contexts) for l in contexts[ind + 1:]]

        if not test.is_soft():
            CO = [1 if (2**gains[ind]) <= 0.05 else 0 for ind in range(len(gains))]
        else:
            assert(test == CoShiftTestType.SOFT_GP)
            raise ValueError("Not Implemented") # CO = [-logg(2**(-gains[ind])) for ind in range(len(gains))]

    elif test == CoShiftTestType.PI_LINEAR:
        Xc, yc = data_sub(Dc, pa), data_sub(Dc, j)
        pval_pair_vec = [co_pair_linear(Xc, yc, k, l) for ind, k in enumerate(contexts) for l in contexts[ind + 1:]]
        CO = [1 if pval_pair_vec[ind] <= 0.05 else 0 for ind in range(len(pval_pair_vec))]


    elif test == CoShiftTestType.SOFT_CMMD or test == CoShiftTestType.SOFT_OT or test == CoShiftTestType.PI_OT:
        raise ValueError("Not Implemented")
    else:
        raise ValueError("Unknown CoTestType")
    return CO

def co_pair_linear(Xc, yc, ci, cj):
    Xi, yi = Xc[ci], yc[ci].reshape(-1, 1)
    Xj, yj = Xc[cj], yc[cj].reshape(-1, 1)
    m1 :LinearRegression = lin_regression(Xi, yi)
    m2 = lin_regression(Xj, yj)

    N = Xi.shape[0]
    p = Xi.shape[1]+1

    resid1 = yi - m1.predict(Xi)
    resid2 = yj - m2.predict(Xj)

    rss1 = resid1.T @ resid1
    rss2 = resid2.T @ resid2
    ss_hat1 = rss1[0, 0] / (N - p)
    ss_hat2 = rss2[0, 0] / (N - p)

    Xii, Xjj = np.empty(shape=(N, p), dtype=np.float),np.empty(shape=(N, p), dtype=np.float)
    Xii[:, 0] = 1
    Xii[:, 1:p] = Xi
    Xjj[:, 0] = 1
    Xjj[:, 1:p] = Xj
    try:
        var_beta_hat1 = np.linalg.inv(Xii.T @ Xii) * ss_hat1
    except (LinAlgError):
        var_beta_hat1 = np.zeros((p,p))
    try:
        var_beta_hat2 = np.linalg.inv(Xjj.T @ Xjj) * ss_hat2
    except (LinAlgError):
        var_beta_hat2 = var_beta_hat1

    pvals = np.zeros(p - 1)
    for p_ in range(1, p):
        se1 = var_beta_hat1[p_, p_] ** 0.5
        se2 = var_beta_hat2[p_, p_] ** 0.5
        z_val = ((m1.coef_[0][p_-1] - m2.coef_[0][p_-1]) / np.sqrt(se1 ** 2 + se2 ** 2))  # bse = standard error
        pvals[p_ - 1] = scipy.stats.norm.sf(abs(z_val))
        # print(f"SE(beta_hat[{p_}]): {z_val} ")
    pval = np.prod(pvals)
    return pval

def co_pair_linear2(Xc, yc, ci, cj):
    Xi, yi = Xc[ci], yc[ci].reshape(-1, 1)
    Xj, yj = Xc[cj], yc[cj].reshape(-1, 1)
    reg1:LinearRegression = lin_regression(Xi, yi)
    reg2 = lin_regression(Xj, yj)

    N = Xi.shape[0]
    p = Xi.shape[1]+1
    y_hat_1 = reg1.predict(Xi)
    y_hat_2 = reg2.predict(Xj)
    Xii, Xjj = np.empty(shape=(N, p), dtype=np.float),np.empty(shape=(N, p), dtype=np.float)
    Xii[:, 0] = 1
    Xii[:, 1:p] = Xi
    Xjj[:, 0] = 1
    Xjj[:, 1:p] = Xj

    # beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ yi
    beta_hat1 = np.linalg.inv(Xii.T @ Xii) @ Xii.T @ yi
    beta_hat2 = np.linalg.inv(Xjj.T @ Xjj) @ Xjj.T @ yj

    residuals1 = yi - y_hat_1
    residuals2 = yj - y_hat_2

    rss1 = residuals1.T @ residuals1
    rss2 = residuals2.T @ residuals2
    ss_hat1 = rss1[0, 0] / (N - p)
    ss_hat2 = rss2[0, 0] / (N - p)
    var_beta_hat1 = np.linalg.inv(Xii.T @ Xii) * ss_hat1
    var_beta_hat2 = np.linalg.inv(Xjj.T @ Xjj) * ss_hat2
    pvals = np.zeros(p-1)
    for p_ in range(1, p):
        se1 = var_beta_hat1[p_, p_] ** 0.5
        se2 = var_beta_hat2[p_, p_] ** 0.5
        z_val = ((beta_hat1[p_] - beta_hat2[p_]) / np.sqrt(se1 ** 2 + se2 ** 2))  # bse = standard error
        pvals[p_-1] = scipy.stats.norm.sf(abs(z_val))
        #print(f"SE(beta_hat[{p_}]): {z_val} ")
    pval = np.prod(pvals)
    return pval

def co_pair_ks(Xc, yc, ci, cj):
    #TODO does this make sense?

    Xi, yi = Xc[ci], yc[ci].reshape(-1,1)
    Xj, yj = Xc[cj], yc[cj].reshape(-1,1)
    reg1:LinearRegression = lin_regression(Xi, yi)
    reg2 = lin_regression(Xj, yj)
    alpha1, alpha2 = reg1.coef_, reg2.coef_

    y_hat_1 = reg1.predict(Xi)
    y_hat_2 = reg2.predict(Xj)
    residuals1 = yi - y_hat_1
    residuals2 = yj - y_hat_2



    ks_test = ks_2samp(residuals1.reshape(-1), residuals2.reshape(-1))
    return ks_test.pvalue

def co_pair_linear_old(Xc, yc, ci, cj):

    Xi, yi = Xc[ci], yc[ci].reshape(-1,1)
    Xj, yj = Xc[cj], yc[cj].reshape(-1,1)
    reg1 = lin_regression(Xi, yi)
    reg2 = lin_regression(Xj, yj)
    from scipy.stats import ttest_ind
    alpha1, alpha2 = reg1.coef_, reg2.coef_
    y_hat_1 = reg1.predict(Xi)
    y_hat_2 = reg2.predict(Xj)
    residuals1 = yi - y_hat_1
    residuals2 = yj - y_hat_2
    rss1 = residuals1.T @ residuals1
    rss2 = residuals2.T @ residuals2
    N = Xi.shape[0]
    p = Xi.shape[1]
    sig1 = rss1[0, 0] / (N - p)
    sig2 = rss1[0, 0] / (N - p)
    se1 = np.linalg.inv(Xi.T @ Xi) * sig1 **0.5
    se2 = np.linalg.inv(Xj.T @ Xj) * sig2 **0.5


    z = ((alpha1 - alpha2) /np.sqrt(se1**2 + se2**2))

    def z2p(z):
        return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))
    return z2p(z)

    p = scipy.stats.norm.sf(abs(z))
    t, p = ttest_ind(reg1.coef_, reg2.coef_)

    return p
    n1 = len(yc[ci])
    n2 = len(yc[cj])
    se_diff = np.sqrt(reg1.coef_ ** 2 / np.var(yc[ci]) + reg2.coef_ ** 2 / np.var(yc[cj]))
    t_stat = (reg1.coef_ - reg2.coef_) / se_diff
    df = n1 + n2 - 2
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    return p_value[0]

def co_pair_compression(Xc, yc, ci, cj, rand_fourier_features):
    ''' Discrepancy (MDL Gain) between Conditional Distributions using Gaussian Process Modelling

        :return: f_i =|= f_j where f_i : X_i -> y_i on data in context i
    '''
    Xi, yi = Xc[ci], yc[ci]
    Xj, yj = Xc[cj], yc[cj]

    gp_i = gp_regression(Xi, yi, rand_fourier_features)
    gp_j = gp_regression(Xj, yj, rand_fourier_features)

    mdl_j, d_j, m_j, _ = gp_j.mdl_score_ytrain()
    mdl_i, d_i, m_i, _ = gp_i.mdl_score_ytrain()

    #Xij, yij = data_group(Xc, yc, partition=[[ci, cj]])
    #Xij, yij = Xij[0], yij[0]

    mdl_i_joint, d_i_joint, m_i_joint, _ = gp_i.mdl_score_ytest(Xj, yj)
    mdl_j_joint,  d_j_joint,  m_j_joint, _ = gp_j.mdl_score_ytest(Xi, yi)
    gain = min(mdl_i - mdl_j_joint, mdl_j - mdl_i_joint) # +mdl_j-mdl_j cancel out

    return gain


def co_pair_opttransport(Xc, yc, ci, cj, rand_fourier_features):
    ''' Discrepancy (Wasserstein Distance) between Conditional Distributions using Gaussian Process Modelling

        :return: f_i =|= f_j where f_i : X_i -> y_i on data in context i
    '''
    Xi, yi = Xc[ci], yc[ci]
    Xj, yj = Xc[cj], yc[cj]

    gp_i = gp_regression(Xi, yi, rand_fourier_features)
    gp_j = gp_regression(Xj, yj, rand_fourier_features)

    GPmean_back_i = gp_i.predict(Xi)
    GPmean_back_j = gp_j.predict(Xi)
    GPresid_back_i = GPmean_back_i - yi
    GPresid_back_j = GPmean_back_j -yi

    # forward: i to j vs. j to j
    GPmean_forward_i =  gp_i.predict(Xj)
    GPmean_forward_j = gp_j.predict(Xj)
    GPresid_forward_i = (GPmean_forward_i - yj)
    GPresid_forward_j = (GPmean_forward_j - yj)

    ot_self = max(sum(GPresid_forward_j ** 2), sum(GPresid_back_i ** 2))
    ot_cross = max(sum(GPresid_forward_i ** 2), sum(GPresid_back_j ** 2))

    ot_pair= max(opt_transport(GPresid_forward_i, GPresid_forward_j), #TODO **2?
                                    opt_transport(GPresid_back_j, GPresid_back_i))
    return ot_pair

def opt_transport(resid_i, resid_j):
    #TODO check out Wass for GP paper
    return wasserstein_distance(resid_i, resid_j)