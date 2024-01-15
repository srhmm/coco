import itertools
import time

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



def linear_regression(X, y, scale=True):
    """Given continuois data X,y in several contexts or groups, linear regression per group

    :param X: X, list of np.arrays
    :param y: y, continuous
    :return: regression coefficient per context/group, for each, pvalues of invariance tests for pairs of contexts
    """
    #coef_k_x = np.array([LinearRegression().fit(X[pi_k], y[pi_k]).coef_
    #                  for pi_k in range(len(X))])
    k = len(X)
    coef_k_x = [_ for _ in range(k)]
    std_k_x = [_ for _ in range(k)]
    resid_bias = [_ for _ in range(k)]
    resid_std = [_ for _ in range(k)]
    #pvals_k_x = [_ for _ in range(k)]
    #invariant_k_x = [_ for _ in range(k)]

    for pi_k in range(len(X)):
        coef_k, std_k, resid_k, resid_std_k = linear_coef_std(X[pi_k], y[pi_k])

        coef_k_x[pi_k] = coef_k
        std_k_x[pi_k] = std_k
        resid_bias[pi_k] = resid_k
        resid_std[pi_k] = resid_std_k

    coef_scaled_k_x = coef_rescale(np.array(coef_k_x), len(X),
                                   X_n=len(coef_k_x[0]))  # X_n=len(LinearRegression().fit(X[0], y[0]).coef_))
    resid_scaled = coef_rescale(np.array(resid_bias), 1, 1)  # X_n=len(LinearRegression().fit(X[0], y[0]).coef_))

    #print()
    #invariant_k, pvals_k = linear_test_invariance(coef_k, std_k)

    #pvals_k_x[pi_k] = pvals_k
    #invariant_k_x[pi_k] = pvals_k

    return np.array(coef_k_x), np.array(std_k_x), coef_scaled_k_x, resid_bias, resid_scaled, resid_std

def linear_test_invariance(coef_k, stdv_k):
    N = len([_ for _ in itertools.combinations(range(len(coef_k)), 2)])
    pval = [_ for _ in range(N)]
    invariant = [_ for _ in range(N)]

    for l, (i, j) in enumerate(itertools.combinations(range(len(coef_k)), 2)):
        invariant[l], pval[l] = linear_test_invariance_pair(coef_k[i], stdv_k[i], coef_k[j], stdv_k[j])
    return invariant, pval

def linear_test_invariance_pair(coef_i, stdv_i, coef_j, stdv_j):
    den = np.sqrt(stdv_i**2 + stdv_j**2)
    if den == 0:
        z = coef_i - coef_j
    else:
        z = (coef_i - coef_j) / den
    t = 1.96
    pval = scipy.stats.norm.sf(abs(z))
    invariant = (z < t and z > -t)

    return invariant, pval


def linear_coef_std(X_k, y_k):
    model = LinearRegression().fit(X_k, y_k)
    N, p = X_k.shape[0], X_k.shape[1] + 1
    X_incpt = np.empty(shape=(N, p), dtype=np.float)
    X_incpt[:, 0] = 1
    X_incpt[:, 1:p] = X_k

    y_hat = model.predict(X_k)
    residuals = y_k - y_hat

    Xlin = np.linspace(0, 100, N)
    resid_bias = LinearRegression().fit(np.expand_dims(Xlin, 1), residuals).intercept_


    residual_sum_of_squares = residuals.T @ residuals
    sigma_squared_hat = residual_sum_of_squares / (N - p)
    var_beta_hat = np.linalg.inv(X_incpt.T @ X_incpt) * sigma_squared_hat
    std = []
    for p_ in range(1, p):
        std.append(var_beta_hat[p_, p_] ** 0.5)
    resid_std = var_beta_hat[0, 0] ** 0.5
    return model.coef_, std, resid_bias, resid_std

def coef_rescale(coef_c_x, C_n, X_n):
    if X_n == 1:
        coef_c_x = coef_c_x.reshape(-1,1)
    else:
        c_n = coef_c_x.shape[0]
        x_n = coef_c_x.shape[1]
        assert x_n==X_n and  c_n == C_n

    coef_c_x = np.transpose([dim_rescale(coef_c_x[:, x_i]) for x_i in range(X_n)])
    return coef_c_x

def dim_rescale(coef_c):
    offset = min(coef_c)
    coef_c = [coef_c[i] - offset for i in range(len(coef_c))]
    if (max(coef_c)==0): return([1 for i in range(len(coef_c))])

    scale = 100/max(coef_c)
    return [coef_c[i]* scale +1 for i in range(len(coef_c))]


