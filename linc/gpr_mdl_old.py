# import numpy as np
# import scipy
# from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process._gpr import GPR_CHOLESKY_LOWER
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.utils import check_random_state
# from sklearn.base import BaseEstimator, RegressorMixin,
# from sklearn.utils.optimize import _check_optimize_result

import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy.optimize

from sklearn.base import BaseEstimator, RegressorMixin, clone, MultiOutputMixin
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_random_state
from sklearn.utils.optimize import _check_optimize_result

GPR_CHOLESKY_LOWER = True

"""Gaussian processes regression with regularization """


# Inherits from GaussianProcessRegressor (sklearn)
# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by: Pete Green <p.l.green@liverpool.ac.uk>
# License: BSD 3 clause

class GaussianProcessRegularizedOld(GaussianProcessRegressor):


    # L = cholesky(K + sigma^2 I)
    # alpha = L^T \ (L \ y)
    # mdl_penalty = alpha^T K alpha
    # mdl_score = - log lik(y) + mdl_penalty
    def _mdl_score(self, X_test, y_test):
        kernel = self.kernel_
        kernel.theta = self.kernel_.theta
        K, K_grad = kernel(X_test, eval_gradient=True)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError: return -np.inf


        if y_test.ndim == 1:
            y_test = y_test[:, np.newaxis]
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_test, check_finite=False)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_test, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        #K_train = self.kernel_(self.X_train_, eval_gradient=False)
        #mdl_penalty = self.mdl_model_train # = np.transpose(self.alpha_) @ K_train @ self.alpha_

        #inner = np.einsum("ik,jk->ijk", alpha, alpha)
        #grad = np.einsum( "ijl,jik->kl", inner, K_grad)
        mdl_score = -log_likelihood + self.mdl_model_train

        return mdl_score, -log_likelihood, self.mdl_model_train

    def fit(self, X, y):
        #print("GPR")
        super(GaussianProcessRegularized, self).fit(X,y)

        kernel = self.kernel_
        K = kernel(self.X_train_, eval_gradient=False)

        sigma = 1
        mdl_pen_train = 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**2 * K))
        if mdl_pen_train == np.inf:
            mdl_pen_train = 0


        mdl_lik_train = -self.log_marginal_likelihood_value_
        mdl_model_train = self.alpha_.T @ K @ self.alpha_

        mdl_train  = mdl_lik_train + mdl_model_train + mdl_pen_train

        self.mdl_lik_train = mdl_lik_train
        self.mdl_model_train = mdl_model_train
        self.mdl_pen_train = mdl_pen_train
        self.mdl_train = mdl_train

    def mdl_score_ytrain(self):
        return self.mdl_train, self.mdl_lik_train, self.mdl_model_train, self.mdl_pen_train

    def mdl_score_ytest(self, X_test, y_test):
        sigma = 1
        kernel = self.kernel_
        K = kernel(X_test, eval_gradient=False)
        X_penalty = 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**2 * K))
        if X_penalty == np.inf:
            X_penalty = 0

        mdl, log_lik, m_penalty = self._mdl_score(X_test, y_test)
        mdl = mdl + X_penalty
        return mdl, log_lik, m_penalty, X_penalty
