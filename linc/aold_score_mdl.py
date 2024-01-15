import math
from functools import lru_cache
from typing import List, Dict, Any, Callable

import pandas as pd
from causallearn.score.LocalScoreFunction import local_score_BDeu, local_score_BIC, local_score_cv_multi, local_score_marginal_multi, local_score_marginal_general, local_score_cv_general

from causallearn.utils.ScoreUtils import *
import numpy as np
from pi import PI
from pi_search_ilp import pi_search_ILP
from pi_search_mdl import pi_mdl
from utils_pi import pi_convertfrom_pair
from vsn import Vsn


class MDLScoreFunction(object):

    def __init__(self, Data_c, #: np.ndarray,
                 vsn : Vsn,
                 use_bic=False):
        self.Data_c = Data_c
        self.C_n = Data_c.shape[0]
        self.score_cache = {}
        self.vsn = vsn
        self.use_bic = use_bic

    def score(self, Data_c, #: np.ndarray,
               i: int, PAi: List[int], parameters:None) -> float:
        hash_key = f'i_{str(i)}_PAi_{str(PAi)}'
        if self.score_cache.__contains__(hash_key):
            return self.score_cache[hash_key]
        else:
            if self.use_bic:
                res = self._score_BIC(Data_c, i, PAi)
            else:
                res = self._score_mdl(Data_c, i, PAi)
            self.score_cache[hash_key] = res
            return res

    def _score_mdl(self, Data_c, #: np.ndarray,
                    i: int, PAi: List[int]):
        #Data = np.mat(data)
        #T = Data.shape[0]
        #X = Data[:, i]

        if len(PAi) != 0:
            if self.vsn.subsample_size is None:
                PAc = np.array([Data_c[c_i][:, PAi] for c_i in range(self.C_n)])
                Yc = np.array([Data_c[c_i][:, i] for c_i in range(self.C_n)])

            else:
                PAc = np.array([Data_c[c_i][:, PAi][: self.vsn.subsample_size] for c_i in range(self.C_n)])
                Yc = np.array([Data_c[c_i][:, i][: self.vsn.subsample_size] for c_i in range(self.C_n)])

            #Pi = [[c_i for c_i in range(self.C_n)]]
            gpr_PA = PI(PAc, Yc, np.random.RandomState(1), skip_regression_pairs=self.vsn.regression_per_pair, rff=self.vsn.rff,pi_search=self.vsn.pi_search)
            gpr_PA.cmp_distances()

            pair_gains = gpr_PA.pair_mdl_gains
            pair_indicator_vars, _, pair_contexts = pi_search_ILP(pair_gains, self.C_n, wasserstein=False)

            pi_star = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, self.C_n)

            pistar_score, _, pistar_model, pistar_data, pistar_penalty = pi_mdl(gpr_PA, pi_star, regression_per_group=self.vsn.regression_per_group)

            score = -pistar_score

        else:
            return 0 #TODO
            #sigma2 = np.sum(np.power(X, 2)) / T
            # BIC
            #score = T * np.log(sigma2)
        return score

    #TODO extend to mult. contexts
    def _score_BIC(self, Data: np.ndarray, i: int, PAi: List[int], parameters=None) -> float:
        """
        Calculate the *negative* local score with BIC for the linear Gaussian continue data case
        Parameters
        ----------
        Data: ndarray, (sample, features)
        i: current index
        PAi: parent indexes
        parameters: lambda_value, the penalty discount of bic
        Returns
        -------
        score: local BIC score
        """

        if parameters is None:
            lambda_value = 1
        else:
            lambda_value = parameters['lambda_value']

        Data = np.mat(Data)
        T = Data.shape[0]
        X = Data[:, i]

        if len(PAi) != 0:
            PA = Data[:, PAi]
            D = PA.shape[1]
            # derive the parameters by maximum likelihood
            H = PA * pdinv(PA.T * PA) * PA.T
            E = X - H * X
            sigma2 = np.sum(np.power(E, 2)) / T
            # BIC
            score = T * np.log(sigma2) + lambda_value * D * np.log(T)
        else:
            sigma2 = np.sum(np.power(X, 2)) / T
            # BIC
            score = T * np.log(sigma2)

        return score
