import itertools
from typing import List

from numpy import ndarray
from causallearn.utils.ScoreUtils import *

from pi import PI
from pi_search_ilp import pi_search_ILP
from pi_search_mdl import pi_mdl
from utils_pi import pi_convertfrom_pair

class MDLPairwiseScoreFunction(object):

    def __init__(self, Data_c: ndarray, subsample: int, regression_per_group):

        self.Data_c = Data_c
        self.C_n = Data_c.shape[0]
        self.X_n = Data_c.shape[2]
        self.score_cache = {}
        self.subsample = subsample
        self.regression_per_group=regression_per_group

        self.pair_pistar_scores = [0 for _ in itertools.permutations(range(self.X_n), 2)]
        self.pair_pistar_model = [0 for _ in itertools.permutations(range(self.X_n), 2)]
        self.pair_pistar_data = [0 for _ in itertools.permutations(range(self.X_n), 2)]
        self.pair_pistar_penalty = [0 for _ in itertools.permutations(range(self.X_n), 2)]
        self.pair_pistars = [[] for _ in itertools.permutations(range(self.X_n), 2)]

        combs = itertools.permutations(range(self.X_n), 2)
        it = -1
        for i,j in combs:
            it = it + 1
            PAc = np.array([Data_c[c_i][:, [i]][: self.subsample] for c_i in range(self.C_n)])
            Yc = np.array([Data_c[c_i][:, j][: self.subsample] for c_i in range(self.C_n)])

            Pi = [[c_i for c_i in range(self.C_n)]]
            gpr_PA = PI(PAc, Yc, np.random.RandomState(1)) #TODO random state relevant for Pi?
            gpr_PA.cmp_distances()

            pair_gains = gpr_PA.pair_mdl_gains
            pair_indicator_vars, _, pair_contexts = pi_search_ILP(pair_gains, self.C_n, wasserstein=False)

            pi_star = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, self.C_n)

            pistar_score, _, pistar_model, pistar_data, pistar_penalty = pi_mdl(gpr_PA, pi_star, regression_per_group=self.regression_per_group)

            #TODO score = -pistar_score
            self.pair_pistar_scores[it] = pistar_score
            self.pair_pistar_model[it] = pistar_model
            self.pair_pistar_data[it] = pistar_data
            self.pair_pistar_penalty[it] = pistar_penalty
            self.pair_pistars[it] = pi_star



    def score(self, Data_c: ndarray, i: int, PAi: List[int], parameters:None) -> float:
        hash_key = f'i_{str(i)}_PAi_{str(PAi)}'
        if self.score_cache.__contains__(hash_key):
            return self.score_cache[hash_key]
        else:
            res = self._score_mdl(Data_c, i, PAi)
            self.score_cache[hash_key] = res
            return res

    def _score_mdl(self, Data_c: ndarray, i: int, PAi: List[int]):
        #Data = np.mat(data)
        #T = Data.shape[0]
        #X = Data[:, i]
        #return 0

        if len(PAi) != 0:
            PAc = np.array([Data_c[c_i][:, PAi][: self.subsample] for c_i in range(self.C_n)])
            Yc = np.array([Data_c[c_i][:, i][: self.subsample] for c_i in range(self.C_n)])

            Pi = [[c_i for c_i in range(self.C_n)]]
            gpr_PA = PI(PAc, Yc, np.random.RandomState(1)) #TODO rst relevant for Pi?
            gpr_PA.cmp_distances()

            pair_gains = gpr_PA.pair_mdl_gains
            pair_indicator_vars, _, pair_contexts = pi_search_ILP(pair_gains, self.C_n, wasserstein=False)

            pi_star = pi_convertfrom_pair(pair_indicator_vars, pair_contexts, self.C_n)

            pistar_score, _, pistar_model, pistar_data, pistar_penalty = pi_mdl(gpr_PA, pi_star, regression_per_group=self.regression_per_group)

            score = -pistar_score

        else:
            return 0 #TODO
            #sigma2 = np.sum(np.power(X, 2)) / T
            # BIC
            #score = T * np.log(sigma2)
        return score

    #TODO extend to mult. contexts
    def _score_BIC(self, Data: ndarray, i: int, PAi: List[int], parameters=None) -> float:
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
