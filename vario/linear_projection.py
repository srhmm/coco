import numpy as np

from vario.linear_regression import linear_regression, linear_test_invariance
from vario.utils import logg
from vario.utils_context_partition import partition_to_groupmap


class LinearProjection():
    """
    Given data in n contexts and a target variable Y with covariates X,
    we represent this data by the linear parameters of f:X -> Y in each context
    """

    def __init__(self, data_each_context, index_Y, indices_X=None):
        self.index_Y = index_Y
        self.indices_X = indices_X
        self.intercept=True

        all_nodes = data_each_context[0].shape[1]
        if indices_X is None:
            self.n_nodes = all_nodes
            indices_X = range(self.n_nodes)
        else:
            self.n_nodes = len(indices_X)

        assert 0 <= self.n_nodes <= all_nodes
        #assert 0 < self.n_nodes <= all_nodes

        self.n_contexts = len(data_each_context)
        self.contexts = range(self.n_contexts)
        self.nodes = range(self.n_nodes)
        assert 0<=index_Y<all_nodes

        self.data_Y = np.array([data_each_context[c_i][:, self.index_Y] for c_i in range(self.n_contexts)])
        if self.n_nodes==0:
            sub_size = data_each_context.shape[1]
            self.data_X = np.array([[[self.data_Y[c_i][x_i], ] for x_i in range(sub_size)] for c_i in self.contexts]) #TODO
            # or:
            #    Xc = np.array([[[0, ] for x_i in range(sub_size)] for c_i in range(C_n)])
        else:
            self.data_X = np.array([data_each_context[c_i][:, [i for i in self.indices_X]] for c_i in range(self.n_contexts)])

        self.linear_regression()


    def linear_regression(self):
        """Linear regression per context

        """
        coef, stdv, coef_scaled, resid_bias, resid_stdv, resid_scaled = linear_regression(self.data_X, self.data_Y)
        self._scaled_coef = coef_scaled
        self._lin_coef = coef
        self._lin_stdv = stdv
        self._lin_resid_bias = resid_bias
        self._lin_resid_scaled = resid_bias
        self._lin_resid_stdv = resid_bias
        # column j: coefficients for variable j in each context, resp. pvals for each pair of contexts

    def _param_group(self, partition, pi_k, x_i):
        return sum([self._scaled_coef[:, x_i][c_i] for c_i in partition[pi_k]]) / len(partition[pi_k])

    def _param_intercept_group(self, partition, pi_k):
        return sum([self._lin_resid_scaled[c_i] for c_i in partition[pi_k]]) / len(partition[pi_k])

    def _invariant_group(self, partition, pi_k, x_i):
        coef_k = [self._lin_coef[:, x_i][c_i] for c_i in partition[pi_k]]
        stdv_k = [self._lin_stdv[:, x_i][c_i] for c_i in partition[pi_k]]
        invariant, pvals = linear_test_invariance(coef_k, stdv_k)
        is_invariant = not (False in invariant)
        return is_invariant

    def _invariant_group_intercept(self, partition, pi_k):
        coef_k = [self._lin_resid_bias[c_i] for c_i in partition[pi_k]]
        stdv_k = [self._lin_resid_stdv[c_i] for c_i in partition[pi_k]]
        invariant, pvals = linear_test_invariance(coef_k, stdv_k)
        is_invariant = not (False in invariant)
        return is_invariant

    def is_invariant_group(self, partition, pi_k):
        return not False in (
            [(self._invariant_group(partition, pi_k, x_i)) for x_i
             in range(len(self.indices_X))])


    def is_invariant(self, partition):

        invariant_params = not False in (
            [not False in [(self._invariant_group(partition, pi_k, x_i)) for pi_k in range(len(partition))] for x_i in
             range(len(self.indices_X))])
        invariant_intercept = (not False in [(self._invariant_group_intercept(partition, pi_k)) for pi_k in range(len(partition))])
        if self.intercept:
            return invariant_params and invariant_intercept
        else:
            return invariant_params

    def _error_group(self, partition, param_group, is_invariant_group, c_i, x_i):
        map = partition_to_groupmap(partition, self.n_contexts)

        if is_invariant_group[map[c_i]]: #invariant_group[x_i][map[c_i]]:
            return 0
        return (self._scaled_coef[:, x_i][c_i] - param_group[x_i][map[c_i]]) ** 2

    def _error_intercept_group(self, partition, param_group, c_i):
        map = partition_to_groupmap(partition, self.n_contexts)

        return (self._lin_resid_scaled[c_i] - param_group[len(param_group)-1][map[c_i]]) ** 2

    def _param_per_group(self, partition):

        #invariant_group = [[self._invariant_group(partition, pi_k, x_i)
        #                for pi_k in range(len(partition))]
        #               for x_i in self.nodes]

        invariant_group = [self.is_invariant_group(partition, pi_k)
                        for pi_k in range(len(partition))]
        param_group = [[self._param_group(partition, pi_k, x_i)
                        for pi_k in range(len(partition))]
                       for x_i in self.nodes]
        if self.intercept:
            param_group.append([self._param_intercept_group(partition, pi_k)for pi_k in range(len(partition))])

        #param_group = [[sum([self._parameters[:, x_i][c_i] for c_i in partition[pi_k]]) / len(partition[pi_k])
        #                for pi_k in range(len(partition))]
        #               for x_i in self.nodes]  # [[c(x1, pi1),...],#[c(x2, pi1),...]]

        param_error = [[self._error_group(partition,
                                          param_group, invariant_group, c_i, x_i)#(self._parameters[:, x_i][c_i] - param_group[x_i][map[c_i]]) ** 2
                        for c_i in self.contexts]
                       for x_i in self.nodes]

        if self.intercept:
            param_error.append([self._error_intercept_group(partition, param_group, c_i) for c_i in self.contexts])

        return param_group, param_error

    def score(self, partition,
              empirical=True,
              resid=False):
        """ Vario score

        :param pi_test: partition
        :param empirical: empirical score on euclidean distance (typically better for finding mechanism changes)
        :param resid: full MDL score including residuals (not so good unless the true data generating process is linear)
        :return: score
        """
        #map = partition_to_groupmap(partition, self.n_contexts)
        #map_onegroup = [0 for _ in range(self.n_contexts)]

        partition_onegroup = [[c_i for c_i in range(self.n_contexts)]]

        param_onegroup, param_error_onegroup = self._param_per_group(partition_onegroup)
        param_group, param_error = self._param_per_group(partition)


        #param_group = [[sum([self._parameters[:, x_i][c_i] for c_i in partition[pi_k]]) / len(partition[pi_k])
        #                for pi_k in range(len(partition))]
        #               for x_i in self.nodes] #[[c(x1, pi1),...],#[c(x2, pi1),...]]


        #param_error = [[(self._parameters[:, x_i][c_i] - param_group[x_i][map[c_i]]) ** 2
        #                for c_i in self.contexts]
        #              for x_i in self.nodes]

        #param_onegroup = [[sum(self._parameters[:, x_i]) / self.n_contexts]
        #                  for x_i in self.nodes]

        #param_error_onegroup = [[(self._parameters[:, x_i][c_i] - param_onegroup[x_i][0]) ** 2 for c_i in self.contexts]
        #                       for x_i in self.nodes]
        #

        if empirical:
            sse = [logg(sum(param_error[x_i]))
                    for x_i in self.nodes]
            sse_one = [logg(sum(param_error_onegroup[x_i]))
                    for x_i in self.nodes]
            vario_score = sum([(sse_one[x_i] - sse[x_i]) / max(len(partition) - 1, 1)
                               for x_i in self.nodes])
        else:
            sig = 1
            mdl_coef = sum([(self.n_contexts / 2) * logg(2 * np.pi * (sig ** 2)) + 1 / (2 * (sig ** 2)) * sum(param_error[x_i])
                            for x_i in self.nodes]) # coefficient errors are per dimension
            mdl_alpha = (logg(self.n_contexts) / 2) * len(partition)
            mdl_pi = self.n_contexts * logg(self.n_contexts) + logg(self.n_contexts) # TODO constant model cost aspects?

            if resid:
                num_samples = sum([len(self.data_Y[c_i]) for c_i in self.contexts])

                residuals_context = [sum((sum([self._parameters[c_i, x_i]*self.data_X[c_i][:,x_i] for x_i in self.nodes])
                                   - self.data_Y[c_i]) ** 2) for c_i in self.contexts]

                mdl_data = (num_samples / 2) * logg(2 * np.pi * (sig ** 2)) \
                           + 1 / (2 * (sig ** 2)) * (logg(sum(residuals_context)))
                vario_score = mdl_coef + mdl_alpha + mdl_pi + mdl_data

            else:
                vario_score = mdl_coef + mdl_alpha + mdl_pi
        return vario_score
