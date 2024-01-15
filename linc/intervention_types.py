from graphical_models import SoftInterventionalDistribution
from typing import Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum


class IvType(Enum):
    PARAM_CHANGE = 1
    # noise interventions
    SCALE = 2
    SHIFT = 3
    # constant
    CONST = 4
    GAUSS = 5
    # hidden variables
    CONFOUNDING = 6
    HIDDEN_PARENT = 7

    def __str__(self):
        if self.value == 1:
            return "IvChange"
        if self.value == 2:
            return "IvScaling"
        if self.value == 3:
            return "IvShift"
        if self.value == 4:
            return "IvPerfect"
        else:
            return "IvOther"

#
# #overrides causaldag classes
#
# @dataclass
# class ScalingIv(SoftInterventionalDistribution):
#     factor: float = 1
#     noise_factor: float = 1
#     mean: Optional[float] = None
#
#     def sample(self, parent_values: Optional[np.ndarray], dag, node) -> np.ndarray:
#         nsamples, nparents = parent_values.shape
#         node_ix = dag._node2ix[node]
#         mean = dag._biases[node_ix] if self.mean is None else self.mean
#         std = dag._variances[node_ix]**.5 * self.noise_factor
#         noise = np.random.normal(loc=mean, scale=std, size=nsamples)
#         parent_ixs = [dag._node2ix[p] for p in dag._parents[node]]
#         if nparents != 0:
#             return np.sum(parent_values * dag._weight_mat[parent_ixs, node]*self.factor, axis=1) + noise
#         else:
#             return noise
#
#     def pdf(self, vals: np.ndarray, parent_values: np.ndarray, dag, node) -> float:
#         pass
#
#
# @dataclass
# class ShiftIv(SoftInterventionalDistribution):
#     shift: float
#
#     def sample(self, parent_values: Optional[np.ndarray], dag, node) -> np.ndarray:
#         from graphical_models import GaussDAG, SampleDAG
#         if isinstance(dag, GaussDAG):
#             nsamples, nparents = parent_values.shape
#             node_ix = dag._node2ix[node]
#             noise = np.random.normal(loc=dag._biases[node_ix] + self.shift, scale=dag._variances[node_ix] ** .5, size=nsamples)
#             parent_ixs = [dag._node2ix[p] for p in dag._parents[node]]
#             if nparents != 0:
#                 return np.sum(parent_values * dag._weight_mat[parent_ixs, node], axis=1) + noise
#             else:
#                 return noise
#         elif isinstance(dag, SampleDAG):
#             nsamples = parent_values.shape[0]
#             samples = np.zeros(nsamples)
#             for sample_num in range(nsamples):
#                 samples[sample_num] = dag.conditionals[node](parent_values[sample_num, :]) + self.shift
#             return samples
#
#     def pdf(self, vals: np.ndarray, parent_values: np.ndarray, dag, node) -> float:
#         pass