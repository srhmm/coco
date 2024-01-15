from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from sklearn.gaussian_process.kernels import RBF

from linc.utils import cantor_pairing
from linc.utils_pi import pi_group_map


class FunctionType(Enum):
    LINEAR_GAUSS = 1
    GP_PRIOR = 2
    def __str__(self):
        if self.value == 1:
            return "LinGauss"
        if self.value == 2:
            return "NonlinGpANM"

def kernel_rbf(X1, X2, length_scale=1):
    return RBF(length_scale=length_scale).__call__(X1, eval_gradient=False)
#
def kernel_exponentiated_quadratic(X1, X2):
    # as in https://peterroelants.github.io/posts/gaussian-process-tutorial/
    sq_norm = -0.5 * scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
    return np.exp(sq_norm)


def gen_data_gp_example(seed=1):
    domain_min=-10
    domain_max=10
    D_n = 50
    C_n = 5
    X = np.expand_dims(np.linspace(domain_min, domain_max, D_n), 1)
    gen_data_gp(X, D_n, C_n, seed,  show = True)

def gen_data_gp(X, D_n, C_n, seed,
                partition = [[0,1], [2,3,4]],
                domain_min=-10, domain_max=10, group_variance=0.2,
                kernel_function=kernel_rbf, #kernel_exponentiated_quadratic,
                show=False):
    """ draw samples from a Gaussian Processs

    :param D_n: number of samples
    :param seed: random seed
    :param f_n: number of functions drawn
    :param domain_min: domain X
    :param domain_max: domain X
    :param kernel_function: kernel
    :param show: plotting
    :return:
    """

    E = kernel_function(X, X)
    f_n = len(partition)
    ys = np.random.RandomState(seed).multivariate_normal(
        mean=np.zeros(D_n), cov=E,
        size=f_n)
    y = np.empty((C_n, D_n))
    for pi_k in range(len(partition)):
        for c_i in partition[pi_k]:
            y[c_i, :] = ys[pi_k, :] + np.random.RandomState(cantor_pairing(seed,c_i)).normal(scale=group_variance,size=D_n) #noise[pi_k, :]#
    if show:
        plot_data_gp(X, y, D_n, C_n, domain_min,domain_max)
    return X, y


def plot_data_gp(X, ys, D_n, f_n, domain_min, domain_max):
    plt.figure(figsize=(6, 4))
    for i in range(f_n):
        plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title((
        str('%s realization(s) at %s points\n' % (f_n, D_n))+
        'sampled from GP with exponentiated quadratic kernel'))
    plt.xlim([domain_min, domain_max])
    plt.show()
    return X, ys
# want ys[0]


def something():
    Xtest = np.random.uniform(-3., 3., (100, 1))
    n = len(Xtest)

    noise_var = 1

    def ker(x, y, l2):
        sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + \
                 np.sum(y ** 2, 1) - 2 * np.dot(x, y.T)
        return np.exp(-.5 * (1 / l2) * sqdist)

    K = ker(Xtest, Xtest, 0.1)
    L = np.linalg.cholesky(K + noise_var * np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, 100)))
    return f_prior



# some Example
def f_sinusoidal(C_n, D_n, X_n, Xc, y_offset, Pi, a1,
                 random_state):
    group_map = pi_group_map(Pi, C_n)
    y_noise_std = [(random_state.choice(2)+1)  for _ in range(C_n)]
    #y_noise_std = random_state.choice(2, C_n, replace=True)+1
    #a1 = random_state.choice(5, len(Pi), replace=False)+1
    print("Group Parameters: ", str(a1))
    #a2 = [random_state.choice(5, 1, replace=False)/2 for pi in Pi]
    #group_map = utils.pi_group_map(Pi, C_n)
    return (np.array([np.squeeze(Xc[c_i] * np.sin(Xc[c_i]) * a1[group_map[c_i]]
                                 + random_state.normal(loc=0.0, scale=y_noise_std[c_i], size=(D_n, X_n))
                                 + y_offset[c_i]) for c_i in range(C_n)]), a1)