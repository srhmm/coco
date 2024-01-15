import numpy as np
from sklearn.linear_model import LogisticRegression
from vario.linear_regression import coef_rescale


def logistic_regression(X, y, scale=True):
    """Given Data X,y in several contexts or groups where y is binary, logistic regression per group

    :param X: X, list of np.arrays
    :param y: y, binary
    :return: regression coefficient per context/group
    """
    coef_k_x = np.array([LogisticRegression().fit(X[pi_k], y[pi_k]).coef_
                      for pi_k in range(len(X))])
    if scale:
        return coef_rescale(coef_k_x, len(X),  X_n=len(coef_k_x[0]))

    return coef_k_x
