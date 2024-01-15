import numpy as np
import math
from sklearn import preprocessing

def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)

def logg (val):
    if(val==0): return 0
    else: return math.log(val)

def scale_data(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return(scaler.transform(y))


def data_groupby_pi(Xc, yc, Pi):
    """Given a partition Pi={pi_k}, pool data within each group pi_k"""
    n_pi = len(Pi)
    n_c = Xc.shape[0]

    assert((len(Xc.shape) == 3) & (len(yc.shape) == 2))
    assert((yc.shape[0] == n_c) & (yc.shape[1] == Xc.shape[1]))

    Xpi = [np.concatenate([Xc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]
    ypi = [np.concatenate([yc[c_i] for c_i in Pi[pi_k]]) for pi_k in range(n_pi)]

    #print("Partition:", Pi)
    #print("Data Shape:", [Xpi[i].shape for i in range(len(Xpi))])

    return(Xpi, ypi)

