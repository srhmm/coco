from glob import glob
from data_gen import DataGen
from cdt.metrics import SHD
from cdt.metrics import SID
import tensorflow as tf

def matrix_poly(matrix, d):
    x = tf.eye(d)+ matrix/d
    return x**d

def _h(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = tf.linalg.trace(expm_A) - m
    return h_A

def acyclify(A):
    np.fill_diagonal(A, 0)
    A_abs = np.abs(A)
    while _h(A_abs) > 0:
        min = np.min(A_abs[np.nonzero(A_abs)])
        ind = np.where(A_abs == min)
        A_abs[ind[0][0], ind[1][0]] = 0
        return A
    A[A_abs == 0] = 0
    return _A

def precision(s0, s1):
    s = set(s0).intersection(s1)
    return len(s)/len(s1)

def recall(s0, s1):
    s = set(s0).intersection(s1)
    return len(s)/len(s0)

def precision_recall(s1, s2):
    return precision(s1, s2), recall(s1, s2)

def metrics(G_inf, G_true):
    e0 = G_true.edges()
    e1 = G_inf.edges()
    prec, rec = precision_recall(e1, e0)
    sid = SID(G_inf, G_true)
    sid_rev = SID(G_true, G_inf)
    shd = SHD(G_inf, G_true)
    return prec, rec, shd, sid, sid_rev

def conf_eval(B, conf):
    bmax = max(B)
    B[B < bmax/3] = 0
    conf_guess = set(np.nonzero(B))
    return precision_recall(conf, conf_guess)
