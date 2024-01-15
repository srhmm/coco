
from graphical_models import GaussDAG
from numpy import ndarray

from pi_tree import PiTree


def TP_tree_adj_cpdag (T:PiTree, G: ndarray, cpdag: ndarray):
    TP, TN, FP,  FP_anticausal, FN  = 0,0,0,0,0
    for i in T.nodes:
        for j in T.nodes:
            if cpdag[i][j] == 1 & cpdag[j][i] == 1:
                if T.is_edge(i, j):
                    if G[i][j] == 1:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                        if G[j][i] == 1:
                            FP_anticausal = FP_anticausal + 1
                else:
                    if G[i][j] == 1:
                        FN = FN + 1
                    else:
                        TN = TN + 1
    return  TP, TN, FP,  FP_anticausal, FN

def TP_tree_dag_cpdag (T:PiTree, G: GaussDAG, cpdag: ndarray):
    TP, TN, FP,  FP_anticausal, FN  = 0,0,0,0,0
    for i in T.nodes:
        for j in T.nodes:
            if cpdag[i][j] == 1 & cpdag[j][i] == 1:
                if T.is_edge(i, j):
                    if (i, j) in G.arcs:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                        if (j, i) in G.arcs:
                            FP_anticausal = FP_anticausal + 1
                else:
                    if (i, j) in G.arcs:
                        FN = FN + 1
                    else:
                        TN = TN + 1
    return  TP, TN, FP,  FP_anticausal, FN

def TP_tree_dag (T:PiTree, G: GaussDAG):
    TP, TN, FP,  FP_anticausal, FN  = 0,0,0,0,0
    for i in T.nodes:
        for j in T.nodes:
            if T.is_edge(i, j):
                if (i, j) in G.arcs:
                    TP = TP + 1
                else:
                    FP = FP + 1
                    if (j, i) in G.arcs:
                        FP_anticausal = FP_anticausal + 1
            else:
                if (i, j) in G.arcs:
                    FN = FN + 1
                else:
                    TN = TN + 1
    return  TP, TN, FP,  FP_anticausal, FN

