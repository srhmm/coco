import numpy as np
from graphical_models import GaussDAG

from vario.utils_context_partition import pi_matchto_pi_pairwise, pi_matchto_pi_exact


def eval_causal_edges(causal_edges, true_partitions, true_dag:GaussDAG):
    match, nomatch = 0,0
    adj = np.zeros((len(true_dag.nodes), len(true_dag.nodes)))

    for entry_for_Y in causal_edges:
        pi, score, parents, iy = causal_edges[entry_for_Y]
        true_pi = true_partitions[iy]
        tr, f = pi_matchto_pi_exact(pi, true_pi)
        match, nomatch = match + tr, nomatch + f
        for ix in parents:
            adj[ix, iy] = 1

    tp , fn, fp, fp_rev, tn = 0,0,0,0,0
    for i in true_dag.nodes:
        for j in true_dag.nodes:
            if (i, j) in true_dag.arcs:
                if adj[i, j] == 1:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if adj[i,j] == 1:
                    if (j, i) in true_dag.arcs:
                        fp_rev = fp_rev + 1
                    else:
                        fp = fp + 1
                else:
                    tn = tn + 1

    print("\n--- Evaluation of DAG search---")
    print("Correct Partitions: ", match, "/", nomatch+match)
    print("Causal (TP) edges: ", tp, "/", tp+fn)
    print("Anticausal (FP): ", fp_rev, "", ", Spurious (FP): ", fp, ", TN:", tn)


def eval_partition(partition, true_partition):
    match, nomatch = pi_matchto_pi_exact(partition, true_partition)

    print("Correct Partitions: ", match, "/", nomatch+match)