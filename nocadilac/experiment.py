import numpy as np
import networkx as nx
from data_gen import DataGen
from model import ConfoundedAE as CAE, train
from util import metrics, conf_eval, acyclify
from pathlib import Path


# Synthetic data: Sec 5.1
def synthetic(N=1000):
    for M in [10, 25, 50]:
        for gen in ['lin', 'quad', 'cub', 'log', 'exp', 'sin']:
            dg  = DataGen(N, M, 0.3, gen=gen)
            dg.net()
            x = dg.data()
            conf = dg.conf_ind

            model = CAE(M, 1)
            best_A, best_B = train(model, x)
            A = acyclify(best_A.numpy())
            best_B = best_B.numpy()

            G_true = dg.G
            G_guess = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
            prec, rec, shd, sid, sid_rev = metrics(G_guess, G_true)

            prec_c, rec_c = conf_eval(best_B, conf)

            with open(f'res/synth/noc-{M}-{gen}.csv', 'a') as f:
                f.write(f'{prec}, {rec}, {shd}, {sid}, {sid_rev}, {prec_c}, {rec_c}')
            break
        break

# REGED: Sec. 5.2
def reged():
    # REGED graph and data
    G = nx.from_scipy_sparse_matrix(loadmat('data/reged/GS.mat')['DAG'])
    data = loadmat('data/reged/data_obs.mat')['data_obs']

    pars = []
    for i in G.nodes:
        if G.out_degree(i) >= 3:
            pars.append(i)

    # Generate graphs around the children of nodes with outdegree 3+
    gs = []
    for i in pars:
        g = []
        nodes = []
        for c in G.successors(i):
            preds = list(G.predecessors(c))
            succ = list(G.successors(c))
            sibs = []
            for s in succ:
                sibs += list(G.predecessors(s))
            mb = preds + succ + sibs
            nodes += mb
        nodes = set(nodes)
        g = G.subgraph(nodes)
        gs.append(g)

    gs = [g for g in gs if 10 <= len(g.nodes) <= 100]

    for i, g in enumerate(gs):
        nodes = list(g.nodes)
        x = data.iloc[:, nodes]

        model = CAE(len(nodes), 1)
        best_A, best_B = train(model, x)
        A = acyclify(best_A.numpy())
        best_B = best_B.numpy()
        G_true = dg.G
        G_guess = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

        prec, rec, shd, sid, sid_rev = metrics(G_guess, G_true)
        prec_c, rec_c = conf_eval(best_B, conf)

        with open(f'res/reged/noc-reged.csv', 'a') as f:
            f.write(f'{prec}, {rec}, {shd}, {sid}, {sid_rev}, {prec_c}, {rec_c}')

# Sachs: Sec. 5.3
def sachs():
    # Sachs graph an ddata
    G = nx.from_scipy_sparse_matrix(loadmat('data/sachs/dag.mat')['DAG'])
    x = loadmat('data/sachs/data.mat')['data']
    x = x[:, 1:]

    nodes = list(G.nodes())
    model = CAE(len(nodes), 1)
    best_A, best_B = train(model, x)
    A = acyclify(best_A.numpy())
    best_B = best_B.numpy()

    G_true = dg.G
    G_guess = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

    with open(f'res/sachs/noc-sachs.csv', 'a') as f:
        f.write(f'{prec}, {rec}, {shd}, {sid}, {sid_rev}, {prec_c}, {rec_c}')

if __name__ == '__main__':
    Path('res/synth').mkdir(parents=True, exist_ok=True)
    Path('res/reged').mkdir(parents=True, exist_ok=True)
    Path('res/sachs').mkdir(parents=True, exist_ok=True)

    synthetic()
    reged()
    sachs()

