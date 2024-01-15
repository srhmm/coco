import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
import operator
import random
from scipy.io import loadmat
import networkx as nx

import os

from method import Method
from util import precision, recall
from config import RESULTS_DIR

random.seed(1)

path = os.path.join(RESULTS_DIR, 'real/reged/')

def run(G, df):
    for v in G.nodes:
        children = list(G.successors(v))
        parents = list(G.predecessors(v))
        if len(children) > 5:
            other_nodes = random.sample([u for u in G.nodes if not (u in children or u in parents or u==v)], len(children))
            G_ = G.subgraph(children + other_nodes)
            df_ = df.loc[:, list(G_.nodes)]
            G_ = nx.relabel_nodes(G_, {v:f'C{i}' for i,v in enumerate(children)})
            df_.rename(columns={v:f'C{i}' for i,v in enumerate(children)}, inplace=True)
            m = Method(df_)
            if m.G.edges:
                m.iterative_fit(k=1)
            model_name = os.path.join(path, f'model_{v}.pickle')
            graph_name = os.path.join(path, f'graph_{v}.pickle')
            with open(model_name, 'wb') as f:
                pickle.dump(m, f)
            with open(graph_name, 'wb') as f:
                pickle.dump(G_, f)

def plot():
    plt.close('all')
    l = []
    for fname in glob.glob(os.path.join(path, 'model_*.pickle')):
        with open(fname, 'rb') as f:
            m = pickle.load(f)
            confidence = (m.orig_score - m.GES_score)/m.GES_score
            D = len(m.G.nodes)
            if confidence > 0:
                conf_true = set(x for x in m.G.nodes if isinstance(x, str) and x.startswith('C'))
                conf = set(y for x, y in m.G.edges if isinstance(x, str) and x.startswith('z'))
                prec = precision(set(), conf, conf_true)
                rec = recall(set(), conf, conf_true)
                f1 = 2*prec*rec/(prec+rec)
                l.append([confidence, prec, rec, f1, D])
    df = pd.DataFrame(l)
    df.columns = ['confidence', 'prec', 'rec', 'f1', 'D']
    df_l = df[df.D <= 15]
    df_r = df[df.D > 15]
    for i, df in enumerate([df_l, df_r]):
        df.sort_values(['confidence'], ascending=False, inplace=True)
        df.reset_index(inplace=True)
        df['confidence'] = df.index/df.shape[0]
        f1 = df.f1.expanding().mean()
        df_i = pd.DataFrame([df.confidence, f1]).T
        df_i.to_csv(f'xrs/reged-{i}.dat', index=None, header=None, sep=' ')


if __name__ == '__main__':
    G = nx.from_scipy_sparse_matrix(loadmat('data/reged/GS.mat')['DAG'])
    df = loadmat('data/reged/data_obs.mat')['data_obs']
    run(G, df)
    plot()
