import pickle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import operator
import random
import os
import re
import sys
import itertools
import subprocess
from collections import defaultdict

from scipy.stats import dirichlet
import numpy as np
import pandas as pd
import pydot

from config import TETRAD_ON, METHOD_ON
from method import Method
from data_generation import generate_data

from structural_metrics import graph_dists
from util import f1, precision, recall, prec_rec_f1

def eval_tetrad(N=5000, D=10, M=1,num_conf=3, algo='gfci'):
    from config import RESULTS_DIR
    conf_set = set(range(5))
    conf_pairs = list(itertools.combinations(conf_set, 2))
    measures = list()

    for alpha in 2**np.linspace(0, 3, 4):
        path = os.path.join(RESULTS_DIR, f'synthetic/alpha={alpha}')
        for i in range(10):

            tetrad_path = os.path.join(path, f'{algo}_conf-{N}-{D}-{M}-{num_conf}[{i}]')
            try:
                with open(tetrad_path, 'rb') as tetrad_file:
                    maybe_conf, def_conf = pickle.load(tetrad_file)
            except FileNotFoundError:
                continue
            maybe_conf_set = set(x for x, _ in maybe_conf).union(y for _, y in maybe_conf)
            def_conf_set = set(x for x, _ in def_conf).union(y for _, y in def_conf)
            maybe_conf_set = maybe_conf_set.difference(def_conf_set)
            all_conf_set = def_conf_set.union(maybe_conf_set)
            pair_prec, pair_rec, pair_f1 = prec_rec_f1(maybe_conf, def_conf, conf_pairs)
            set_prec, set_rec, set_f1 = prec_rec_f1(set(), all_conf_set, conf_set)
            measures.append([alpha, pair_prec, pair_rec, pair_f1, set_prec, set_rec, set_f1])
    df = pd.DataFrame(measures, columns=['alpha', 'pair_prec', 'pair_rec', 'pair_f1', 'set_prec', 'set_rec', 'set_f1'])
    return df


def eval_method(N=5000, D=10, M=1,num_conf=3):
    from config import RESULTS_DIR
    conf_set = set(range(num_conf))
    conf_pairs = list(itertools.combinations(conf_set, 2))
    measures = list()
    for alpha in 2**np.linspace(0, 3, 4):
        path = os.path.join(RESULTS_DIR, f'synthetic/alpha={alpha}')
        for i in range(10):
            m_path = os.path.join(path, f'm_conf-{N}-{D}-{M}-{num_conf}[{i}]')
            try:
                with open(m_path, 'rb') as m_file:
                    m = pickle.load(m_file)
            except FileNotFoundError:
                continue
            confidence = (m.orig_score - m.GES_score)/m.orig_score
            considered_confounded = confidence > 0
            if considered_confounded:
                zs = set(x for x,y in m.G.edges if isinstance(x, str) and x.startswith('z'))
                confounded_sets_inferred = {z:set(y for x,y in m.G.edges if x==z) for z in zs}
                seed = confounded_sets_inferred['z1']
            else:
                seed = set()
            seed_pairs = set(itertools.combinations(seed, 2))
            pair_prec, pair_rec, pair_f1 = prec_rec_f1(set(), seed_pairs, conf_pairs)
            set_prec, set_rec, set_f1 = prec_rec_f1(set(), seed, conf_set)
            measures.append([alpha, pair_prec, pair_rec, pair_f1, set_prec, set_rec, set_f1, confidence, considered_confounded])
    df = pd.DataFrame(measures, columns=['alpha', 'pair_prec', 'pair_rec', 'pair_f1', 'set_prec', 'set_rec', 'set_f1', 'confidence', 'considered_confounded'])
    return df

def confidence_f1(N=5000, M=1):
    plt.close('all')
    df = pd.DataFrame()
    f1s = []
    for (D, num_conf) in [(10, 3), (25, 5), (50, 5)]:
        df_ = eval_method(N, D, M, num_conf)
        df_['D'] = D
        df_['num_conf'] = num_conf
        df = df.append(df_, ignore_index=True)
    for D in df.D.unique():
        df_ = df[df.D == D]
        df_.sort_values(['confidence'], ascending=False, inplace=True)
        df_.reset_index(inplace=True)
        df_['confidence'] = df_.index/df_.shape[0]
        f1 = df_.set_f1.expanding().mean()
        f1s.append(f1)
    df1 = pd.DataFrame(f1s).T
    df1.columns=['conf10', 'conf25', 'conf50']
    df1.index = df1.index/df1.shape[0]
    df1.to_csv('xrs/conf-f1.dat', sep=' ', header=False)

def alpha_performance(N=5000, M=1):
    plt.close('all')
    df = pd.DataFrame()
    for (D, num_conf) in [(10, 3), (25, 5), (50, 5)]:
        df_ = eval_method(N, D, M, num_conf)
        df = df.append(df_, ignore_index=True)
    alphas = df.alpha.unique()
    f1s, confs = [], []
    for alpha in alphas:
        df_ = df[df.alpha == alpha]
        f1_a = df_.set_f1.quantile([0, 0.25, 0.5, 0.75, 1])
        conf_a = df_.confidence.quantile([0, 0.25, 0.5, 0.75, 1])
        f1s.append(f1_a)
        confs.append(conf_a)
    f1s = pd.DataFrame(f1s).T
    f1s.columns = ['1.0', '2.0', '4.0', '8.0']
    confs = pd.DataFrame(confs).T
    confs.columns = ['1.0', '2.0', '4.0', '8.0']
    f1s.to_csv('xrs/alpha-f1.dat', sep=' ', header=False)
    confs.to_csv('xrs/alpha-conf.dat', sep=' ', header=False)

def score_comparison():
    N = 5000
    M = 1
    for score in ['pair', 'set']:
        x = defaultdict(list)
        for (D, num_conf) in [(10, 3), (25, 5)]:
            df_t = eval_tetrad(N, D, M, num_conf)
            df_m = eval_method(N, D, M, num_conf)
            df_t = df_t[df_m.considered_confounded == True]
            df_m = df_m[df_m.considered_confounded == True]
            for i, alpha in enumerate(2**np.linspace(0, 3, 4)):
                rec_t = getattr(df_t[df_t.alpha == alpha], f'{score}_rec')
                prec_t = getattr(df_t[df_t.alpha == alpha], f'{score}_prec')
                rec_m = getattr(df_m[df_m.alpha == alpha], f'{score}_rec')
                prec_m = getattr(df_m[df_m.alpha == alpha], f'{score}_prec')
                rec_t_mean = np.mean(rec_t)
                prec_t_mean = np.mean(prec_t)
                rec_m_mean = np.mean(rec_m)
                prec_m_mean = np.mean(prec_m)
                f1_t = f1(prec_t_mean, rec_t_mean)
                f1_m = f1(prec_m_mean, rec_m_mean)
                x[alpha].extend([rec_t_mean, prec_t_mean, f1_t, rec_m_mean, prec_m_mean, f1_m])
    return x

def f1_comp():
    x = score_comparison()
    f1_diffs = []
    r = 0.05
    for alpha in x:
        f1_diffs.extend([x[alpha][2]- x[alpha][5], x[alpha][8]- x[alpha][11]])
    f1_diffs = f1_diffs[1:]
    z0 = 0
    s = 0.5
    f1_diffs = [0] + f1_diffs
    q = len(f1_diffs)
    alpha = [s] + ([1]*(q-1))
    i_t = lambda z: 1 if z > 2*r else 0
    i_r = lambda z: 1 if (-2*r <= z and z <= 2*r) else 0
    i_m = lambda z: 1 if z < -2*r else 0
    thetas = []
    I_t = np.array([[i_t(f1_diffs[i]+f1_diffs[j]) for i in range(q)] for j in range(q)])
    I_r = np.array([[i_r(f1_diffs[i]+f1_diffs[j]) for i in range(q)] for j in range(q)])
    I_m = np.array([[i_m(f1_diffs[i]+f1_diffs[j]) for i in range(q)] for j in range(q)])
    assert (I_t + I_r + I_m).all()
    for _ in range(10000):
        w = dirichlet.rvs(alpha)[0]
        theta_t = w.dot(I_t).dot(w.T)
        theta_r = w.dot(I_r).dot(w.T)
        theta_m = w.dot(I_m).dot(w.T)
        thetas.append((theta_t, theta_r, theta_m))
    df = pd.DataFrame(thetas)
    df_trans = bary(df)
    df_trans.to_csv('xrs/hex-temp.csv')
    subprocess.call('./to_hexbin.R')
    df_trans = pd.read_csv('xrs/hex-temp.csv', sep=' ')
    df_trans.columns = ['x', 'y', 'counts']
    df_trans.counts = df_trans.counts/sum(df_trans.counts)
    df_trans.sort_values('counts', inplace=True)
    df_trans.to_csv('xrs/hexbin.dat', index=None, sep=' ')

def bary(df):
    transform = np.array([[-1/np.sqrt(3), 0, 1/np.sqrt(3)], [0, 1, 0]])
    df.loc[:, 0] += np.abs(np.random.normal(0, 0.01, df.shape[0]))
    df_trans = df.dot(transform.T)
    df_trans.columns = ['x', 'y']
    fig, ax = plt.subplots()
    return df_trans

def performance_tables(N=5000, M=1):
    df_t = pd.DataFrame()
    df_m = pd.DataFrame()
    for (D, num_conf) in [(10, 3), (25, 5), (50, 5)]:
        _df_m = eval_method(N, D, M, num_conf)
        _df_t = eval_tetrad(N, D, M, num_conf)
        _df_m['D'] = D
        _df_t['D'] = D
        df_t = df_t.append(_df_t, ignore_index=True)
        df_m = df_m.append(_df_m, ignore_index=True)
    alphas = df_m.alpha.unique()
    with open('xrs/f1-table.tex', 'w') as f:
        for l, score_type in enumerate(('pair', 'set')):
            for alpha in alphas:
                _df_t = df_t[df_t.alpha == alpha]
                _df_m = df_m[df_m.alpha == alpha]
                t_vals = [_df_t[_df_t.D == d].loc[:, [f'{score_type}_rec', f'{score_type}_prec', f'{score_type}_f1']].mean() for d in (10, 25, 50)]
                m_vals = [_df_m[_df_m.D == d].loc[:, [f'{score_type}_rec', f'{score_type}_prec', f'{score_type}_f1']].mean() for d in (10, 25, 50)]
                print(alpha, end=' & ', file=f)
                for i, (tval, mval)  in enumerate(zip(t_vals, m_vals)):
                    end = ' & ' if i < 2 else r' \\'
                    print(' & '.join(f'{v:.2f}' for v in tval), '&', ' & '.join(f'{v:.2f}' for v in mval), end=end, file=f)
                print('', file=f)
            if not l:
                print('''\midrule
\multicolumn{19}{c}{\emph{Scores measuring the quality of detecting confounded pairs of nodes}} \\
\midrule''', file=f)

if __name__ == '__main__':
    random.seed(0)
    alpha_performance()
    confidence_f1()
    performance_tables()
    # graph_dists()
    f1_comp()
