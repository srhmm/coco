import operator
import numpy as np
import itertools
import random
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from networkx.exception import NetworkXNoCycle
from sklearn import linear_model, metrics

from network import Network
from model import Confounded
from ges import GES



class Method(object):
    def __init__(self, x, z=None, G_true=None, confounded_set=None, num_seeds=0, **kwargs):
        self.G_true = G_true
        self.x = x
        self.z = z
        self.confounded_set = confounded_set
        self.num_seeds = num_seeds
        self.M = kwargs.get('M', 1)
        self.G = kwargs.get('G_init', None)

        # This is the set of nodes which we for whatever reason know cannot be in the confounded set
        # E.g. if we ran this algorithm before and added a new node Z1 to the graph, then we'd know that
        # Z1 can't be in the seed set in the future iterations
        self.forbidden_seeds = kwargs.get('forbidden_seeds', [])
        # self.outgoing_only = kwargs.get('outgoing_only', set())

        # If we didn't include a network to start from, we'll infer one using GES over the observed data.
        if not self.G:
            self.fit_ges()
        self.baseline_scores()

    def fit_ges(self):
        g1 = GES(data=self.x, criterion='compression', outgoing_only=self.forbidden_seeds)
        g1.create_empty_graph()

        for G1 in g1.learn_structure():
            G = g1.G
        self.G = G

    @staticmethod
    def _linear_regression_score(X, y, alpha=1.0):
        model = linear_model.Ridge(alpha)
        N, m = X.shape
        if X.shape[1] == 1:
            X = X.reshape((-1, 1))
        model.fit(X, y)
        w = model.coef_
        loss = metrics.mean_squared_error(y, model.predict(X))
        loss = loss + m/2 * np.log(N)/N

        return loss, model

    @staticmethod
    def fit_confounder(x, M=1):
        cf = Confounded(M, progressbar=False, verbose=False)
        cf.add_data(x)
        cf.create_model()
        cf.fit_model()
        z = cf.trace['Z'][0]
        return z

    def score_data_given_graph(self, x, G):
        _sum = 0
        leaves = [v for v in G.nodes if not list(G.predecessors(v)) and v in x.columns]
        interior = [v for v in G.nodes if v not in leaves and v in x.columns]
        for v in leaves:
            _sum += np.mean((x.loc[:, v] - np.mean(x.loc[:, v]))**2)
        for v in interior:
            pred = list(G.predecessors(v))
            _sum += self._linear_regression_score(x.loc[:, pred].values, x.loc[:, v])[0]
        return _sum

    def markov_blanket_from_graph(self, v):
        G = self.G
        pred = [u for u in G.nodes if (u, v) in G.edges]
        desc = [u for u in G.nodes if (v, u) in G.edges]
        sib = [u for u in G.nodes if any((u, x) in G.edges for x in desc)]

        blanket = [v] + pred + desc + sib
        return tuple(blanket)


    def _conf_score(self, seed, z=None):
        _x = self.x.copy()
        _G = self.G.copy()

        if seed:
            if z is None:
                z = self.fit_confounder(_x, M=self.M)
            _x['z'] = z
            _G.add_node('z')
            for node in seed:
                _G.add_edge('z', node)
        conf_score = self.score_data_given_graph(_x, _G)
        return conf_score, z

    def forward_step(self, seed, z, smart=True):
        G = self.G
        x = self.x
        sub_G = G.copy()
        cands = [v for v in G.nodes if (not v in seed and v not in self.forbidden_seeds)]
        diffs = {}
        Zs = {}
        scores = {}
        for u in seed:
            sub_G.add_edge('z', u)
        _x = x.copy()
        for v in cands:
            cand_G = sub_G.copy()

            _x['z'] = z
            unconf_cand_score = self.score_data_given_graph(_x, cand_G) 

            cand_G.add_edge('z', v)

            if False:
                _seed = seed.copy()
                _seed.add(v)
                z = fit_confounder(x.loc[:, tuple(_seed)], M=self.M)
                _x['z'] = z

            new_score = self.score_data_given_graph(_x, cand_G) 

            diffs[v] = unconf_cand_score - new_score
            Zs[v] = z
            scores[v] = new_score
        return diffs, Zs, scores


    def backward_step(self, seed, z, smart=True):
        G = self.G
        x = self.x
        sub_G = G.copy()
        sub_G.add_node('z')
        for node in seed:
            sub_G.add_edge('z', node)
        _x = x.copy()
        _x['z'] = z
        current_score = self.score_data_given_graph(_x, sub_G)
        diffs = {}
        Zs = {}
        scores = {}
        for v in seed:
            cand_G = sub_G.copy()
            cand_G.remove_edge('z', v)

            if False:
                _seed = seed.copy()
                _seed.remove(v)
                z = fit_confounder(x.loc[:, tuple(_seed)], M=self.M)
                _x['z'] = z
            new_score = self.score_data_given_graph(_x, cand_G)

            diffs[v] = current_score - new_score
            Zs[v] = z
            scores[v] = new_score
        return diffs, Zs, scores

    def _unconnected_score(self):
        G_empty = nx.DiGraph()
        for node in self.G.nodes:
            G_empty.add_node(node)
        self.unconnected_score = self.score_data_given_graph(self.x, G_empty)

    def _true_score(self):
        z = self.z
        confounded_set = self.confounded_set

        self.true_score = self._conf_score(confounded_set, z)[0]
        self.true_score_no_conf = self._conf_score(set(), z)[0]

    def _GES_score(self):
        self.GES_score = self.score_data_given_graph(self.x, self.G)

    def baseline_scores(self):
        self._unconnected_score()
        self._GES_score()
        self._true_score()

    def _baseline_pr(self):
        confounded_set = self.confounded_set
        if self.confounded_set is not None:
            try:
                prec_orig = len(seed.intersection(confounded_set))/len(seed)
            except ZeroDivisionError:
                prec_orig = np.nan
            rec_orig = len(seed.intersection(confounded_set))/len(confounded_set)

    def confounded_set_candidates(self, smart=True, setup=None, K=10):
        G = self.G
        x = self.x
        confounded_set = self.confounded_set

        # How do our inferred things work?
        self.seeds = {}
        self.orig_scores = {}
        self.fw_best = {}
        self.bw_best = {}
        self.zs = {}
        potential_seeds = [set(self.markov_blanket_from_graph(v)) for v in G.nodes]
        # We sort the seeds by how often they occur in the list in total and then sort in descending frequency
        # potential_seeds = map(set, zip(*sorted(seed_hist.items(), reverse=True, key=lambda x: x[1])))
        # True set and one random sample from the markov blankets since they give very similar scores
        self.num_forward = -1
        self.num_backward = -1
        if self.num_seeds > 0 and self.confounded_set is not None:
            potential_seeds = [set(confounded_set)] + list(random.sample(potential_seeds, self.num_seeds-1))
        elif self.num_seeds > 0 and self.confounded_set is None:
            potential_seeds = list(random.sample(potential_seeds, self.num_seeds))
        elif self.num_backward == 0 and self.confounded_set is not None:
            potential_seeds = [set(confounded_set)] + potential_seeds
        else:
            pass
        # This counts how many seeds we tried so far that actually looked like a confounder would help
        count = 0
        max_num_seeds = 10
        for v, seed in enumerate(potential_seeds):
            if (max_num_seeds > 0) and (count >= max_num_seeds):
                break
            for node in self.forbidden_seeds:
                try:
                    seed.remove(node)
                except KeyError:
                    pass
            size_orig = len(seed)
            conf_score, z = self._conf_score(seed)
            orig_score = conf_score

            score_best = orig_score
            improvement = (score_best < self.GES_score)
            if improvement:
                count += 1
            while improvement and len(seed) <= K:
                self.num_forward += 1
                improvement = False
                diffs, Zs, scores = self.forward_step(seed, z)
                try:
                    v_best, diff_best = max(diffs.items(), key=operator.itemgetter(1))
                except ValueError:
                    break
                improvement = (diff_best > 0)
                if improvement:
                    z = Zs[v_best]
                    score_best = scores[v_best]
                    seed.add(v_best)
            forward_best_score = score_best
            improvement = (score_best < self.GES_score)
            while improvement:
                self.num_backward += 1
                improvement = False
                diffs, Zs, scores = self.backward_step(seed, z)
                try:
                    v_best, diff_best = max(diffs.items(), key=operator.itemgetter(1))
                except ValueError:
                    break
                improvement = (diff_best > 0)
                if improvement:
                    z = Zs[v_best]
                    score_best = scores[v_best]
                    seed.remove(v_best)
            backward_best_score = score_best
            if confounded_set is not None:
                try:
                    prec = len(seed.intersection(confounded_set))/len(seed)
                except ZeroDivisionError:
                    prec = np.nan
                rec = len(seed.intersection(confounded_set))/len(confounded_set)
            size = len(seed)
            self.seeds[v] = seed
            self.orig_scores[v] = orig_score
            self.fw_best[v] = forward_best_score
            self.bw_best[v] = backward_best_score
            self.zs[v] = z
        return self.seeds, self.orig_scores, self.fw_best, self.bw_best

    def iterative_fit(self, k=0):
        G = self.G
        x = self.x
        confounded_set = self.confounded_set
        i = 0
        forbidden_seeds = []
        self.orig_score = curr_GES_score = self.GES_score
        print(self.orig_score)
        while True:
            i += 1
            x_ = x.copy()
            seeds, _, _, bw_best = self.confounded_set_candidates()
            best_seed, best_score, best_z = self.best_seed()
            x_.loc[:, f'z{i}'] = best_z
            self.forbidden_seeds.append(f'z{i}')
            self.x = x_

            self.fit_ges()
            self._GES_score()
            print(curr_GES_score, self.GES_score)
            if self.GES_score < curr_GES_score:
                x = x_
                curr_GES_score = self.GES_score
            else:
                self.x = x
                self.fit_ges()
                self._GES_score()
                break
            if k and (i >= k):
                break

    def best_seed(self):
        key = min(self.orig_scores.items(), key=operator.itemgetter(1))[0]
        return self.seeds[key], self.bw_best[key], self.zs[key]
