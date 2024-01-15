import itertools
import random
import operator

import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd
from sklearn import linear_model, metrics

class GES(object):
    def __init__(self, data=None, G=None, **kwargs):
        self.data = data
        self.D = data.shape[1]
        self.N = data.shape[0]
        self.G = G
        self.eps = kwargs.get('eps', 0.1)
        self.K = kwargs.get('K', self.D)
        self.last_improvement = 0
        self.criterion = kwargs.get('criterion', 'compression')
        self.outgoing_only = kwargs.get('outgoing_only', set())

    def _get_data(self, columns):
        try:
            return self.data.loc[:, columns]
        except AttributeError:
            return self.data[:, columns]

    def create_empty_graph(self):
        """Create an empty graph and associate with each node the loss we would get
        if we were to regress it on a zero vector."""
        self.G = nx.DiGraph()
        try:
            for c in self.data.columns:
                self.G.add_node(c)
        except:
            for i in range(self.D):
                self.G.add_node(i)

        for node in self.G.nodes:
            node_data = self._get_data(node)
            self.G.nodes[node]['empty_loss'] = np.mean((node_data - np.mean(node_data))**2)
            self.G.nodes[node]['loss'] = self.G.nodes[node]['empty_loss']


    def linear_regression_score(self, preds, target, alpha=0.0):
        """Regresses the target variable on the potential predictors and returns the loss
        as well as the inferred model."""
        X = self._get_data(preds)
        y = self._get_data(target)
        loss, model = self._linear_regression_score(X, y)
        return loss, model

    @staticmethod
    def _linear_regression_score(X, y, alpha=0.0):
        model = linear_model.Ridge(alpha)
        N, m = X.shape
        model.fit(X, y)

        # BIC score
        loss = metrics.mean_squared_error(y, model.predict(X))
        loss = loss + m/2 * np.log(N)/N

        return loss, model

    def eval_add_edge(self, new_pred, target):
        # TODO: eval_f
        self.eval_f = self.linear_regression_score
        old_loss = self.G.nodes[target]['loss']
        preds = list(self.G.predecessors(target)) + [new_pred]
        new_loss, model = self.eval_f(preds, target)

        diff = old_loss - new_loss
        # If this edge gives the lowest loss so far, update the model associated with the node
        if new_loss < self.min_loss[target] - self.eps:
            self.min_loss[target] = new_loss
            self.G.nodes[target]['model'] = model

        return diff


    def try_add_edge(self, v1, v2):
        self.G.add_edge(v1, v2)
        # If the edge introduces a cycle, remove it again
        try:
            nx.find_cycle(self.G, source=v1)
            self.G.remove_edge(v1, v2)
            success = 0
        except:
            # Update the loss if we add the edge and remove the edge from candidates
            self.G.nodes[v2]['loss'] -= self.improvement[(v1, v2)]
            success = 1
        # No matter whether we added the edge or not, it's no longer a candidate
        # so we can remove everything associated with it...
        try:
            self.poss_pairs.remove((v1, v2))
            self.poss_pairs.remove((v2, v1))
        except ValueError:
            pass
        try:
            del self.improvement[(v1, v2)]
            del self.improvement[(v2, v1)]
        except KeyError:
            pass
        return v2, success

    def _best_key(self):
        if self.criterion == 'compression':
            key, val = max(self.improvement.items(), key=operator.itemgetter(1))
        # elif self.criterion == 'asymmetry':
        #     key, val = max(self.asym.items(), key=operator.itemgetter(1))
        return key, val

    def add_edge_step(self):
        for v1, v2 in self.update_edge_eval:
            try:
                self.improvement[(v1, v2)] = self.eval_add_edge(v1, v2)
            except ValueError:
                pass

        v2 = None
        success = 0
        # We want to add edges only if there's a "significant" improvement
        while self.improvement.values() and (max(self.improvement.values()) > self.eps):
            max_key, last_improvement = self._best_key()
            v2_, success = self.try_add_edge(*max_key)
            # We want to add only one edge at a time
            if success:
                self.last_edge = max_key
                self.last_improvement = last_improvement
                v2 = v2_
                break
        # Only those edges which have the node to which we just added an incoming edge as target
        # need to have their score updated by the decomposability of the score
        try:
            if len(list(self.G.predecessors(v2))) >= self.K:
                self.poss_pairs = [(x, y) for x, y in self.poss_pairs if y != v2]
                self.improvement = {k: self.improvement[k] for k in self.poss_pairs}
        except nx.NetworkXError:
            pass
        self.update_edge_eval = [(x, y) for (x, y) in self.poss_pairs if y == v2]
        return success

    def eval_remove_edge(self, old_pred, target):
        self.eval_f = self.linear_regression_score
        old_loss = self.G.nodes[target]['loss']
        preds = list(self.G.predecessors(target))
        preds.remove(old_pred)
        if not preds:
            new_loss = self.G.nodes[target]['empty_loss']
            model = None
        else:
            new_loss, model = self.eval_f(preds, target)

        diff = old_loss - new_loss
        # If this edge gives the lowest loss so far, update the model associated with the node
        if new_loss < self.min_loss[target] - self.eps:
            self.min_loss[target] = new_loss
            self.G.nodes[target]['model'] = model

        return diff

    def remove_edge(self, v1, v2):
        self.G.remove_edge(v1, v2)
        self.G.nodes[v2]['loss'] -= self.improvement[(v1, v2)]
        self.last_improvement = self.improvement[max_key]
        del self.improvement[(v1, v2)]

    def remove_edge_step(self):
        for v1, v2 in self.G.edges:
            self.improvement[(v1, v2)] = self.eval_remove_edge(v1, v2)

        success = 0
        v2 = None
        try:
            # Remove the edge with the largest improvement
            max_key = max(self.improvement.keys(), key=(lambda k: self.improvement[k]))
            if self.improvement[max_key] > self.eps:
                v2 = self.remove_edge(*max_key)
                success = 1
        except:
            pass

        self.update_edge_eval = [(x, y) for (x, y) in self.poss_pairs if y == v2]
        return success

    def update_poss_pairs(self, new_nodes):
        """Updates the possible pairs after we have added new nodes, due to, say, confounders. Since we only allow confounders rather than mediators,
        we add only edges outgoing from the new nodes but no incoming ones."""
        pp = self.poss_pairs
        # Allow only for outgoing edges from the confounders, not incoming edges...
        pp = pp + itertools.product(new_nodes, self.G.nodes)
        self.poss_pairs = [e for e in pp if e not in self.G.edges]

    def learn_structure(self):
        poss_pairs = itertools.product(self.G.nodes, self.G.nodes)
        # No self-loops
        poss_pairs = [(v1, v2) for v1, v2 in poss_pairs if v1 != v2]
        # No need to consider edges we already know to be there
        poss_pairs = [e for e in poss_pairs if e not in self.G.edges]
        poss_pairs = [(v1, v2) for v1, v2 in poss_pairs if v2 not in self.outgoing_only]

        self.poss_pairs = poss_pairs

        self.update_edge_eval = self.poss_pairs
        self.last_edge = None

        self.improvement = dict()
        self.min_loss = {node: self.G.nodes[node]['loss'] for node in self.G.nodes}

        # Add edges
        while True:
            success = self.add_edge_step()
            yield self.G
            if not success:
                break
        self.G_add = self.G
        self.improvement = dict()

        # Remove edges
        i = 0
        while True:
            i = i+1
            yield self.G
            try:
                success = self.remove_edge_step()
            except:
                success = False
            if not success:
                break
