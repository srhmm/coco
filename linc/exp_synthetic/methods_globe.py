import numpy as np
from pulp import PulpSolverError

import upq
from competitors.globe.globeWrapper import GlobeWrapper
from out import Out
from pi_mec import PiMEC
from sparse_shift import cpdag2dags
from pi_tree import PiTree
from vsn import Vsn


class GLOBE:
    """
    GLOBE on pooled data from multiple contexts. (A wrapper for experiments)
    """

    def __init__(self, cpdag, dag):
        self.domains_ = []
        self.cpdag = cpdag  # adj matrix
        self.dag = dag
        self.min_gains_, self.min_sig_ = np.zeros((len(cpdag), len(cpdag))),np.zeros((len(cpdag), len(cpdag)))
        self.maxenv_only = True
        open('competitors_data/globe_data.txt', 'w').close()


    def add_environment(self, interventions):
        self.domains_.append(interventions)
        with open('competitors_data/globe_data.txt', 'a') as f:
            np.savetxt(f, interventions, fmt='%1.3f', delimiter=",")

    def get_min_dags(self, soft):
        dag = self._globe(self.domains_)
        self.min_dags_ = dag
        #self.min_gains_, self.min_sig_ = mec.conf_mat(dag)
        return self.min_dags_

    def get_min_cpdag(self, soft):
        cpdag = self.min_dags_# (np.sum(self.min_dags_, axis=0) > 0).astype(int)
        return cpdag

    def _globe(self, Xs):
        filename = 'competitors_data/globe_data.txt'
        Max_Interactions = 2;  # See the Instantiation section of the publication
        log_results = True;  # Set this to true if you would like to store the log of the experiment to a text file
        verbose = True;  # Set this to true if you would like see the log output printed to the screen

        globe = GlobeWrapper(Max_Interactions, log_results, verbose);

        globe.loadData(filename);
        network = globe.run();

        print(network);

        return network
