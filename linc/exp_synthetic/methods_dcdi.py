import argparse

import numpy as np
import competitors.dcdi_exp.dcdi
from competitors.dcdi_exp.dcdi.main import main

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
class DCDI:
    """
    UTIGSP on data from multiple contexts. (A wrapper for experiments)
    """

    def __init__(self, cpdag, dag):
        self.domains_ = []
        self.cpdag = cpdag  # adj matrix
        self.dag = dag
        self.maxenv_only = True
        np.save('competitors_data/data1.npy', [])
        np.save('competitors_data/data_interv1.npy', [])

    def add_environment(self, interventions):
        i = len(self.domains_)
        k = 1
        if i==0:
            fnm = f'competitors_data/data_{k}.npy'
            np.save(fnm, interventions)
            pass
        else:
            fnm = f'competitors_data/data_interv{k}.npy'

            if i == 1:
                np.save(fnm, interventions)
            else:
                contexts_so_far = np.load(fnm)
                contexts = np.vstack([contexts_so_far, interventions])
                np.save(fnm, interventions)
        self.domains_.append(interventions)

            #np.savetxt(f, interventions, fmt='%1.3f', delimiter=",")

    def get_min_dags(self, soft):
        dag = self._dcdi(self.domains_)
        self.min_dags_ = dag
        return self.min_dags_

    def get_min_cpdag(self, soft):
        cpdag = self.min_dags_# (np.sum(self.min_dags_, axis=0) > 0).astype(int)
        return cpdag

    def _dcdi(self, Xs):
        nvars =Xs[0].shape[1]
        PATH = '/Users/sarah/impls/linc/exp_synthetic/competitors_data/'
        #python3 main.py --train --data-path {path} --num-vars 5 --i-dataset {i} --exp-path expp{ct} --model DCDI-DSF --intervention --intervention-type perfect --intervention-knowledge unknown --reg-coeff 0.5')
        args  = self._dcdi_default_args('DCDI-DSF', PATH, PATH, 1, nvars)
        main(args)


    def _dcdi_default_args(self,  model, data_path, exp_path, i, n_vars):
        args = Namespace(exp_path=exp_path, train=True, random_seed=42, data_path=data_path, i_dataset=i, num_vars=n_vars,train_samples=.8,
                         test_samples=None,num_folds=5,fold=0, train_batch_size=64,
                         num_train_iter=1000000, normalize_data=True, #TODO
                         regimes_to_ignore=None, #TODO
                         test_on_new_regimes=None, #TODO
                         model=model, num_layers=2, hid_dim=16, nonlin='leaky-relu', flow_num_layers=2,flow_hid_dim=16,
                         intervention=True,
                         dcd=None,
                         intervention_type ='perfect',
                         intervention_knowledge='unknown',
                         coeff_interv_sparsity=1e-8,
                         optimizer='rmsprop', lr=1e-3,lr_reinit=None, lr_schedule=None,stop_crit_win=100,reg_coef=0.1,
                         omega_gamma=1e-4, omega_mu=1e-4, mu_init=1e-8,mu_mult_factor=2, gamma_init=0., h_threshold=1e-8,
                         patience=10, train_patience=5, train_patience_post=5, plot_freq=10000,
                         no_w_adjs_log=None, plot_density=None, gpu=False,float=False
                         )
        return args

