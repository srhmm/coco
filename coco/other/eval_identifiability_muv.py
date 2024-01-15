from collections import defaultdict
from statistics import mean, stdev, variance

import numpy as np
import pandas as pd

from co_test_type import CoCoTestType, CoShiftTestType
from co_co import CoCo
from dag_gen import _linearity, _random_gp, _random_poly, _random_nonlinearity, dag_to_mec
from utils import f1_score, data_check_soft, pval_to_map
from dag_confounded import DAGConfounded
from mi_sampling import Sampler

'''
class ResultsIdentifiabilityMuv:
    def __init__(self):
        self.eval_pairs = {}
        self.eval_components = {}

        self.result_pairs = {}
        self.result_components = {}


    def init_entry(self, N, NA, NC, metric, is_oracle):
        key = self.get_key(N, NA, NC)
        if not self.eval_pairs.__contains__(key):
            self.eval_pairs[key] = {}

        if not self.eval_components.__contains__(key):
            self.eval_components[key] = {}
        if not self.eval_pairs[key].__contains__(metric):
            self.eval_pairs[key][metric] = {}

        if not self.eval_components[key].__contains__(metric):
            self.eval_components[key][metric] = {}

        if not self.eval_pairs[key][metric].__contains__(is_oracle):
            self.eval_pairs[key][metric][is_oracle] = pd.DataFrame({'tp': [], 'fp': [], 'tn': [], 'fn': [], 'ttp' : [], 'tfp': [], 'ffp': [],
                              'f1': [] , 'f1b': [], 'f1c': []})

        if not self.eval_components[key][metric].__contains__(is_oracle):
            self.eval_components[key][metric][is_oracle] = pd.DataFrame({'jacc': [], 'ari': [], 'ami': [], 'tp': [], 'fp': [], 'tn': [], 'fn': [], 'f1': [],
                                                                         'tp-adj': [], 'fp-adj': [], 'tn-adj': [], 'fn-adj': [], 'f1-adj': []})

    def get_key(self, N,  NAstar, NC):
        return str(N) + '_' + str(NAstar) + '_' + str(NC)

    def get_result(self, metric, #='mi',
                   is_oracle='oracle'):

        for key in self.eval_components:
            if not self.result_components.__contains__(key):
                self.result_components[key] = {}
            if not self.result_components[key].__contains__(metric):
                self.result_components[key][metric] = {}
            N = len(self.eval_components[key][metric][is_oracle]['ari'])
            tp = sum(self.eval_components[key][metric][is_oracle]['tp'])
            fp = sum(self.eval_components[key][metric][is_oracle]['fp'])
            tn = sum(self.eval_components[key][metric][is_oracle]['tn'])
            fn = sum(self.eval_components[key][metric][is_oracle]['fn'])
            f1 = sum(self.eval_components[key][metric][is_oracle]['f1'])/N
            f1stdev = stdev(self.eval_components[key][metric][is_oracle]['f1'])
            f1sig = variance(self.eval_components[key][metric][is_oracle]['f1'])

            tpa = sum(self.eval_components[key][metric][is_oracle]['tp-adj'])
            fpa = sum(self.eval_components[key][metric][is_oracle]['fp-adj'])
            tna = sum(self.eval_components[key][metric][is_oracle]['tn-adj'])
            fna = sum(self.eval_components[key][metric][is_oracle]['fn-adj'])
            f1a = sum(self.eval_components[key][metric][is_oracle]['f1-adj'])/N

            f1astdev = stdev(self.eval_components[key][metric][is_oracle]['f1-adj'])
            f1asig = variance(self.eval_components[key][metric][is_oracle]['f1-adj'])

            jacc = mean(self.eval_components[key][metric][is_oracle]['jacc']) #/N
            ari = mean(self.eval_components[key][metric][is_oracle]['ari']) #/N
            ami = mean(self.eval_components[key][metric][is_oracle]['ami']) #/N

            #f1 = f1_score(tp, fp, fn)
            #f1a = f1_score(tpa, fpa, fna)

            self.result_components[key][metric][is_oracle] =\
                { 'jacc': jacc, 'ari': ari, 'ami': ami, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'f1': f1, 'f1stdev' : f1stdev, 'f1sig': f1sig,
                 'tp-adj': tpa, 'fp-adj': fpa, 'tn-adj': tna, 'fn-adj': fna, 'f1-adj': f1a, 'f1stdev-adj' : f1astdev, 'f1sig-adj': f1asig
                  }

        for key in self.eval_pairs:
            if not self.result_pairs.__contains__(key):
                self.result_pairs[key] = {}
            if not self.result_pairs[key].__contains__(metric):
                self.result_pairs[key][metric] = {}
            tp = sum(self.eval_pairs[key][metric][is_oracle]['tp'])
            fp = sum(self.eval_pairs[key][metric][is_oracle]['fp'])
            tn = sum(self.eval_pairs[key][metric][is_oracle]['tn'])
            fn = sum(self.eval_pairs[key][metric][is_oracle]['fn'])

            N = len(self.eval_pairs[key][metric][is_oracle]['f1'])
            f1 = sum(self.eval_pairs[key][metric][is_oracle]['f1'])/N
            f1stdev = stdev(self.eval_pairs[key][metric][is_oracle]['f1'])
            f1sig = variance(self.eval_pairs[key][metric][is_oracle]['f1'])

            ttp = sum(self.eval_pairs[key][metric][is_oracle]['ttp'])
            tfp = sum(self.eval_pairs[key][metric][is_oracle]['tfp'])
            ffp = sum(self.eval_pairs[key][metric][is_oracle]['ffp'])
            #f1a = f1_score(tp, fp, fn)
            #f1b = f1_score(ttp+tfp, ffp, fn)
            #f1c = f1_score(ttp, fp, fn)

            self.result_pairs[key][metric][is_oracle] = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'ttp':ttp, 'tfp':tfp, 'ffp':ffp,
                                                   'f1': f1, 'f1stdev': f1stdev, 'f1sig': f1sig }#'f1b': f1b,'f1': f1c}

    def show_by_contexts(self, metric, rng_contexts, n_observed_shifts=None, n_confounder_shifts=None):
        return self._show_by(metric, rng_contexts, 0, None, n_observed_shifts, n_confounder_shifts)

    def show_by_observed_shifts(self, metric, rng_observed_shifts, n_contexts, n_confounder_shifts=None):
        return self._show_by(metric, rng_observed_shifts, 1, n_contexts, None, n_confounder_shifts)

    def show_by_confounder_shifts(self, metric, rng_confounder_shifts, n_contexts, n_observed_shifts=None):
        return self._show_by(metric, rng_confounder_shifts, 2, n_contexts, n_observed_shifts, None)

    def _show_by(self, metric, rng, i_range, n_contexts=None, n_observed_shifts=None, n_confounder_shifts=None):
        frame = pd.DataFrame({'f1': [], 'acc_null': []})
        for r in rng:
            f1s, null_accs = [], []
            for key in self.result_pairs[metric]:
                k = np.int64(key.split('_'))
                if k[i_range] != r:
                    continue
                if n_contexts is not None and k[0] != n_contexts:
                    continue
                if n_observed_shifts is not None and (k[1] != n_observed_shifts):
                    continue
                if n_confounder_shifts is not None and (k[2] != n_confounder_shifts):
                    continue
                f1, null_acc = self.result_pairs[metric][key]['f1'], self.result_pairs[metric][key]['null_acc']
                f1s.append(f1)
                null_accs.append(null_acc)

            res = [None, None]
            if len(f1s) > 0:
                res = [mean(f1s), mean(null_accs)]
            frame.loc[len(frame)] = res
        return frame

    def write(self, write_file, n_contexts, n_shifts_observed, n_shifts_confounders,  metrics,
             methods ): # , metric = 'mi'):

        key = self.get_key(n_contexts, n_shifts_observed, n_shifts_confounders)
        write_file.write("\n"+key)
        for metric in metrics:
            write_file.write("\n\tMETRIC:"+metric)
            for method in methods:
                self.get_result(metric, method)
                write_file.write("\n\t\t"+method+":\t\t\t")
                for i in ['tp','fp',  'tn', 'fn', 'f1', 'f1stdev', 'f1sig']: #, 'f1c']:
                    if self.result_pairs[key][metric][method].__contains__(i):
                        write_file.write("[" + str(i) +"-pairs] " + str(np.round(self.result_pairs[key][metric][method][i], 2)) + "\t")
                write_file.write("\n\t\t\t\t\t\t")
                for i in ['tp-adj', 'fp-adj', 'tn-adj', 'fn-adj', 'f1-adj',  'f1stdev-adj', 'f1sig-adj' ]: #, 'f1c']:
                    if self.result_components[key][metric][method].__contains__(i):
                        write_file.write("[" + str(i) +"] " + str(np.round(self.result_components[key][metric][method][i], 2)) + "\t")
                write_file.write("\n\t\t\t\t\t\t")
                for i in ['tp', 'fp', 'tn', 'fn', 'f1', 'f1stdev', 'f1sig', 'jacc', 'ari', 'ami']: #, 'f1c']:
                    if self.result_components[key][metric][method].__contains__(i):
                        write_file.write("[" + str(i) +"] " + str(np.round(self.result_components[key][metric][method][i], 2)) + "\t")
        write_file.flush()

    def write_final(self, write_file, metrics, methods): # , metric='mi'):
        for method in methods:
            for metric in metrics:
                self.get_result(metric, method)
            write_file.write("\n\n"+method+"\nCONTEXTS_OBSSHIFT_CFDSHIFT")

            for metric in metrics:

                TP, FP, TN, FN = 0, 0, 0, 0
                TPA, FPA, TNA, FNA = 0, 0, 0, 0
                F1A, F1,F1stdev,F1sig, F12, F12stdev,F12sig, F1Astdev, F1Asig = 0, 0, 0, 0, 0, 0, 0, 0, 0
                TP_pairs, FP_pairs, TN_pairs, FN_pairs = 0, 0, 0, 0
                JACC, ARI, AMI, ct = 0, 0, 0, 0
                write_file.write("\n\tMETRIC:" + metric )
                for key in self.result_pairs:
                    write_file.write("\n\t\t\t"+key+"\t")
                    #for i in  [  'f1' , 'f1stdev', 'f1sig']:
                    #    if self.result_pairs[key][metric][method].__contains__(i):
                    write_file.write(f"\t\t[f1-p] {str(np.round(self.result_pairs[key][metric][method]['f1'], 2))} \t+-{str(np.round(self.result_pairs[key][metric][method]['f1stdev'], 2))} ({str(np.round(self.result_pairs[key][metric][method]['f1sig'], 2))}) \t")

                   # write_file.write("\n\t\t\t\t\t")
                    #for i in [ 'f1', 'f1stdev', 'f1sig']:
                    #    if self.result_components[key][metric][method].__contains__(i):
                    write_file.write(f"\t\t[f1-c] {str(np.round(self.result_components[key][metric][method]['f1'], 2))} \t+-{str(np.round(self.result_components[key][metric][method]['f1stdev'], 2))} ({str(np.round(self.result_components[key][metric][method]['f1sig'], 2))}) \t")
                       #     write_file.write("[" + str(i) + "] " + str(
                       #         np.round(self.result_components[key][metric][method][i], 2)) + "\t")
                    #write_file.write("\n\t\t\t\t\t")
                    write_file.write(
                        f"\t\t[f1-a] {str(np.round(self.result_components[key][metric][method]['f1-adj'], 2))} \t+-{str(np.round(self.result_components[key][metric][method]['f1stdev-adj'], 2))} ({str(np.round(self.result_components[key][metric][method]['f1sig-adj'], 2))}) \t")

                    #for i in [  'f1-adj' ,'f1stdev-adj', 'f1sig-adj', 'jacc', 'ari', 'ami']:  # , 'f1c']:
                     #   if self.result_components[key][metric][method].__contains__(i):
                     #       write_file.write("[" + str(i) + "] " + str(
                     #           np.round(self.result_components[key][metric][method][i], 2)) + "\t")
                    # TODO mess
                    TP_pairs += self.result_pairs[key][metric][method]['tp']
                    FP_pairs += self.result_pairs[key][metric][method]['fp']
                    TN_pairs += self.result_pairs[key][metric][method]['tn']
                    FN_pairs += self.result_pairs[key][metric][method]['fn']
                    F1 += self.result_pairs[key][metric][method]['f1']
                    F1sig += self.result_pairs[key][metric][method]['f1sig']
                    F1stdev += self.result_pairs[key][metric][method]['f1stdev']

                    TP += self.result_components[key][metric][method]['tp']
                    FP += self.result_components[key][metric][method]['fp']
                    TN += self.result_components[key][metric][method]['tn']
                    FN += self.result_components[key][metric][method]['fn']
                    F12 += self.result_components[key][metric][method]['f1']
                    F12sig += self.result_components[key][metric][method]['f1sig']
                    F12stdev += self.result_components[key][metric][method]['f1stdev']

                    TPA += self.result_components[key][metric][method]['tp-adj']
                    FPA += self.result_components[key][metric][method]['fp-adj']
                    TNA += self.result_components[key][metric][method]['tn-adj']
                    FNA += self.result_components[key][metric][method]['fn-adj']
                    F1A += self.result_components[key][metric][method]['f1-adj']
                    F1Asig += self.result_components[key][metric][method]['f1sig-adj']
                    F1Astdev += self.result_components[key][metric][method]['f1stdev-adj']

                    JACC += self.result_components[key][metric][method]['jacc']
                    ARI += self.result_components[key][metric][method]['ari']
                    AMI += self.result_components[key][metric][method]['ami']
                    ct = ct + 1
                write_file.write("\n\tOVERALL\n\t\tx_x_x\t[tp-pairs] " + str(TP_pairs) + "\t[fp-pairs] " + str(FP_pairs) + "\t[tn-pairs] " + str(TN_pairs) + "\t[fn-pairs] " + str(FN_pairs) + "\t[f1-pairs] " + str(round(f1_score(TP_pairs, FP_pairs, FN_pairs), 2))+ "\t+-" + str(np.round(F1sig, 2))+ "(" + str(np.round(F1stdev, 2)) + ")"
                + "\n\t\t\t\t[tp-adj] " + str(TPA) + "\t[fp-adj] " + str(FPA) + "\t[tn-adj] " + str(TNA) + "\t[fn-adj] " + str(FNA) + "\t[f1-adj] " + str(np.round(f1_score(TPA, FPA, FNA), 2)) + "\t+-" + str(np.round(F1Asig, 2))+ "("  + str(np.round(F1Astdev, 2))+ ")"
                + "\n\t\t\t\t[tp] " + str(TP)  + "\t[fp] " + str(FP)  + "\t[tn] " + str(TN) + "\t[fn] " + str(FN) + "\t[f1] " + str(np.round(f1_score(TP, FP, FN),2))+ "\t+-" + str(np.round(F12sig, 2))+ "(" + str(np.round(F12stdev, 2))+ ")"+ "\n\t\t\t\t[jacc] "+ str(round(JACC/ct,2)) + "\t[ari] "+ str(round(ARI/ct,2)) + "\t[ami] "+ str(round(AMI/ct,2)) )

                write_file.flush()

        write_file.flush()

    def update(self, dag, n_contexts, n_shifts_observed, n_shifts_confounders, coco_results, methods):

        key = self.get_key(n_contexts, n_shifts_observed, n_shifts_confounders)

        for (metric, coco, cocoracle) in coco_results:
            for method in methods:
                self.init_entry(n_contexts, n_shifts_observed, n_shifts_confounders, metric, method)
            #self.init_entry(n_contexts, n_shifts_observed, n_shifts_confounders, metric, 'oracle')
            #self.init_entry(n_contexts, n_shifts_observed, n_shifts_confounders, metric, 'coco')
            #self.init_entry(n_contexts, n_shifts_observed, n_shifts_confounders, metric, 'cocoracle')
                if method == 'oracle':
                    pairs = coco.eval_oracle_edges(dag, skip_source_nodes=False)
                    components = coco.eval_oracle_graph_cuts(dag)
                elif method == 'cocoracle':
                    pairs = cocoracle.eval_estimated_edges(dag, skip_source_nodes=False)
                    components = cocoracle.eval_estimated_graph_cuts(dag)
                else:
                    pairs = coco.eval_estimated_edges(dag, skip_source_nodes=False)
                    components = coco.eval_estimated_graph_cuts(dag)

                self.eval_pairs[key][metric][method].loc[len(self.eval_pairs[key][metric][method])] = pairs
                self.eval_components[key][metric][method].loc[
                    len(self.eval_pairs[key][metric][method])] = components


def eval_identifiability_muv():

    SHIFT_TEST = CoShiftTestType.PI_KCI
    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    METHODS = ['oracle', 'cocoracle', 'coco']
    FUN_FORMS = [#(_linearity(), "LIN"),
                 (_random_nonlinearity(), "NLIN")
                 ]
    KNOWN_COMPONENTS = True

    sampler = Sampler()
    D_n = 500
    n_nodes = 10
    n_confounders = 2

    reps = 10
    VARY_SHIFTS = [[[(5, 0, 1),(5, 0, 2), (5, 0, 3), (5, 1, 2), (5, 1, 3), (5, 2, 3), (5, 2, 2),(5, 3, 3) ]]]
    VARY_CONTEXTS = [[[(3, 0, 1), (4,0,1), (6, 0, 1), (8, 0, 1), (10, 0, 1), (12, 0, 1) # (6, 0, 3), (8, 0, 4), (10, 0, 5), (12, 0, 6)
                        ]]]
    EXP_IDENTIF = [
        [[(N, i, k)
         for i in range(min(k,max(2,np.int64(np.floor(N / 2)))))]
         for k in range(N-1)]for N in [2, 4, 6, 8]]

    EXP_NIDENTIF = [
        [[(N, i, k) for i in range(k, N)]
        for k in range(N)] for N
        in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]

    dataset_id = 0

    for fun_form, fun_str in FUN_FORMS:

        fnm = str(SHIFT_TEST) + '_' +str(CONFOUNDING_TEST) + '_' + fun_str + '_' + str(D_n)

        write_file = open(f"log_{fnm}.csv", "w+")
        write_final_file = open(f"result_{fnm}.csv", "w+")

        for nm, lst, biv in [
           ('BIVARIATE, inc. confd shifts', VARY_SHIFTS, True),
           ('BIVARIATE, inc. contexts', VARY_CONTEXTS, True),
           ('CONFOUNDED COMPONENTS,  inc. confd shifts', VARY_SHIFTS, False),
           ('CONFOUNDED COMPONENTS,  inc. contexts', VARY_CONTEXTS, False)
            ]:
            BIVARIATE = biv

            for fl in [write_file, write_final_file]:
                fl.write("\n\nEXPERIMENT: " + nm)
                fl.write("\nMETHOD: " + str(CONFOUNDING_TEST) + " x " + str(SHIFT_TEST))
                fl.write("\nFUN_FORM: " + fun_str)
                fl.flush()
            res = ResultsIdentifiabilityMuv()
            for n in lst:
                for i in n:
                    for (n_contexts, n_shifts_observed, n_shifts_confounders) in i:
                        seed = 2
                        for rep in range(reps):
                            print("REP", rep, res.get_key(n_contexts, n_shifts_observed, n_shifts_confounders))
                            seed += 1
                            np.random.seed(seed)


                            list_n_shifts_observed = [0, n_shifts_observed]
                            list_n_shifts_confounders = [n_shifts_confounders]
                            if BIVARIATE:
                                n_components = 1
                                list_n_confounded_nodes = [2]
                                if  rep % 2 == 0: #np.random.uniform(0, 1) < 0.5:
                                    list_n_confounded_nodes = [0] # or empty
                                dag = DAGConfounded(seed, n_contexts, 2, 1, list_n_confounded_nodes,
                                                       list_n_shifts_observed,
                                                       list_n_shifts_confounders, is_bivariate=True
                                                       )

                                D = dag.gen_data(seed, D_n,
                                                       _functional_form=fun_form,
                                                       oracle_partition=False,
                                                       noise_iv=False
                                                       )
                                D_cocoracle = dag.gen_data(seed, D_n,
                                                       _functional_form=fun_form,
                                                       oracle_partition=True,
                                                       noise_iv=False
                                                       )

                            else:
                                if KNOWN_COMPONENTS:
                                    n_components = n_confounders
                                else:
                                    raise NotImplementedError()
                                remaining = n_nodes - n_confounders# + 1
                                list_n_confounded_nodes = []
                                for _ in range(n_confounders):
                                    if remaining == 0:
                                        size = 1
                                    else:
                                        size = np.random.choice([i for i in range(2, remaining + 1)], 1)[0]
                                    list_n_confounded_nodes.append(size)
                                    remaining = remaining - size

                                list_n_shifts_observed = [0, n_shifts_observed]
                                list_n_shifts_confounders = [n_shifts_confounders]

                                dag = DAGConfounded(seed, n_contexts, n_nodes, n_confounders, list_n_confounded_nodes,
                                                    list_n_shifts_observed,
                                                    list_n_shifts_confounders, is_bivariate=False
                                                    )

                                D = dag.gen_data(seed, D_n,
                                                 _functional_form=fun_form,
                                                 oracle_partition=False,
                                                 noise_iv=False
                                                 )
                                D_cocoracle = dag.gen_data(seed, D_n,
                                                       _functional_form=fun_form,
                                                       oracle_partition=True,
                                                       noise_iv=False
                                                       )
                            #for ci in D.shape[0]:
                            write_data = open(f"../jci_data/D_{dataset_id}.csv", "w+")
                            dataset_id += 1
                            write_data.write(data_to_jci(D))
                            #todo metadata

                            coco = CoCo(dag.G, D,
                                           CONFOUNDING_TEST,
                                           SHIFT_TEST,
                                           sampler, n_components, dag = dag
                                        )
                            cocoracle = CoCo(dag.G, D_cocoracle,
                                           CONFOUNDING_TEST,
                                           SHIFT_TEST,
                                           sampler, n_components, dag = dag
                                        )

                            #TODO select n_components.
                            #TODO the best dag acc. to sparse shift:

                            #mec, node_order = dag_to_mec(dag.G)
                            #cocos = [None for _ in range(len(mec))]
                            #for i, dag_i in enumerate(mec):
                            #    results_i = coco_mi.score_dag(dag_i, node_order)
                            #    sim_mi, sim_01, sim_pval, sim_cent, sim_causal_01, sim_causal_pval = results_i

                            #    cocos[i] = results_i
                            #print()

                            ##coco_dag= CoCo(dag, D,  CONFOUNDING_TEST,  SHIFT_TEST,  sampler  )
                            #coco_dag.score ...

                            coco_results = [('mi', coco, cocoracle)
                                            #, ('unknown_dag', ...) ->before using additional ones, double check
                                            ]
                            res.update(dag, n_contexts, n_shifts_observed, n_shifts_confounders, coco_results, METHODS)

                        res.write(write_file, n_contexts, n_shifts_observed, n_shifts_confounders, ['mi'], METHODS)
            res.write_final(write_final_file, ['mi'], METHODS)

        write_file.close()
        write_final_file.close()

'''