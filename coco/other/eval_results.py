from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class EvalCoCo:
    def __init__(self):
        self.record = {}

    def add(self,
            #coco,
            scores_mutual,
            scores_marginal,
            #i, j, pa_i, pa_j,
            truly_confounded: bool, truly_causal: bool,
            test_nm):

        self._update_record(scores_mutual, scores_marginal, truly_confounded, truly_causal,
                            test_nm)

    def _update_record(self, scores, scores_marginal,
                       truly_confounded, truly_causal, key):
        if not self.record.__contains__(key):
            self.record[key] = {}
            self.record[key] = \
                pd.DataFrame({'mi': [scores['mi']], #'mi_shift_vectors': [scores['mishifts']],
                              #'ami': [scores['ami']], #'ami_shift_vectors': [scores['amishifts']],
                              'emi': [scores['emi']],
                              'ent': [scores['ent']], #'ent_shifts': [scores['ent_shifts']],
                              'emp_eent': [scores['emp_eent']], 'emp_emi': [scores['emp_emi']],
                              'exp_cf': [scores['exp_cf']], 'exp_caus': [scores['exp_caus']],
                              'exp_cf2': [scores['exp_cf2']], 'exp_caus2': [scores['exp_caus2']],
                              'ent_rev': [scores['ent_rev']], #'ent_rev_shifts': [scores['ent_rev_shifts']],
                              'joint_shifts': [scores['joint_shifts']], 'indep_shifts': [scores['indep_shifts']],'mss_shifts': [scores['mss_shifts']],
                              'mss_caus':  [scores_marginal['mss_caus']],
                              'indep_shifts_caus':  [scores_marginal['indep_shifts_caus']],
                              'joint_shifts_caus':  [scores_marginal['joint_shifts_caus']],
                              'comp_mi_caus':  [scores_marginal['comp_mi_caus']],
                              'comp_ami_caus':  [scores_marginal['comp_ami_caus']],
                              'comp_ent_caus':  [scores_marginal['comp_ent_caus']],
                              'comp_aent_caus':  [scores_marginal['comp_aent_caus']],
                              'comp_bothent_caus':  [scores_marginal['comp_bothent_caus']],
                              'comp_bothaent_caus':  [scores_marginal['comp_bothaent_caus']],
                              'comp_entpa_caus':  [scores_marginal['comp_entpa_caus']],
                              'comp_entch_caus':  [scores_marginal['comp_entch_caus']],
                              'comp_entami_caus':  [scores_marginal['comp_entami_caus']],
                              'comp_ent_rev_caus':  [scores_marginal['comp_ent_rev_caus']],
                              'cfd': [truly_confounded], 'caus': [truly_causal]})  # td clean this
        else:
            self.record[key].loc[len(self.record[key])] = [scores['mi'], #scores['mishifts'], scores['ami'],
                                                           #scores['amishifts'],
                                                           scores['emi'],
                                                           scores['ent'], #scores['ent_shifts'],
                                                           scores['emp_eent'], scores['emp_emi'],
                                                           scores['exp_cf'], scores['exp_caus'],
                                                           scores['exp_cf2'], scores['exp_caus2'],
                                                           scores['ent_rev'], #scores['ent_rev_shifts'],
                                                           scores['joint_shifts'], scores['indep_shifts'],
                                                           scores['mss_shifts'], scores_marginal['mss_caus'],
                                                           scores_marginal['indep_shifts_caus'],
                                                           scores_marginal['joint_shifts_caus'],
                                                           scores_marginal['comp_mi_caus'],
                                                           scores_marginal['comp_ami_caus'],
                                                           scores_marginal['comp_ent_caus'],
                                                           scores_marginal['comp_aent_caus'],
                                                           scores_marginal['comp_bothent_caus'],
                                                           scores_marginal['comp_bothaent_caus'],
                                                           scores_marginal['comp_entpa_caus'],
                                                           scores_marginal['comp_entch_caus'],
                                                           scores_marginal['comp_entami_caus'],
                                                           scores_marginal['comp_ent_rev_caus'],
                                                           truly_confounded, truly_causal]  # td clean this

    def f1_co(self, test_nm='cocoracle', metric='exp_cf', filter_truly_causal=None):
        df = self.record[test_nm]  # .sort_values(by=[metric])
        if filter_truly_causal is not None:
            df = df[df['caus'] == filter_truly_causal]
        return self._f1_coca(df, 'cfd', metric)

    def f1_ca(self, test_nm='cocoracle', metric='exp_caus', filter_truly_cfd=None):
        df = self.record[test_nm]  # .sort_values(by=[metric])
        if filter_truly_cfd is not None:
            df = df[df['cfd'] == filter_truly_cfd]
        return self._f1_coca(df, 'caus', metric)

    def _f1_coca(self, df, case, metric):
        tp = len(df[(df[case] == True) & (df[metric] == 1)])
        fp = len(df[(df[case] == True) & (df[metric] == 0)])
        fn = len(df[(df[case] == False) & (df[metric] == 1)])
        tn = len(df[(df[case] == False) & (df[metric] == 0)])
        den = (tp + 1 / 2 * (fp + fn))
        if den == 0:
            f1 = 0
        else:
            f1 = tp / den
        return tp, fp, fn, tn, f1

    def plot_co(self, test_nm='cocoracle', metric='mi', filter_truly_causal=None):
        df = self.record[test_nm].sort_values(by=[metric])
        if filter_truly_causal is not None:
            df = df[df['caus'] == filter_truly_causal]
        fig, ax = plt.subplots()
        df['id'] = range(len(df))
        ax.scatter(df['id'], df[metric], c=df['cfd'].map({True: 'green', False: 'red'}))

    def plot_ca(self, test_nm='cocoracle', metric='mi_partitions', filter_truly_cfd=None):
        df = self.record[test_nm].sort_values(by=[metric])
        if filter_truly_cfd is not None:
            df = df[df['cfd'] == filter_truly_cfd]
        fig, ax = plt.subplots()
        df['id'] = range(len(df))
        ax.scatter(df['id'], df[metric], c=df['caus'].map({True: 'green', False: 'red'}))

    def plot_coca(self, test_nm='cocoracle', metric='mi_partitions'):
        self._plot_cao(metric, test_nm, colored_attribute='cfd', secondary_attribute='caus')

    def plot_caco(self, test_nm='cocoracle', metric='mi_partitions'):
        self._plot_cao(metric, test_nm, colored_attribute='caus', secondary_attribute='cfd')

    def _plot_cao(self, metric, test_nm, colored_attribute, secondary_attribute):
        df = self.record[test_nm].sort_values(by=[metric])
        fig, ax = plt.subplots()
        df['id'] = range(len(df))
        categs = df.groupby(secondary_attribute)
        markers = ['x', 'o']
        for (name, cat), marker in zip(categs, cycle(markers)):
            ax.scatter(cat['id'], cat[metric], marker=marker,
                       c=cat[colored_attribute].map({True: 'green', False: 'red'}),
                       label=f'{secondary_attribute}: {name} ({colored_attribute} colored)')
            # sub_categs = df.groupby(colored_attribute)
            # colors = ['green', 'red']
            # for (nname, ccat), color in zip(sub_categs, cycle(colors)):
            # cat = cat[test_nm].sort_values(by=[metric])
            #    ax.scatter(ccat['id'], ccat[metric], marker=marker, c=color,
            #    #c=cat[evaluated_attribute].map({True: 'green', False: 'red'}),
            #       label=f'{secondary_attribute}: {name}, {colored_attribute}: {nname}')
        ax.set_ylabel(metric)
        ax.legend()

    def count_caus(self, metric='ent_pi', test='cocoracle'):
        df = self.record[test]
        df = df.loc[abs(df[metric]) > 0.01]  # todo not np.isclose 0
        return len(np.where(df["caus"] == True)[0])

    def count_acaus(self, metric='ent_pi', test='cocoracle'):
        df = self.record[test]
        df = df.loc[abs(df[metric]) > 0.01]
        return len(np.where(df["caus"] == False)[0])

    def count_record(self, key):
        self.record[key]['sum_shifts'] = self.record[key]['indep_shifts'] + self.record[key]['joint_shifts']

    # TODO update these.
    def _eval(self, cutoff, key, metric):

        df = self.record[key][metric].sort_values(by=['mi'])
        df_upper = df[df['mi'] >= cutoff]
        df_lower = df[df['mi'] < cutoff]
        tp, fp = len(df_upper[df_upper['cfd'] == True]), len(df_upper[df_upper['cfd'] == False])
        tn, fn = len(df_lower[df_lower['cfd'] == True]), len(df_lower[df_lower['cfd'] == False])
        return tp, fp, tn, fn

    def _cases_cf__mi(self, key):
        case = 'cf'
        cases_good = sum([1 if i <= 0 else 0 for i in self.CO_vals[case][key]])
        vals_good = sum([i if i <= 0 else 0 for i in self.CO_vals[case][key]]) / cases_good

        cases = sum([1 if i > 0 else 0 for i in self.CO_vals[case][key]])
        vals = sum([i if i > 0 else 0 for i in self.CO_vals[case][key]]) / cases
        return cases_good, vals_good, cases, vals

    def _cases_overestimated_mi(self, key):  # bad cases
        case = 'ucf'
        cases = sum([1 if i < 0 else 0 for i in self.CO_vals[case][key]])
        return cases, sum([i if i < 0 else 0 for i in self.CO_vals[case][key]]) / cases

    def _estim_f1_CO(self, case='cf', key='coco'):  # 'ucf'
        dic = self.CO_matches[case][key]
        tp, fp, tn, fn = dic['TP'], dic['FP'], dic['TN'], dic['FN']
        den = (tp + 1 / 2 * (fp + fn))
        if den == 0:
            f1 = 0
        else:
            f1 = tp / den
        return f1

    def _estim_f1(self, key, metric):
        best = -np.inf
        arg = None
        for i in range(8):
            tp, fp, tn, fn = self.eval(i / 10, key, metric)
            den = (tp + 1 / 2 * (fp + fn))
            if den == 0:
                f1 = 0
            else:
                f1 = tp / den
            print(i / 10, round(f1, 2), tp, fp)
            if f1 > best:
                best = f1
                arg = i / 10
        return arg, best

    def sort(self, key, metric, sort_val='mi_partitions'):
        self.record[key][metric] = self.record[key][metric].sort_values(
            by=[sort_val])
