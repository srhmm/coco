import math
from math import comb, lgamma, exp

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from sklearn.metrics.cluster import contingency_matrix, expected_mutual_information

from dag_gen import gen_partition
from mi_hyp import emi_mv, emi
from mi_sampling import sampling_mi_entropy, Sampler
from co_confounding_tests import test_confounded_Z_mi_sampling, test_confounded_Z_ami_sampling, \
    test_confounded_exact_mi_sampling
from utils import partition_to_vector, map_to_shifts, shifts_to_map, partition_to_map, confound_partition
from linc.utils_pi import pi_enum


class ResultsIdentifiability:
    def __init__(self):
        self.record = {}
        self.record_detailed = {}

    def init_sub(self, N, mAstar, mBstar, mA, mB, mC): #NA, NB, NC):
        key = self.get_key(N, mAstar, mBstar, mA, mB, mC, False)
        key_detailed = self.get_key(N, mAstar, mBstar, mA, mB, mC, True)

        frame = pd.DataFrame({'mi_star': [], 'mi_indep': [], 'mi_dep': [],
                                             'ami_star': [],'ami_indep': [], 'ami_dep': [],
                                             'emi_star': [],'emi_indep': [], 'emi_dep': [], #'emi2_indep': [], 'emi2_dep': [],
                                             #'emi_condZ_indep': [], 'emi_condZ_dep': [], 'emi_mv_indep': [],
                                             #'emi_mv_dep': [], 'emi_mvCondZ_indep': [], 'emi_mvCondZ_dep': [],
                                             'idtf_mi': [], 'fp_mi': [], 'idtf_ami': [], 'fp_ami': [], 'idtf_sampling':[], 'fp_sampling':[]
                                             })
        if not self.record.__contains__(key):
            self.record[key] = {}
            self.record[key] = frame

        if not self.record_detailed.__contains__(key_detailed):
            self.record_detailed[key] = {}
            self.record_detailed[key_detailed] = frame


    def get_key(self, N, mAstar, mBstar, mA, mB, mC, detailed) :
        NAstar = len(np.unique(mAstar))-1
        NBstar = len(np.unique(mBstar))-1
        NA = len(np.unique(mA))-1
        NC = len(np.unique(mC))-1
        NB = len(np.unique(mB))-1
        if detailed:
            return str(N) +'_' + str(NAstar) + str(NBstar) + str(NC) +'_'+ str(NA) + str(NB) + str(NC)
        return str(N) +'_' + str(NAstar) + str(NBstar) + str(NC)

    def get_result(self, a='idtf_mi', b='fp_mi'):
        self.result = {}
        self.result_detailed = {}
        for key in self.record:
            tp = len(np.where(self.record[key][a] == True)[0])
            fn = len(np.where(self.record[key][a] == False)[0])
            fp = len(np.where(self.record[key][b] == True)[0])
            tn = len(np.where(self.record[key][b] == False)[0])
            self.result[key] = {'dep_tp': tp, 'dep_fn': fn, 'indep_fp': fp, 'indep_tn': tn}

        for key in self.record_detailed:
            tp = len(np.where(self.record_detailed[key][a] == True)[0])
            fn = len(np.where(self.record_detailed[key][a] == False)[0])
            fp = len(np.where(self.record_detailed[key][b] == True)[0])
            tn = len(np.where(self.record_detailed[key][b] == False)[0])
            self.result_detailed[key] = {'dep_tp': tp, 'dep_fn': fn, 'indep_fp': fp, 'indep_tn': tn}

    def update(self, N, mAstar, mBstar, mA, mB, mC, mAindep, mBindep, sampler):

        key = self.get_key(N, mAstar, mBstar, mA, mB, mC, False)
        key_detailed = self.get_key(N, mAstar, mBstar, mA, mB, mC, True)

        contingency_AB = contingency_matrix(mA, mB, sparse=True)
        contingency_AB = contingency_AB.astype(np.float64, copy=False)
        contingency_ABindep = contingency_matrix(mAindep, mBindep, sparse=True)
        contingency_ABindep = contingency_ABindep.astype(np.float64, copy=False)
        contingency_ABstar = contingency_matrix(mAstar, mBstar, sparse=True)
        contingency_ABstar = contingency_ABstar.astype(np.float64, copy=False)

        ami_dep = adjusted_mutual_info_score(mA, mB)
        ami_star = adjusted_mutual_info_score(mAstar, mBstar)
        ami_indep = adjusted_mutual_info_score(mAindep, mBindep)

        mi_dep = mutual_info_score(mA, mB)
        mi_star = mutual_info_score(mAstar, mBstar)
        mi_indep = mutual_info_score(mAindep, mBindep)

        emi_dep = expected_mutual_information(contingency_AB, N)
        emi_star = expected_mutual_information(contingency_ABstar, N)
        emi_indep = expected_mutual_information(contingency_ABindep, N)

        emi2_dep = emi(mA, mB, [0 for _ in range(N)], constrained=False)
        emi2_indep = emi(mAindep, mBindep, [0 for _ in range(N)], constrained=False)

        emi_prior_dep = emi(mA, mB, mC, constrained=True)
        emi_prior_indep = emi(mAindep, mBindep, mC, constrained=True)

        emi_mv_dep = emi_mv(mA, mB, mC, constrained=False)
        emi_mv_indep = emi_mv(mAindep, mBindep, mC, constrained=False)
        emi_mv_prior_dep = emi_mv(mA, mB, mC, constrained=True)
        emi_mv_prior_indep = emi_mv(mAindep, mBindep, mC, constrained=True)

        #idtf_mv = False
        #if emi_dep < emi_mv_prior_dep and (not math.isclose(emi_dep, emi_mv_prior_dep)):
        #    idtf_mv = True

        idtf_ami, _, _ = test_confounded_Z_ami_sampling(mA, mB, sampler)

        fp_ami, _, _ = test_confounded_Z_ami_sampling(mAindep, mBindep, sampler)

        idtf_mi, _, _ = test_confounded_Z_mi_sampling(mA, mB, sampler)

        fp_mi, _, _ = test_confounded_Z_mi_sampling(mAindep, mBindep, sampler)

        idtf_sampling = test_confounded_exact_mi_sampling(mA, mB, sampler)

        fp_sampling =  test_confounded_exact_mi_sampling(mAindep, mBindep, sampler)

        result = [mi_star, mi_indep, mi_dep,
                                                       ami_star, ami_indep, ami_dep,
                                                       emi_star, emi_indep, emi_dep,
                                                       #emi2_indep, emi2_dep, emi_prior_indep, emi_prior_dep,
                                                       #emi_mv_indep, emi_mv_dep, emi_mv_prior_indep, emi_mv_prior_dep,
                                                       idtf_mi, fp_mi, idtf_ami, fp_ami, idtf_sampling, fp_sampling]
        self.record[key].loc[len(self.record[key])] = result
        self.record_detailed[key_detailed].loc[len(self.record_detailed[key_detailed])] = result

def eval_identifiability():
    sampler = Sampler()
    reps = 10

    EXPECT_IDENTIFIABLE = [
        [[[(N, i, j, k) for i in range(k-1)] for j in range(k-1)] for k in range(max(2,np.int64(np.floor(N / 2))))] for N
        in [2, 4, 8, 10, 12]]
    DENSE = [
        [[[(N, i, j, k) for i in range(k, N)] for j in range(k, N)] for k in range(N)] for N
        in [2, 4, 8, 10, 12]]

    res_identifiable = ResultsIdentifiability()
    res_dense = ResultsIdentifiability()
    for res, lst in [(res_identifiable, EXPECT_IDENTIFIABLE), (res_dense, DENSE) ]:
        for n in lst:
            for i in n:
                for j in i:
                    for (N, NA, NB, NC) in j:
                        seed = 0
                        for rep in range(reps):
                            seed += 5
                            Astar, Bstar, C = gen_partition(seed, N, NA), gen_partition(seed + 1, N, NB), gen_partition(seed + 2, N, NC)
                            mAstar = partition_to_map(Astar)
                            mBstar = partition_to_map(Bstar)
                            mC = partition_to_map(C)
                            #mA, mB = confound_partition(Astar, C), confound_partition(Bstar, C)

                            mA, mB = confound_partition(mAstar, mC, N), confound_partition(mBstar, mC, N)

                            shiftsA = len(np.unique(mA)) - 1
                            shiftsB = len(np.unique(mB)) - 1
                            Aindep, Bindep = gen_partition(seed + 3, N, shiftsA), gen_partition(seed + 4, N, shiftsB)
                            mAindep, mBindep = partition_to_map(Aindep), partition_to_map(Bindep)
                            res.init_sub(N, mAstar, mBstar, mA, mB, mC)
                            res.update(N, mAstar, mBstar, mA, mB, mC, mAindep, mBindep, sampler)
    res_identifiable.get_result('idtf_mi', 'fp_mi')
    res_mi = res_identifiable.result
    res_identifiable.get_result('idtf_ami', 'fp_ami')
    res_ami = res_identifiable.result
    return res

def eval_identifiability_quick():
    reps = 10
    EXPECT_IDENTIFIABLE = [
        [[[(N, i, j, k) for i in range(k )] for j in range(k )] for k in range(max(2,np.int64(np.floor(N / 2))))] for N
        in [2, 4, 8, 10, 12]]
    samp = Sampler()
    for n in EXPECT_IDENTIFIABLE:
        for i in n:
            for j in i:
                for (N, NA, NB, NC) in j:
                    seed = 0
                    print(N, NA, NB, NC)
                    for rep in range(reps):
                        seed += 5
                        Astar, Bstar, C = gen_partition(seed, N, NA), gen_partition(seed + 1, N, NB), gen_partition(
                            seed + 2, N, NC)
                        mAstar = partition_to_map(Astar)
                        mBstar = partition_to_map(Bstar)
                        mC = partition_to_map(C)
                        #mA, mB = confound_partition(Astar, C), confound_partition(Bstar, C)
                        mA, mB = confound_partition(mAstar, mC), confound_partition(mBstar, mC)

                        shiftsA = len(np.unique(mA)) - 1
                        shiftsB = len(np.unique(mB)) - 1
                        Aindep, Bindep = gen_partition(seed + 3, N, shiftsA), gen_partition(seed + 4, N, shiftsB)
                        mAindep, mBindep = partition_to_map(Aindep), partition_to_map(Bindep)
                        tp, _, _ = test_confounded_Z_mi_sampling(mA, mB, samp)
                        fp, _, _ = test_confounded_Z_mi_sampling(mAindep, mBindep, samp)
                        if tp and not fp:
                            print("\tSUCCESS", )
                        if tp and fp:
                            print("\tboth")
                        if (not tp):
                            print("\tmiss")



EXPECT_IDENTIFIABLE = [ (0, 2, 2), (0, 2, 5), (0, 2, 8),
        (0, 5, 2),  (0, 5, 8),
        (2, 5, 2),
        (2, 2, 1), (2, 2, 5),
                        (0,0, 3), (0, 1, 3),(0, 2, 3),
                        (0,0,4), (0, 1, 4),(0, 2, 4),
                        (0,0,5), (0, 1, 5),(0, 2, 5)
        ]
EXP_IDENTIFIABLE = [[[(i, j, k) for i in range(k)] for j in range(k)] for k in range(5)]
CASES_CAUSE_DENSE = [(8, 2, 4),(5, 0, 4), (2, 3, 8),(5, 2, 4),(5, 2, 3),  (5, 0, 3), (5, 5, 8), (5, 5, 2), (5, 5, 1), (5, 2, 5), (5,0, 5),(9, 0, 9),(9, 0, 8),(9, 0, 5),(9, 2, 5), (5, 0, 9), (5, 0, 5)]
CASES_EFFECT_DENSE = [(2, 8, 4),(0, 5, 4), (2, 8, 3),(2, 5, 3), (2, 5, 4), (0, 5, 3),  (5, 5, 8),(5, 5, 2), (5, 5, 1), (2, 5, 5), (0, 5, 5),(0, 9, 9),(0, 9, 8),(0, 9, 5),(2, 9, 5), (0, 5, 9), (0, 5, 5)]
CASES_CONFOUNDER_DENSE = [(2, 2, 8), (2, 5, 9), (2, 5, 8),  (0, 2, 9)]
CASES_CAUSE_EFFECT_EQUAL= [(1, 1, 1), (2, 2, 2), (5, 5, 5), (8, 8, 8), (9, 9, 9),
             (0, 0, 0), (1, 1, 0), (2, 1, 0), (9, 9, 0),
             (1, 1, 9), (2, 2, 9), (5, 5, 9), (8, 8, 9),
            (8, 8, 1), (8, 8, 2), (8, 8, 5)]
