import itertools
from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from co_co import CoCo
from co_test_type import CoShiftTestType
from utils import confound_partition
from dag_gen import gen_random_directed_er, gen_partitions, \
    gen_partition, _linearity
from invariance_confounding import gen_data_from_graph
from eval_results import EvalCoCo
from utils import data_check_soft, partition_to_vector, map_to_shifts, shifts_to_map, partition_to_map


def show_causal_cases_coco(re, key='cause_shifts', outer='cause', inner='effect', caus_1='exp_caus', name_1='ours', caus_2='mss_caus', name_2='mss'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Causal direction discovery
    f1_ca_cf = [
        [re[key][n_shifts][n_inner_shifts].f1_ca('cocoracle', caus_1, filter_truly_cfd=True)[4] for n_inner_shifts
         in range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]
    f1_ca_uncf = [
        [re[key][n_shifts][n_inner_shifts].f1_ca('cocoracle', caus_1, filter_truly_cfd=False)[4] for n_inner_shifts
         in range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]
    f1_ca_all = [
        [re[key][n_shifts][n_inner_shifts].f1_ca('cocoracle', caus_1)[4] for n_inner_shifts in
         range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]

    # Causal direction discovery
    f1_mss_ca_cf = [
        [re[key][n_shifts][n_inner_shifts].f1_ca('cocoracle', caus_2, filter_truly_cfd=True)[4] for n_inner_shifts
         in range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]
    f1_mss_ca_uncf = [
        [re[key][n_shifts][n_inner_shifts].f1_ca('cocoracle', caus_2, filter_truly_cfd=False)[4] for n_inner_shifts
         in range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]
    f1_mss_ca_all = [
        [re[key][n_shifts][n_inner_shifts].f1_ca('cocoracle', caus_2)[4] for n_inner_shifts in
         range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]

    df_mss_ca_cf = pd.DataFrame(f1_mss_ca_cf)
    df_mss_ca_uncf = pd.DataFrame(f1_mss_ca_uncf)
    df_mss_ca_all = pd.DataFrame(f1_mss_ca_all)

    df_ca_all = pd.DataFrame(f1_ca_all)
    df_ca_uncf = pd.DataFrame(f1_ca_uncf)
    df_ca_cf = pd.DataFrame(f1_ca_cf)

    df_mss_ca_all.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]
    df_mss_ca_uncf.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]
    df_mss_ca_cf.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]
    df_ca_all.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]
    df_ca_cf.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]
    df_ca_uncf.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]

    plts = [[df_ca_uncf, df_ca_cf], [df_mss_ca_uncf, df_mss_ca_cf]]
    titles = [[f'Caus. Dir. ({name_1}, uncf. cases, w incr. {outer} shifts)',
               f'Caus. Dir. ({name_1}, cf cases, w incr. {outer} shifts)'],
              # 'CausD (with increasing #effect shifts)'],
              [f'Caus. Dir. ({name_2}, uncf. cases, w incr. {outer} shifts)',
               f'Caus. Dir. ({name_2}, cf cases, w incr. {outer} shifts)'], ]
    fig, ax = plt.subplots(2, 2)
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            df = plts[i][j]
            p = sns.lineplot(df, ax=col, legend=(i == 0 and j == 0))
            p.set(ylim=(-0.1, 1.1))
            p.set_title(titles[i][j])

def show_confounded_cases_bycontexts_coco(re_bycontexts, key='confounder_shifts', outer='cause', inner='effect', metric='exp_cf2'):

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_shifts = 2
    n_inner_shifts = 2
    f1_co_causal = [
        [re_bycontexts[ci][key][n_shifts][n_inner_shifts].f1_co('cocoracle', metric, filter_truly_causal=True)[4] #for n_inner_shifts in range(len(re_bycontexts[ci][key][n_shifts]))]
        for n_inner_shifts in range(len(re_bycontexts[0][key][n_shifts]))]        for ci in range(len(re_bycontexts))]
    f1_co_anti = [[ re_bycontexts[ci][key][n_shifts][n_inner_shifts].f1_co('cocoracle', metric, filter_truly_causal=False)[4] # for n_inner_shifts in range(len(re_bycontexts[ci][key][n_shifts]))]
                    for n_inner_shifts in range(len(re_bycontexts[0][key][n_shifts]))] for ci in range(len(re_bycontexts))]
    f1_co_all = [
        re_bycontexts[ci][key][n_shifts][n_inner_shifts].f1_co('cocoracle', metric)[4] #for n_inner_shifts in range(len(re_bycontexts[ci][key][n_shifts]))]
        for ci in range(len(re_bycontexts))]

    df_co_all = pd.DataFrame(f1_co_all)
    df_co_causal = pd.DataFrame(f1_co_causal)
    df_co_anti = pd.DataFrame(f1_co_anti)
    #df_co_all.columns =  [f'{i} {inner} shifts' for i in range(len(re_bycontexts[0][key][0]))]
    #df_co_anti.columns = [f'{i} {inner} shifts' for i in range(len(re_bycontexts[0][key][0]))]
    #df_co_causal.columns = [f'{i} {inner} shifts' for i in range(len(re_bycontexts[0][key][0]))]

    plts = [[df_co_causal, df_co_anti]]
    titles= [[f'Conf. Discov. (ours, causal dir., w incr. {outer} shifts)',f'Conf. Discov. (ours, rev., with incr. {outer} shifts)']]
    fig, ax = plt.subplots(1,2)
    for j, row in enumerate(ax):
        col = row
        i = 0
        # for j, col in enumerate(row):
        df = plts[i][j]
        p = sns.lineplot(df, ax=col)
        p.set(ylim=(-0.1, 1.1))
        p.set_title(titles[i][j])

def show_confounded_cases_coco(re , key='confounder_shifts', outer='cause', inner='effect', metric='exp_cf'):

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    f1_co_causal = [
        [re[key][n_shifts][n_inner_shifts].f1_co('cocoracle', metric, filter_truly_causal=True)[4] for n_inner_shifts in range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]
    f1_co_anti = [
        [re[key][n_shifts][n_inner_shifts].f1_co('cocoracle', metric, filter_truly_causal=False)[4] for n_inner_shifts in range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]
    f1_co_all = [
        [re[key][n_shifts][n_inner_shifts].f1_co('cocoracle', metric)[4] for n_inner_shifts in range(len(re[key][n_shifts]))] for
        n_shifts in range(len(re[key]))]

    df_co_all = pd.DataFrame(f1_co_all)
    df_co_causal = pd.DataFrame(f1_co_causal)
    df_co_anti = pd.DataFrame(f1_co_anti)
    df_co_all.columns =  [f'{i} {inner} shifts' for i in range(len(re[key][0]))]
    df_co_anti.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]
    df_co_causal.columns = [f'{i} {inner} shifts' for i in range(len(re[key][0]))]

    plts = [[df_co_causal, df_co_anti]]
    titles= [[f'Conf. Discov. (ours, causal dir., w incr. {outer} shifts)',f'Conf. Discov. (ours, rev., with incr. {outer} shifts)']]
    fig, ax = plt.subplots(1,2)
    for j, row in enumerate(ax):
        col = row
        i = 0
        # for j, col in enumerate(row):
        df = plts[i][j]
        p = sns.lineplot(df, ax=col)
        p.set(ylim=(-0.1, 1.1))
        p.set_title(titles[i][j])

def show_cases_coco():
    n_contexts = range(10)
    results = [eval_cases_coco(n_c) for n_c in n_contexts]

    for i, n_c in enumerate(n_contexts):
        show_causal_cases_coco(results[i])
        show_confounded_cases_coco(results[i])

def eval_cases_coco(n_c):
    reps = 20
    fun_form = _linearity()
    test = CoShiftTestType.PI_ORACLE
    test_nm = 'coco-vario'

    n_vars = 10


    shifts_outer = range(n_c)
    shifts_inner = range(n_c)
    n_base_shifts = 2

    res_eachn_shifts = {'cause_shifts': [None for _ in range(len(shifts_outer))],
                        'confounder_shifts': [None for _ in range(len(shifts_outer))]}
                       # 'both_shift': [None for _ in range(n_c)]}

    for n_outer_shifts in shifts_outer:

        res_eachn_shifts['cause_shifts'][n_outer_shifts] = [None for _ in range(len(shifts_inner))]
        res_eachn_shifts['confounder_shifts'][n_outer_shifts] = [None for _ in range(len(shifts_inner))]
        for n_inner_shifts in shifts_inner:

            res_cause_shifts = eval_coco(reps, test, test_nm,
                                 fun_form, n_c, n_vars,
                                 n_shifts_causes=n_outer_shifts,
                                 n_shifts_effect=n_inner_shifts,
                                 n_shifts_confounder=n_base_shifts)  #case no_shifts: read off the unconfounded cases in res_confounder_shifts

            res_confounder_shifts = eval_coco(reps, test, test_nm,
                               fun_form, n_c, n_vars,
                               n_shifts_causes=n_inner_shifts,
                               n_shifts_effect=n_base_shifts, #case no shifts? read off res_cause_shifts confounded cases?
                               n_shifts_confounder=n_outer_shifts)

            res_eachn_shifts['cause_shifts'][n_outer_shifts][n_inner_shifts] = res_cause_shifts
            res_eachn_shifts['confounder_shifts'][n_outer_shifts][n_inner_shifts] = res_confounder_shifts

    return res_eachn_shifts


def eval_coco(reps, test, test_nm, fun_form,
              n_c, n_vars, n_shifts_causes, n_shifts_effect, n_shifts_confounder):
    results = EvalCoCo()
    seed = 0
    for it in range(reps):
        results, seed = eval_coco_rep(results, seed, test, test_nm, fun_form,
                                      n_c, n_vars, n_shifts_causes, n_shifts_effect, n_shifts_confounder)
        seed = seed + 1
    return results


def eval_coco_rep(results, seed, test, test_nm, fun_form,
                  n_c, n_vars, n_shifts_causes, n_shifts_effect, n_shifts_confounder):
    # TODO important: also generate the data with the correct partition_size for effect and confounders, for non-oracle case.

    only_oracle = False
    if test is CoShiftTestType.PI_ORACLE:
        only_oracle=True

    np.random.seed(seed)
    M = n_vars  # Num vars
    N = 500
    C = n_c

    partition_size_causes = n_shifts_causes
    partition_size_effect = n_shifts_effect
    partition_size_confounder = n_shifts_confounder

    found_confounder = False

    data_check = True
    trial = 0

    # Set aside one node Z in the DAG that confounds at least two nodes.
    # If no such Z exists, skip this DAG.
    # If such Z exists, consider the children of Z as confounded nodes that should be tested, i.e. keep Z hidden and use only the causal parents of the nodes to estimate a pair(of contexts)wise discrepancy vector, then cmp. mutual info of this vector.
    # .. and consider all other nodes as nonconfounded nodes where we use ALL causal parents to get the pairwise discrepancy vector
    while (not found_confounder or not data_check) and trial < 1000:

        G = gen_random_directed_er(M, seed)
        for z in np.random.permutation(G.nodes):
            is_confounding = (len(list(G.successors(z))) > 1)
            if is_confounding:
                for i, j in itertools.combinations(G.nodes(), 2):
                    if (i == z) or (j == z):
                        continue

                    pa_i = list(G.predecessors(i))
                    pa_j = list(G.predecessors(j))
                    sub_pa_i = [p for p in pa_i if not (p == z) and not (p == j)]
                    sub_pa_j = [p for p in pa_j if not (p == z) and not (p == i)]
                    if (len(sub_pa_i) == 0) or (len(sub_pa_j) == 0):
                        continue
                    Z = z
                    found_confounder = True

        partition_size = partition_size_causes
        partitions = gen_partitions(C, partition_size, M, seed, G)

        # generate below:

        seed = seed + 1
        trial = trial + 1

    if not found_confounder or not data_check:
        raise (ValueError(f"No confounders for this seed: {seed}"))

    # Generate partitions such that Z confounds its children's partitions (used in oracle case)
    # Use data from true causal graph (in nonoracle case)
    # and infer partitions from true causal parents when Z does not confound a pair of variables, and otherwise partitions inferred from all causal variables but Z

    conf = False
    uconf = False

    for i, j in itertools.combinations(G.nodes(), 2):
        if (i == Z) or (j == Z):
            continue

        pa_i = list(G.predecessors(i))
        pa_j = list(G.predecessors(j))

        # Parent Sets
        # -Causal direction unknown
        rev_pa_i = pa_i + [j]
        rev_pa_j = [p for p in pa_j if not (p == i)]

        # -Confounder unobserved
        obs_pa_i = [p for p in pa_i if not (p == Z)]
        obs_pa_j = [p for p in pa_j if not (p == Z)]

        obsrev_pa_i = [p for p in rev_pa_i if not (p == Z)]
        obsrev_pa_j = [p for p in rev_pa_j if not (p == Z)]

        if (len(obs_pa_i) == 0) or (len(obs_pa_j) == 0) or (len(rev_pa_i) == 0) or (len(rev_pa_j) == 0):
            continue  # TODO need some comparison of marginal distributions for this case.

        # Make sure that at least one confounded case is encountered
        truly_confounded = ((i in G.successors(Z)) and (j in G.successors(Z)))
        if truly_confounded:
            if conf:
                continue
            conf = True
        else:
            if uconf:
                continue
            uconf = True

        # True partitions
        pistar_i = partitions[i]

        part_i = partition_to_vector(pistar_i)
        pistar_j = gen_partition(seed, C, partition_size_effect)
        part_j = partition_to_vector(pistar_j)

        partitions[j] = pistar_j
        vec_i = map_to_shifts(part_i)
        vec_j = map_to_shifts(part_j)
        map_i = part_i
        map_j = part_j

        if truly_confounded:
            pi_z = gen_partition(seed, C, partition_size_confounder)
            part_z = partition_to_vector(pi_z)  # part_z = partition_to_vector(partitions[Z])
            partitions[Z] = pi_z
            vec_z = map_to_shifts(part_z)  # [1 if x != y else 0 for k, x in enumerate(part_z) for y in part_z[k + 1:]]

            # - Oracle observed partitions change when not accounting for the confounder
            vec_i = np.maximum(vec_i, vec_z)
            vec_j = np.maximum(vec_j, vec_z)
            map_i, map_j = confound_partition(pistar_i, pi_z), confound_partition(pistar_j, pi_z)

        # - Oracle observed partitions change when not accounting for the effect
        rev_vec_i = np.maximum(vec_i, vec_j)
        rev_vec_j = vec_j  # TODO: does it actually remain the same when we remove parent i?

        rev_map_i, rev_map_j = shifts_to_map(rev_vec_i, C ), shifts_to_map(rev_vec_j, C)#TODO
        # Partitions are all set, so generate the data
        data_check = False
        trials = 0
        while not data_check and trials < 1000:
            X = gen_data_from_graph(N, G, C, partitions, seed,
                                    fun_form)
            # Make sure that data contains no nan/inf
            data_check = data_check_soft(X)
            seed = seed + 1
            trials = trials + 1

        # COCO
        cocoracle = CoCo(X, CoShiftTestType.PI_ORACLE)
        coco = CoCo(X, test)

        # Causal direction
        cocoracle.score_edge(pa_i, i, pa_j, j, vec_i, vec_j, map_i, map_j)
        if not only_oracle:
            coco.score_edge(obs_pa_i, i, obs_pa_j, j)

        # Anticausal direction
        cocoracle.score_edge(rev_pa_j, j, rev_pa_i, i, rev_vec_j, rev_vec_i, rev_map_i, rev_map_j)
        if not only_oracle:
            coco.score_edge(obsrev_pa_j, j, obsrev_pa_i, i)

        scores_mutual = cocoracle.get_scores(pa_i, i, pa_j, j)
        # Compare the causal and anticausal directions
        scores_marginal = cocoracle.compare_directions(i, j, pa_i, pa_j, rev_pa_j, rev_pa_i)

            #i, j, pa_i, pa_j,
        results.add(scores_mutual, scores_marginal, #i, j, pa_i, pa_j,
                    truly_confounded, truly_causal=True,
                    test_nm='cocoracle')

        scores_mutual = cocoracle.get_scores(rev_pa_j, j, rev_pa_i, i)
        scores_marginal = cocoracle.compare_directions(j, i, rev_pa_j, rev_pa_i, pa_i, pa_j) #TODO confusing.

        results.add(scores_mutual, scores_marginal, #j, i, rev_pa_j, rev_pa_i,
                    truly_confounded, truly_causal=False,
                    test_nm='cocoracle')

        if not only_oracle:
            scores_mutual = coco.get_scores(obs_pa_i, i, obs_pa_j, j)
            scores_marginal = coco.compare_directions(i, j, obs_pa_i, obs_pa_j, obsrev_pa_j, obsrev_pa_i)

            results.add(#i, j, obs_pa_i, obs_pa_j,
                    scores_mutual, scores_marginal, truly_confounded, truly_causal=True,
                    test_nm=test_nm)

            scores_mutual = coco.get_scores(obsrev_pa_j, j, obsrev_pa_i, i)
            scores_marginal = coco.compare_directions(j, i, obsrev_pa_j, obsrev_pa_i, obs_pa_i, obs_pa_j)

            results.add(scores_mutual, scores_marginal, #j, i, obsrev_pa_j, obsrev_pa_i,
                    truly_confounded, truly_causal=False,
                    test_nm=test_nm)

    return results, seed
