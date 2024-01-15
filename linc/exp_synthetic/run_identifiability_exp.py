# This script was adapted from sparse_shift (see LICENSE)

import argparse
import logging
import itertools
from time import perf_counter

import numpy as np
from cdt.metrics import SID, SHD
from pulp import PulpSolverError
from tqdm import tqdm
from joblib import Parallel, delayed

from exp_synthetic.run_mec import _sample_dag
from intervention_types import IvType
from nonlinear_dag import NonlinearDAG
from pi_tree import is_insignificant
from sparse_shift import (
    sample_cdnod_sim,
    erdos_renyi_dag,
    barabasi_albert_dag,
    complete_dag,
)
from sparse_shift import dag_true_orientations, dag_false_orientations, \
    dag_precision, dag_recall, average_precision_score
from sparse_shift import dag2cpdag, cpdag2dags

import os
import warnings

from utils_pi import pi_group_map

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

def _sample_dag_Y(dag_simulator, n_variables, dag_density, seed=None):
    """
    Samples a DAG around variable Y with at least one parent and child
    """
    # TODO
    if dag_simulator == "er":
        dag = erdos_renyi_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == "ba":
        dag = barabasi_albert_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == 'complete':
        dag = complete_dag(n_variables)
    else:
        raise ValueError(f"DAG simulator {dag_simulator} not valid optoion")

    dag = erdos_renyi_dag(n_variables, 0, seed=seed) #dag_density, seed=seed)
    dag[0][1]=0
    dag[1][0]=1
    dag[2][0]=0
    dag[0][2]=1
    dag[0][3]=0
    dag[3][0]=0
    #TODO
    return dag, 0
    # Make sure Y has at least one parent, child, and other variable
    count = 0
    targets = np.where(dag==1)
    is_valid=False
    if len(targets[0])>0:
        for target in targets[0]:
            unrelated = [x for x in range(n_variables) if
                         (x in np.where(dag[target, :] == 0)[0] and x in np.where(dag[:, target] == 0)[0])]
            is_valid = (1 in dag[target, :] and 1 in dag[:, target] and len(unrelated) > 0)
            if is_valid:
                y = target

    while (not is_valid) and (count <100):
        # Don't sample already solved MECs
        np.random.seed(seed)
        new_seed = int(1000*np.random.uniform())
        dag = _sample_dag(dag_simulator, n_variables, dag_density, new_seed)

        if len(targets[0]) > 0:
            for target in targets[0]:
                unrelated = [x for x in range(n_variables) if
                             (x in np.where(dag[target, :] == 0)[0] and x in np.where(dag[:, target] == 0)[0])]
                is_valid = (1 in dag[target, :] and 1 in dag[:, target] and len(unrelated) > 0)
                if is_valid:
                    y = target
        count += 1
    if count >= 100:
        raise ValueError(f"Cannot sample a DAG in these settings with at least one parent, child, and unrelated variable  ({[dag_simulator, n_variables, dag_density]})")
    print(dag)
    return dag, y


def _sample_interventions_Y(n_variables, n_contexts,
                            n_changes_y,
                            n_changes_covariates,
                            y,
                            seed=None):
    """n_changes_y mechanism changes for a node Y, n_changes_covariates mechanism changes for all other variables"""
    np.random.seed(seed)

    changing_contexts = [(np.random.choice(n_contexts, n_changes_y, replace=False) if (p==y) else np.random.choice(n_contexts, n_changes_covariates, replace=False))
                         for p in range(n_variables)]
    sampled_targets = [[] for _ in range(n_contexts)]
    for x in range(n_contexts):
        for i, n in  enumerate(changing_contexts):
            if x in n:
                sampled_targets[x].append(i)
    return sampled_targets


def _sample_datasets(data_simulator, sample_size, dag, intervention_targets,
                     seed=None, iv_type=IvType.PARAM_CHANGE):
    """
    Samples multi-environment data from a specified distribution
    """
    if data_simulator == "cdnod":
        np.random.seed(seed)
        domain_seed = int(1000 * np.random.uniform())
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
            )
            for i, targets in enumerate(intervention_targets)
        ]
    else:
        raise ValueError(f"Data simulator {data_simulator} not valid option")

    return Xs


def main(args):
    # Determine experimental settings
    from settings import get_experiment_params, get_param_keys

    # Initialize og details
    logging.basicConfig(
        filename="../logging.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(f"NEW RUN:")
    logging.info(f"Args: {args}")
    logging.info(f"Experimental settings:")
    logging.info(get_experiment_params(args.experiment))

    # Create results csv header
    header_mec = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["TPc", "FPc", "FPr" ], #discovering causal variables
            ["TPpi", "TNpi", "FPpi", "FNpi" ], #discovering  partitions
            ["Lpistar","Lpa", "Lch", "Lr", "gain", "sig_gain", "rt"]
        ]
    )


    # Additional identifier for this run (for debugging)
    identifier= '_rebuttal' # '_rff_ilp'
    if not os.path.exists('../results/'):
        os.makedirs('../results/')

    # Results: Decisions for orienting edges in a MEC
    write_file = open(f"../results/identifiability{identifier}.csv", "w+")
    write_file.write(", ".join(header_mec) + "\n")
    write_file.flush()

    # Construct parameter grids
    param_dicts = get_experiment_params(args.experiment)
    prior_indices = 0
    logging.info(f'{len(param_dicts)} total parameter dictionaries')

    for params_dict in param_dicts:
        param_keys, param_values = zip(*params_dict.items())
        params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

        # Iterate over
        logging.info(f'{len(params_grid)} total parameter combinations')

        for i, params in enumerate(params_grid):
            logging.info(f"Params {i} / {len(params_grid)}")
            run_identifiability_setting(
                args=args,
                params_index=i + prior_indices,
                write_file=write_file,
                **params,
            )
        
        prior_indices += len(params_grid)
    logging.info(f'Complete')


def run_identifiability_setting(
    args,
    params_index,
    write_file, experiment,
    n_variables,
    n_total_environments,
    n_changes_Y, n_changes_covariates, #sparsity,
    intervention_targets,
    sample_size,
    dag_density,
    reps,
    data_simulator,
    dag_simulator,
):

    # Determine experimental settings
    from exp_synthetic.settings import get_experiment_methods

    name = args.experiment

    if n_variables < 4 or n_changes_covariates > n_total_environments or n_changes_Y > n_total_environments:
        logging.info(f"Skipping: n_changes {n_changes_covariates} or {n_changes_Y}  greater than n_environments {n_total_environments}")
        print(f"Skipping: n_changes {n_changes_covariates} or {n_changes_Y}  greater than n_environments {n_total_environments}")
        return

    experimental_params = [
        params_index,
        n_variables,
        n_total_environments,
        n_changes_Y, n_changes_covariates,
        intervention_targets,
        sample_size,
        dag_density,
        reps,
        data_simulator,
        dag_simulator,
    ]
    experimental_params = [str(val).replace(", ", ";") for val in experimental_params]

    def _run_rep(rep, write):
        results = []
        rshift=150
        # Get DAG and target node
        true_dag, y = _sample_dag_Y(dag_simulator, n_variables, dag_density, seed=rep+rshift)
        true_cpdag = dag2cpdag(true_dag)
        mec_size = len(cpdag2dags(true_cpdag))
        total_edges = np.sum(true_dag)
        unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

        # Get interventions
        if intervention_targets is None:
            sampled_targets = _sample_interventions_Y(
                n_variables, n_total_environments, n_changes_Y, n_changes_covariates, y, seed=rep+rshift
            )
        else:
            sampled_targets = intervention_targets


        # Sample dataset
        if data_simulator is None:
            return results

        Xs = _sample_datasets(
            data_simulator, sample_size, true_dag, sampled_targets, seed=rep+rshift
        )

        print("*** Rep: ", rep+rshift,"# Contexts: ", n_total_environments,  "# Vars:",  n_variables,
              "# Samples:", sample_size,  "# Changes (Y): ", n_changes_Y ,"# Changes (X): ", n_changes_covariates , "***" )
        #open('competitors_data.txt', 'w').close()
        # Compute empirical results
        for save_name, method_name, mch, hyperparams in get_experiment_methods(
            args.experiment
        ):
            time_st = perf_counter()

            #np.save('competitors_data/DAG1.npy',true_cpdag)

            mch = mch(cpdag=true_cpdag, dag=true_dag,
                      **hyperparams)

            max_env = len(Xs)

            for n_env, X in enumerate(Xs):
                n_env += 1
                mch.add_environment(X)
                soft_todo = [True, False]

                #for LINC, only consider the full data with all environments, and no soft/hard score distinction
                if hasattr(mch, 'maxenv_only'):
                    if mch.maxenv_only and (n_env < max_env):
                        continue
                #    soft_todo = [True]
                #for others, add one environment at a time to the data and discover a DAG over it (following the original implementation)
            #for soft in soft_todo:

                # Discover best DAG in the MEC
                pa = np.where(true_dag[:,y]==1)[0]
                ch = np.where(true_dag[y,:]==1)[0]
                unrel = [x for x in range(n_variables) if x not in pa and x not in ch]#(x in np.where(true_dag[y, :] == 0)[0] and x in np.where(true_dag[:, y] == 0)[0])]

                assert(len(pa)>0 and len(ch)>0 and len(unrel)>0)
                pistar, pistar_score, subset_pi, subset_scores = mch.get_mechanisms(y, [pa, ch, unrel]) #mch.get_min_dags(soft)
                imin  =min(range(len(subset_scores)), key=subset_scores.__getitem__)
                if imin==0: TPc = 1
                else: TPc = 0
                if imin==1: FPc = 1
                else: FPc = 0
                if imin==2: FPr = 1
                else: FPr = 0

                pa_score, pa_pi= subset_scores[0],subset_pi[0]
                ch_score, ch_pi= subset_scores[1],subset_pi[1]
                r_score, r_pi= subset_scores[2],subset_pi[2]
                gain = ch_score - pa_score
                sig_gain = gain

                if is_insignificant(gain, False, False, mdl_gain=False, alpha=0.05):
                    sig_gain = 0

                TPpi, TNpi, FPpi, FNpi= 0,0,0,0

                #Evaluate the partition found for causal parent against ground truth interventions
                context_assignment = pi_group_map(pa_pi, n_total_environments)

                print("-- True: ", sampled_targets,  "Guess: ", context_assignment)

                for ci in range(n_total_environments):
                    for cj in range(n_total_environments):
                        if ci == cj:
                            continue
                        same_group = (((y in sampled_targets[ci]) and (y in sampled_targets[cj]))
                                      or ((y not in sampled_targets[ci]) and (y not in sampled_targets[cj])))
                        same_assigned = (context_assignment[ci] == context_assignment[cj])
                        if same_group:
                            if same_assigned:
                                TPpi = TPpi + 1
                            else:
                                FNpi = FNpi + 1
                        else:
                            if same_assigned:
                                FPpi = FPpi + 1
                            else:
                                TNpi = TNpi + 1

                        #if true_dag[i][j] == 1:
                        #    if min_dag[i][j]==1:
                        #        TP = TP +1
                        #    else:
                        #        FN = FN + 1
                        #else:
                        #    if min_dag[i][j]==1:
                        #        FP = FP +1
                        #    else:
                        #        TN = TN + 1

                runtime= round(perf_counter() -time_st, 5)
                #Print precision, recall for the DAG estimated up to environment n_env
                result = ", ".join(
                    map(
                        str,
                        experimental_params + [
                            method_name,
                            True,
                            n_env,
                            rep,
                            TPc, FPc, FPr,
                            TPpi, TNpi, FPpi, FNpi,
                            pistar_score, pa_score, ch_score, r_score,
                            gain, sig_gain, runtime
                        ],
                    )
                ) + "\n"

                if write:
                    write_file.write(result)
                    write_file.flush()
                else:
                    results.append(result)
                    #results_dagsearch.append(result_dag_search)

        return results

    rep_shift = 0
    if args.jobs is not None:
        results = Parallel(
                n_jobs=args.jobs,
            )(
                delayed(_run_rep)(rep + rep_shift, False) for rep in range(reps)
            )
        for result in np.concatenate(results):
            write_file.write(result)
        write_file.flush()
    else:
        for rep in tqdm(range(reps)):
            try:
                _run_rep(rep + rep_shift, write=True)
            except(PulpSolverError):
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        help="experiment parameters to run",
    )
    parser.add_argument(
        "--jobs",
        help="Number of jobs to run in parallel",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--quick",
        help="Enable to run a smaller, test version",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    main(args)
