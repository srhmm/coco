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
from intervention_types import IvType
from nonlinear_dag import NonlinearDAG
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

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def _sample_dag(dag_simulator, n_variables, dag_density, seed=None):
    """
    Samples a DAG from a specified distribution
    """
    if dag_simulator == "er":
        dag = erdos_renyi_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == "ba":
        dag = barabasi_albert_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == 'complete':
        dag = complete_dag(n_variables)
    else:
        raise ValueError(f"DAG simulator {dag_simulator} not valid optoion")

    count = 0
    if len(cpdag2dags(dag2cpdag(dag))) == 1:
        # Don't sample already solved MECs
        np.random.seed(seed)
        new_seed = int(1000*np.random.uniform())
        dag = _sample_dag(dag_simulator, n_variables, dag_density, new_seed)
        count += 1
        if count > 100:
            raise ValueError(f"Cannot sample a DAG in these settings with nontrivial MEC ({[dag_simulator, n_variables, dag_density]})")

    return dag


def _sample_interventions(n_variables, n_total_environments, sparsity, seed=None):
    np.random.seed(seed)
    if isinstance(sparsity, float):
        sparsity = np.round(n_variables * sparsity).astype(int)
    sampled_targets = [
        np.random.choice(n_variables, sparsity, replace=False)
        for _ in range(n_total_environments)
    ]
    print(len(sampled_targets[0]))
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
    elif data_simulator == "linc":
        np.random.seed(seed)
        domain_seed = int(1000 * np.random.uniform())
        dag = NonlinearDAG(dag.nodes, dag.arcs)
        partitions = [None for _ in range(len(dag.nodes))]
        for node_i in range(len(dag.nodes)):
            for context_j, targets in enumerate(intervention_targets):
                if node_i in targets:
                    pass #TODO

        Xs  = dag.sample_data(sample_size, len(intervention_targets), seed, partitions, 0, iv_type, iv_type, scale=True)

    else:
        raise ValueError(f"Data simulator {data_simulator} not valid optoion")

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
            ["Number of possible DAGs", "MEC size", "MEC total edges", "MEC unoriented edges"],
            ["True orientation rate", "False orientation rate", "Precision", "Recall", 'Average precision', 'Runtime', "sid", "shd"],
        ]
    )

    # Create edge_decisions csv header
    header_edges = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["DAG total edges", "Correct", "Incorrect", "SigCorrect", "SigIncorrect", "SigMissed", "Gain", "SigGain", "Runtime"]
            #-> TP when correctly directed; FP when incorrectly directed; FN when causal but undirected due to insignificance; no TNs here as all edges in the MEC should be directed
        ]
    )
    header_dag = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["TP","TN","FP","FN","shd","sid","runtime"]
        ]
    )

    # Additional identifier for this run (for debugging)
    identifier=  'rebuttal_ctx_nonexhaustive6' #  #_rebuttal_samp2' #  '_rebuttal_ctx3'
    if not os.path.exists('../results/'):
        os.makedirs('../results/')

    # Results: Decisions for orienting edges in a MEC
    write_file = open(f"../results/mec{identifier}.csv", "w+")
    write_file.write(", ".join(header_mec) + "\n")
    write_file.flush()

    # Results: DAG search with unknown mec
    write_file_dagsearch = open(f"../results/dag{identifier}.csv", "w+")
    write_file_dagsearch .write(", ".join(header_dag) + "\n")
    write_file_dagsearch .flush()

    # Debug: Decisions for each edge (only for LINC)
    write_file_edges = open(f"../results/edge_decisions{identifier}.csv", "w+")
    write_file_edges.write(", ".join(header_edges) + "\n")
    write_file_edges.flush()

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
            run_experimental_setting(
                args=args,
                params_index=i + prior_indices,
                write_file=write_file,
                write_file_edges=write_file_edges,
                write_file_dagsearch=write_file_dagsearch,
                **params,
            )
        
        prior_indices += len(params_grid)
    logging.info(f'Complete')


def run_experimental_setting(
    args,
    params_index,
    write_file, write_file_edges, write_file_dagsearch, experiment,
    n_variables,
    n_total_environments,
    sparsity,
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


    if sparsity is not None and sparsity > n_variables:
        logging.info(f"Skipping: sparsity {sparsity} greater than n_variables {n_variables}")
        return

    experimental_params = [
        params_index,
        n_variables,
        n_total_environments,
        sparsity,
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
        rshift=600
        # Get DAG
        true_dag = _sample_dag(dag_simulator, n_variables, dag_density, seed=rep+rshift)
        true_cpdag = dag2cpdag(true_dag)
        mec_size = len(cpdag2dags(true_cpdag))
        total_edges = np.sum(true_dag)
        unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

        # Get interventions
        if intervention_targets is None:
            sampled_targets = _sample_interventions(
                n_variables, n_total_environments, sparsity, seed=rep+rshift
            )
        else:
            sampled_targets = intervention_targets

        # Skipping oracle experiments

        # Sample dataset
        if data_simulator is None:
            return results

        Xs = _sample_datasets(
            data_simulator, sample_size, true_dag, sampled_targets, seed=rep+rshift
        )

        print("*** Rep: ", rep+rshift,"# Contexts: ", n_total_environments,  "# Vars:",  n_variables,
              "# Samples:", sample_size,  "Sparsity", sparsity , "***" )
        #open('competitors_data.txt', 'w').close()
        # Compute empirical results
        for save_name, method_name, mch, hyperparams in get_experiment_methods(
            args.experiment
        ):
            time_st = perf_counter()

            np.save('competitors_data/DAG1.npy',true_cpdag)

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
                    soft_todo = [True]
                #for others, add one environment at a time to the data and discover a DAG over it (following the original implementation)
                for soft in soft_todo:

                    # Discover best DAG in the MEC
                    min_cpdag = mch.get_min_dags(soft)

                    true_orients = np.round(dag_true_orientations(true_dag, min_cpdag), 4)
                    false_orients = np.round(dag_false_orientations(true_dag, min_cpdag), 4)
                    precision = np.round(dag_precision(true_dag, min_cpdag), 4)
                    recall = np.round(dag_recall(true_dag, min_cpdag), 4)

                    sid = SID(true_dag, min_cpdag)
                    shd = SHD(true_dag, min_cpdag)

                    if hasattr(mch, 'pvalues_'):
                        ap = np.round(average_precision_score(true_dag, mch.pvalues_), 4)
                    else:
                        ap = None

                    TP, FP, FN, TN = 0,0,0,0
                    #Print each decision, i.e. for the DAG estimated for all (!) environments, evaluate each edge
                    if n_env == max_env and hasattr(mch, 'min_gains_'):
                        gains = mch.min_gains_
                        sigs = mch.min_sig_
                        min_dag = min_cpdag
                        edges = 0

                        for i in range(len(true_dag)):
                            for j in range(len(true_dag)):
                                tp, fp, tp_sig, fp_sig, fn_sig= 0, 0, 0, 0, 0
                                if true_dag[i][j] == 1:
                                    edges = edges+1
                                    if min_dag[i][j]==1:
                                        TP = TP +1
                                        gain = gains[i][j]
                                        sig = sigs[i][j]
                                        tp = 1
                                        if sigs[i][j] > 0:
                                            tp_sig = 1
                                    else:
                                        FN = FN + 1
                                        gain = gains[j][i]
                                        sig = sigs[j][i]
                                        fp = 1
                                        if sigs[j][i] > 0:
                                            fp_sig = 1
                                        else:
                                            fn_sig = 1
                                    edge_decision = ", ".join(
                                        map(
                                            str,
                                            experimental_params + [
                                                method_name,
                                                soft,
                                                n_env,
                                                rep,
                                                edges,
                                                tp, fp, tp_sig, fp_sig, fn_sig,
                                                gain, sig
                                            ],
                                        )
                                        ) + "\n"
                                    if write:
                                        write_file_edges.write(edge_decision)
                                        write_file_edges.flush()

                                else:
                                    if min_dag[i][j]==1:
                                        FP = FP +1
                                    else:
                                        TN = TN + 1

                    runtime= round(perf_counter() -time_st, 5)
                    #Print precision, recall for the DAG estimated up to environment n_env
                    result = ", ".join(
                        map(
                            str,
                            experimental_params + [
                                method_name,
                                soft,
                                n_env,
                                rep,
                                len(mch.get_min_dags(soft)),
                                mec_size,
                                total_edges,
                                unoriented_edges,
                                true_orients,
                                false_orients,
                                precision,
                                recall,
                                ap,
                                runtime,
                                sid,
                                shd
                            ],
                        )
                    ) + "\n"
                    result_dag_search = ", ".join(
                        map(
                            str,
                            experimental_params + [
                                method_name,
                                soft,
                                n_env,
                                rep,
                                TP,
                                TN,
                                FP,
                                FN,
                                shd,
                                sid,
                                runtime
                            ],
                        )
                    ) + "\n"
                    if write:
                        write_file.write(result)
                        write_file_dagsearch.write(result_dag_search)
                        write_file.flush()
                        write_file_dagsearch.flush()
                    else:
                        results.append(result)
                        #results_dagsearch.append(result_dag_search)

                # Save pvalues
                if not os.path.exists(f'./results/pvalue_mats/{name}/'):
                    os.makedirs(f'./results/pvalue_mats/{name}/')
                if hasattr(mch, 'pvalues_'):
                    np.save(
                        f"./results/pvalue_mats/{name}/{name}_{save_name}_pvalues_params={params_index}_rep={rep}.npy",
                        mch.pvalues_,
                    )
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
