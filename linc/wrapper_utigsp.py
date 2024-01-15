import cdt
from causaldag import rand
from causaldag import unknown_target_igsp
import causaldag as cd
import numpy as np
import random
from causaldag import partial_correlation_suffstat, partial_correlation_test
from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester, MemoizedCI_Tester
from graphical_model_learning import gsp
from statistics import mean, stdev
from function_types import FunctionType
from gen_context_data import gen_context_data
from intervention_types import IvType
from out import Out


def arc_to_adj(arcs, node_n):
    adj = np.zeros((node_n, node_n))
    for (i,j) in arcs:
        adj[i][j] = 1
    return adj

def test_utigsp(iters = 20, C_n = 5, D_n = 500, node_n= 5,
    fun_type = FunctionType.GP_PRIOR,
    iv_type  = IvType.PARAM_CHANGE,
    initial_seed = 100, iv_per_node=[1,1],  out :Out = None
    ):
    assert iters >=2

    sids, sids_cpdag = [0 for _ in range(iters)], [0 for _ in range(iters)]
    shds, shds_cpdag = [0 for _ in range(iters)], [0 for _ in range(iters)]

    cts = [(0,0,0,0) for _ in range(iters)]
    for i in range(iters):
        seed = i + initial_seed
        sid, shd, sidc, shdc,  tps, fps, tns, fns = test_utigsp_sub(C_n, D_n, node_n , fun_type, iv_type, seed, iv_per_node)
        sids[i], sids_cpdag[i] = sid, sidc
        shds[i], shds_cpdag[i] = shd, shdc
        cts[i] = (tps, fps, tns, fns)

    print ("UTIGSP ", C_n, ",",D_n, ",", node_n,  ",",fun_type,  ",",iv_type , "seeds: ", initial_seed,  "-", initial_seed + iters, ", iv per node:", iv_per_node[0] ,"-", iv_per_node[1],
    "\n\tSHD:", np.mean(shds) , #"\tcpdag:",  np.mean(shds_cpdag) ,
           ":\n\tSID:", np.mean(sids) ,# "\tcpdag:" , np.mean(sids_cpdag) ,
    "\n\tTP:", np.mean([ct[0] for ct in cts]), " per DAG \t=", sum([ct[0] for ct in cts]), "total,",
    "\n\tFP:", np.mean([ct[1] for ct in cts]), " per DAG \t= ", sum([ct[1] for ct in cts]), "total,",
    "\n\tTN:", np.mean([ct[2] for ct in cts]), " per DAG \t= ", sum([ct[2] for ct in cts]), "total,",
    "\n\tFN:", np.mean([ct[3] for ct in cts]), " per DAG \t= ", sum([ct[3] for ct in cts]), "total",
           )
    if out is not None:
        out.printto("UTIGSP ", C_n, ",",D_n, ",", node_n,  ",",fun_type,  ",",iv_type , "seeds: ", initial_seed,  "-", initial_seed + iters, ", iv per node:", iv_per_node[0] ,"-", iv_per_node[1],
    "\n\tSHD:", np.mean(shds) , #"\tcpdag:",  np.mean(shds_cpdag) ,
           ":\n\tSID:", np.mean(sids) ,# "\tcpdag:" , np.mean(sids_cpdag) ,
    "\n\tTP:", np.mean([ct[0] for ct in cts]), " per DAG \t=", sum([ct[0] for ct in cts]), "total,",
    "\n\tFP:", np.mean([ct[1] for ct in cts]), " per DAG \t= ", sum([ct[1] for ct in cts]), "total,",
    "\n\tTN:", np.mean([ct[2] for ct in cts]), " per DAG \t= ", sum([ct[2] for ct in cts]), "total,",
    "\n\tFN:", np.mean([ct[3] for ct in cts]), " per DAG \t= ", sum([ct[3] for ct in cts]), "total",
           )

    i = iv_per_node[1] + 0.5
    z = 1.96
    fct = node_n * (node_n - 1)
    sid_norm = np.array([sid / fct for sid in sids])
    print("\t(" + str(i) + "," + str(round(mean(sid_norm), 3)) + ")",  # "\t-var:", round(variance(sid_norm), 3),
              "\t+=(" + str(i) + ", " + str(round(z * stdev(sid_norm) / np.sqrt(len(sids)), 3)) + ")",
              "\t-=(" + str(i) + ", " + str(round(z * stdev(sid_norm) / np.sqrt(len(sids)), 3)) + ")")
    return sids, sids_cpdag, shds, shds_cpdag, cts

def test_utigsp_sub(C_n, D_n , node_n, fun_type, iv_type, seed, iv_per_node):
    Dc, dag, Gc, \
    target, parents, children, confounder, \
    partitions_X, observational_X = gen_context_data(C_n, D_n, node_n, seed, fun_type, iv_type,
                                                     iv_type_covariates=iv_type, iid_contexts=False, iv_per_node=iv_per_node, #[1,1],
                                                     partition_search=False) #TODO make sure context 1 is observational

    obs_samples = Dc[0]
    iv_samples_list = [Dc[i] for i in range(1, C_n)]
    obs_suffstat = partial_correlation_suffstat(obs_samples)
    invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)

    # Create conditional independence tester and invariance tester
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    # Run UT-IGSP
    n_settings  = C_n - 1
    nodes = set(range(node_n))
    setting_list = [dict(known_interventions=[]) for _ in range(n_settings)]
    est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)

    true_adj = arc_to_adj(dag.arcs, node_n)
    est_adj = arc_to_adj(est_dag.arcs, node_n)

    sid = cdt.metrics.SID(true_adj, est_adj)
    sid_cpdag = cdt.metrics.SID_CPDAG(true_adj, est_adj)
    shd = cdt.metrics.SHD(true_adj, est_adj)
    shd_cpdag = cdt.metrics.SHD_CPDAG(true_adj, est_adj)

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(node_n):
        for j in range(node_n):
            if true_adj[i][j] == 1:
                if est_adj[i][j] == 1:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if est_adj[i][j] == 1:
                    fp = fp + 1
                else:
                    tn = tn + 1

    return sid, shd, sid_cpdag, shd_cpdag, tp, fp, tn, fn



def test_gsp():
    np.random.seed(12312)
    nnodes = 5
    nodes = set(range(nnodes))
    dag = rand.directed_erdos(nnodes, .5)
    gdag = rand.rand_weights(dag)
    samples = gdag.sample(100)
    suffstat = partial_correlation_suffstat(samples)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
    est_dag = gsp(nodes, ci_tester)
    dag.shd_skeleton(est_dag)

# The example given in causaldag, https://uhlerlab.github.io/causaldag/utigsp.html
def test_utigsp1():
    # Generate a random graph
    nnodes = 10
    nodes = set(range(nnodes))
    exp_nbrs = 2
    d = cd.rand.directed_erdos(nnodes, exp_nbrs / (nnodes - 1))
    g = cd.rand.rand_weights(d)

    # Choose random intervention targets
    num_targets = 2
    num_settings = 2
    targets_list = [random.sample(nodes, num_targets) for _ in range(num_settings)]
    print(targets_list)

    # Generate observational data
    nsamples_obs = 1000
    obs_samples = g.sample(nsamples_obs)

    # Generate interventional data
    iv_mean = 1
    iv_var = .1
    nsamples_iv = 1000
    ivs = [{target: cd.GaussIntervention(iv_mean, iv_var) for target in targets} for targets in targets_list]
    iv_samples_list = [g.sample_interventional(iv, nsamples_iv) for iv in ivs]


    # Form sufficient statistics
    obs_suffstat = partial_correlation_suffstat(obs_samples)
    invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)


    # Create conditional independence tester and invariance tester
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    # Run UT-IGSP
    setting_list = [dict(known_interventions=[]) for _ in targets_list]
    est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
    print(est_targets_list)