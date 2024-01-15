from methods_globe import GLOBE
from exp_synthetic.methods_dcdi import DCDI
#from exp_synthetic.methods_globe import GLOBE
from exp_synthetic.methods_utigsp import UTIGSP
from sparse_shift import MinChange, FullMinChanges, ParamChanges
from exp_synthetic.methods_linc import LINC

BASE_C = [5]
BASE_C_dag = [3]
BASE_s = [1]
BASE_n = [6]
BASE_p = [0.3]
BASE_d = [500]
BASE_rep = [20]
BASE_sim_data= ['cdnod']
BASE_sim_dag= ['er']
PARAMS_DICT = {
"dag": [
    {
            "experiment": ["sparsity"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C_dag,
            "sparsity": [0,1,2,3,4,5,6],
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
            "experiment": ["dag_density"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C_dag,
            "sparsity": BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": [0.3,0.5,0.7],
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
            "experiment": ["n_variables"],
            "n_variables": [3,6,9,12],
            "n_total_environments": BASE_C_dag,
            "sparsity": BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        }
],
"mec": [
    {
            "experiment": ["sparsity"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C,
            "sparsity": [0,1,2,3,4,5,6],
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
            "experiment": ["dag_density"],
            "n_variables": BASE_n,
            "n_total_environments": BASE_C,
            "sparsity": BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": [0.3,0.5,0.7],
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        },
        {
            "experiment": ["n_variables"],
            "n_variables": [3,6,9,12],
            "n_total_environments": BASE_C,
            "sparsity": BASE_s,
            'intervention_targets': [None],
            "sample_size": BASE_d,
            "dag_density": BASE_p,
            "reps": BASE_rep,
            "data_simulator": BASE_sim_data,
            "dag_simulator": BASE_sim_dag
        }
],
"pairwise_power_run": [
        #running for LINC rff gain unknown mec:
        {
            "experiment": ["sparsity"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [2,3,4,5,6],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["dag_density"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["n_variables"],
            "n_variables": [3, 6, 9, 12],
            "n_total_environments": [3],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },],
    "pairwise_power_running": [
        #running for LINC rff gain unknown mec:
        {
            "experiment": ["sparsity"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [2,3,4,5,6],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["dag_density"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["n_variables"],
            "n_variables": [3, 6, 9, 12],
            "n_total_environments": [3],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
],
    "pairwise_power_other":[
        {
            "experiment": ["dag_density"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["n_variables"],
            "n_variables": [3, 6, 9, 12],
            "n_total_environments": [1],
            "sparsity": [0],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    "pairwise_power_next": [
        {
            "experiment": ["sparsity"],
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [2,4,6],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["n_variables"],
            "n_variables": [3, 9, 12], #can skip 6
            "n_total_environments": [5],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [10],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    "pairwise_power_gp_done": [
{
            "experiment": ["dag_density"],
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["sparsity"],
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [ 3,  5], #can skip 1 #[2,4]
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["n_variables"],
            "n_variables": [3, 9, 12], #can skip 6
            "n_total_environments": [5],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [10],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    "pairwise_power_todo": [
        {
            "experiment": ["n_total_environments"],
            "n_variables": [6],
            "n_total_environments": [1,2,3,4,6,7,8,9,10,11,12,13,14,15],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [5],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["sample_size"],
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [50, 100, 200, 1000, 2000],
            "dag_density": [0.3],
            "reps": [10],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    "pairwise_power_spars1_c3": [
        {
            "experiment": ["dag_density"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [50],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
         {
             "experiment": ["n_variables"],
             "n_variables": [3, 6, 9, 12],
             "n_total_environments": [3],
             "sparsity": [1],
             'intervention_targets': [None],
             "sample_size": [500],
             "dag_density": [0.3],
             "reps": [50],
             "data_simulator": ['cdnod'],
             "dag_simulator": ["er"],
         },
            {
            "experiment": ["n_total_environments"],
            "n_variables": [6],
            "n_total_environments":  [1,3,5,8,10,12,15], #TODO [2,4,6,7,9,11,13,14]  # [1,2,3,4,5,6,7,8,9,10,11,12,14,14,15],
            "sparsity": [1],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        # {
        #     "experiment": "sample_size",
        #     "n_variables": [6],
        #     "n_total_environments": [3],
        #     "sparsity": [1/3],
        #     'intervention_targets': [None],
        #     "sample_size": [50, 100, 200, 500, 1000, 2000],
        #     "dag_density": [0.3],
        #     "reps": [5], #[40],
        #     "data_simulator": ['cdnod'],
        #     "dag_simulator": ["er"],
        # },
        ],
    "pairwise_power_large": [

        {
            "experiment": ["sparsity"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1, 2, 3, 4, 5],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [25],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["n_total_environments"],
            "n_variables": [6],
            "n_total_environments":  [1,3,5,8,10,12,15],# [1,2,3,4,5,6,7,8,9,10,11,12,14,14,15],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [10],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["sample_size"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [50, 100, 200, 500, 1000, 2000],
            "dag_density": [0.3],
            "reps": [10],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    "pairwise_power_all": [
        {
            "experiment": ["sparsity"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1, 2, 3, 4, 5],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment":[ "dag_density"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment":[ "n_variables"],
            "n_variables": [3, 6, 9, 12],
            "n_total_environments": [3],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment":[ "sample_size"],
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [50, 100, 200, 500, 1000],
            "dag_density": [0.3],
            "reps": [5],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "experiment": ["n_total_environments"],
            "n_variables": [6],
            "n_total_environments":  [1,3,5,8,10,12,15],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [5],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    # "environment_convergence": [{
    #     "n_variables": [6],
    #     "n_total_environments": [10],
    #     "sparsity": [1, 2, 4],
    #     'intervention_targets': [None],
    #     "sample_size": [500],
    #     "dag_density": [0.3],
    #     "reps": [20],
    #     "data_simulator": ["cdnod"],
    #     "dag_simulator": ["er"],
    # }],
    # "soft_samples": [{
    #     "n_variables": [6],
    #     "n_total_environments": [5],
    #     "sparsity": [1, 2, 3, 4, 5, 6],
    #     'intervention_targets': [None],
    #     "sample_size": [50, 100, 200, 300, 500],
    #     "dag_density": [0.3],
    #     "reps": [20],
    #     "data_simulator": ["cdnod"],
    #     "dag_simulator": ["er"],
    # }],
    "oracle_rates": [{
        "n_variables": [4, 6, 8, 10, 12],
        "n_total_environments": [5],
        "sparsity": [1/5, 1/3, 1/2, 2/3, 4/5],
        'intervention_targets': [None],
        "sample_size": [None],
        "dag_density": [0.1, 0.3, 0.5, 0.7, 0.9],
        "reps": [5], #[20],
        "data_simulator": [None],
        "dag_simulator": ["er"],
    }],
    "oracle_select_rates": [
        {
            "n_variables": [6, 8, 10, 12],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [5], #[20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [1, 2, 3, 4, 5, 6, 7],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [5], #[20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [15],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [5], #[20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3, 0.5, 0.7, 0.9, 0.1],
            "reps": [5], #[20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            # Since 'ba' can't handle all of the same settings as 'er'
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3, 0.5, 0.7, 0.9],
            "reps": [5], #[20],
            "data_simulator": [None],
            "dag_simulator": ["ba"],
        },
    ],
    "bivariate_power": [{
        "n_variables": [2],
        "n_total_environments": [2],
        "sparsity": [None],
        'intervention_targets': [
            [[], [0]],
            [[], [1]],
            [[], []],
            [[], [0, 1]],
        ],
        "sample_size": [500],
        "dag_density": [None],
        "reps": [50],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["complete"],
    }],
    "bivariate_multiplic_power": [{
        "n_variables": [2],
        "n_total_environments": [2],
        "sparsity": [None],
        'intervention_targets': [
            [[], [0]],
            [[], [1]],
            [[], []],
            [[], [0, 1]],
        ],
        "sample_size": [500],
        "dag_density": [None],
        "reps": [50],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["complete"],
    }]
}


# save name, method name, algo, hpyerparams
ALL_METHODS = [
    (
        'mch_kci',
        'Min changes (KCI)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {
                "KernelX": "GaussianKernel",
                "KernelY": "GaussianKernel",
                "KernelZ": "GaussianKernel",
            },
        }
    ),
    (
        'mch_lin',
        'Min changes (Linear)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'invariant_residuals',
            'test_kwargs': {'method': 'linear', 'test': "whitney_levene"},
        }
    ),
    (
        'mch_gam',
        'Min changes (GAM)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'invariant_residuals',
            'test_kwargs': {'method': 'gam', 'test': "whitney_levene"},
        }
    ),
    (
        'mch_fisherz',
        'Min changes (FisherZ)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'fisherz',
            'test_kwargs': {},
        }
    ),
    (
        'full_pc_kci',
        'Full PC (KCI)',
        FullMinChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {},
        }
    ),
    (
        'mc',
        'MC',
        ParamChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
        }
    )
]

METHODStest= [(
        'dcdi', #'utigsp',#'linc_gp_nogain',#
        'dcdi', #'utigsp', #'LINC_GP_nogain',#
        DCDI,
        {
        },)]
METHODS_MEC = [(
        'globe',
        'globe',
        GLOBE,
        {
    },
),]
METHODS_DAG= [(
        'linc_rff_dag',
        'linc_rff_dag',
        LINC,
        {
            'rff': True,
            'mdl_gain': True, 'pi_search' : False,
            'ILP': True,
            'known_mec': False #important!
        },
    ),
]
METHODS_MEC2= [(
        'linc_gp_nogain',
        'linc_go_nogain',
        LINC,
        {
            'rff': False,
            'mdl_gain': False, 'pi_search' : True,
            'ILP': True,
            'known_mec' : True,
        },
    ),
(
        'linc_rff_nogain',
        'linc_rff_nogain',
        LINC,
        {
            'rff': True,
            'mdl_gain': False, 'pi_search' : True,
            'ILP': True,
            'known_mec' : True
        },
    ),
]
METHODS_DICT = {
    "dag": METHODS_DAG, #ALL_METHODS,
    "mec": METHODS_MEC, #ALL_METHODS,
    "quick": ALL_METHODS,
    #"pairwise_power": ALL_METHODS,
    #"pairwise_power_quick": ALL_METHODS,
    # "environment_convergence": ALL_METHODS,
    # "soft_samples": ALL_METHODS,
    #"oracle_rates": [],
    #"oracle_select_rates": [],
    #"bivariate_power": ALL_METHODS,
    #"bivariate_multiplic_power": ALL_METHODS,
    #     (
    #         'mch_kcd',
    #         'KCD',
    #         MinChange,
    #         {
    #             'alpha': 0.05,
    #             'scale_alpha': True,
    #             'test': 'fisherz',
    #             'test_kwargs': {'n_jobs': -2, 'n_reps': 100},
    #         }
    #     ),
    # ],
}


def get_experiment_params(exp):
    return PARAMS_DICT[exp]


def get_param_keys(exp):
    return list(PARAMS_DICT[exp][0].keys())


def get_experiments():
    return list(PARAMS_DICT.keys())


def get_experiment_methods(exp):
    return METHODS_DICT[exp]
