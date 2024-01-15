from experiments.exp_coco.method_types import all_coco_methods, MethodType
from experiments.exp_coco.run_coco import run_coco


def example_small(path):
    METHODS = [MethodType.MSS]#all_coco_methods()

    # pattern: (n_observed_vars, n_latent_vars, n_contexts, n_shifts_observed, n_shifts_latent)
    # vary the latent shifts
    case1 = (5, 1, 6, 1, 1, 'n_Zshift_1')
    #case2 = (5, 1, 6, 1, 2, 'n_Zshift_1')
    #case3 = (5, 1, 6, 1, 3, 'n_Zshift_1')
    reps = 2
    show_result = ""

    CASES = {'5_1_6_1_i' : {'1': case1 }} #,  '2': case2, '3': case3}}

    res = run_coco(reps, CASES, METHODS, path=path)
    for identifier in CASES:
        res.write_methods_tex(identifier, path + "/tex_coco", CASES[identifier], METHODS)
