def example_gen_context_data():
    from gen_context_data import gen_context_data
    from function_types import FunctionType
    from intervention_types import IvType
    import matplotlib.pyplot as plt
    import numpy as np
    pi = [[0,1], [2,3],[4]]
    Dc, G, Gc, target, parents, children, confounder, partitions_X, observational_X = gen_context_data(5, 1000, 2, 3,
                                                                                                       FunctionType.GP_PRIOR,
                                                                                                       IvType.SCALE,
                                                                                                       iv_type_covariates=IvType.CONST,
                                                                                                       iid_contexts=True,
                                                                                                       partition_search=True,
                                                                                                       partition_Y=pi)

    Xc = np.array([Dc[c_i][:, [p for p in parents]] for c_i in range(5)])
    yc = np.array([Dc[c_i][:, target] for c_i in range(5)])
    plt.figure(figsize=(6, 4))
    for i in range(5):
        plt.scatter(Xc[i], yc[i])

    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title((
            str('%s context(s) w %s samples\n' % (5, 1000)) +
            'sampled from a GP under %s intervention(s)' %(len(pi)-1)))
    plt.xlim([-5, 5])
    plt.show()