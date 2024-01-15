class Vsn:
    def __init__(self, rff,
                 mdl_gain = True, #in tree search
                 rff_features = 200, #not used downstream yet?
                 ilp_in_tree_search=True, #for PiTree
                 ilp = True, ilp_partitions = False, # for PiDAG
                 #arguments not needed anymore:
                 regression_per_group=False,
                 regression_per_pair=False,
                 #structure=False,
                 subsample_size=None,
                 pi_search=True, #for rff which lik
                 clus=False
                 ):
        self.clustering=clus
        self.rff = rff
        self.rff_features = rff_features
        self.mdl_gain=  mdl_gain
        self.regression_per_group=  regression_per_group
        self.regression_per_pair=  regression_per_pair
        self.structure = False #not used
        #self.normalize = normalize
        self.subsample_size = subsample_size
        self.pi_search=pi_search

        self.ilp_in_tree_search=  ilp_in_tree_search
        self.ilp, self.ilp_partitions = ilp, ilp_partitions
        self.vario_in_tree_search = False #vario score not meant for full DAG discovery
        self.emp = False #empirical Vario score


