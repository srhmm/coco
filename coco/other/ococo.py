

class OCoCo:
    """
    Tests whether relationships X_i -> X_j between pairs of nodes are Confounded
        using pairwise Correlations between Contexts
    """

    def __init__(self, Dc, test, sampler=None):
        self.Dc = Dc
        self.n_c = len(Dc)
        self.n_nodes = Dc[0].shape[1]

        # Per tested variable pair, the partitions and mechanism shifts
        self.pair_pi = {}
        self.pair_shifts = {}

        # Per tested variable pair, shift counts and mutual information
        self.scores = {}
        self.test = test

        if sampler is None:
            self.sampler = Sampler()

    @staticmethod
    def _get_hash(pa_i, i, pa_j, j):
        return f'i_{str(i)} | pa_{str(pa_i)} -> j_{str(j)} | pa_{str(pa_j)}'

    def _get_entry(self, pa_i, i, pa_j, j, dic):
        hash_key = self._get_hash(pa_i, i, pa_j, j)
        assert dic.__contains__(hash_key)
        return dic[hash_key]

    def get_scores(self, pa_i, i, pa_j, j):
        return self._get_entry(pa_i, i, pa_j, j, self.scores)

    def get_partitions(self, pa_i, i, pa_j, j):
        return self._get_entry(pa_i, i, pa_j, j, self.pair_pi)

    def score_bivariate(self, i, j, oracle_shifts_i=None, oracle_shifts_j=None,
                   oracle_partition_i=None, oracle_partition_j=None):
        # TODO choose appropriate test for marginal discrepancies, as one variable doesn't have parents ...
        pa_i = []
        pa_j = [i]
        self.score_edge(pa_i, i, pa_j, j, oracle_shifts_i, oracle_shifts_j,
                        oracle_partition_i, oracle_partition_j)
        pass

    def score_causal_dag(self, G, oracle_partitions=None):
        # TODO for each undirected edge in the MEC=pair of connected nodes, score edge
        for i, j in itertools.combinations(G.nodes(), 2):
            pa_i = list(G.predecessors(i))
            pa_j = list(G.predecessors(j))

            # TODO do indices match because dag counting starts from 1?
            pi_i, pi_j = None, None
            if oracle_partitions is not None:
                pi_i = oracle_partitions[i]
                pi_j = oracle_partitions[j]
            self.score_edge(pa_i, i, pa_j, j, pi_i, pi_j)
        raise (NotImplementedError())


    def score_edge(self, pa_i, i, pa_j, j, oracle_shifts_i=None, oracle_shifts_j=None,
                   oracle_map_i=None, oracle_map_j=None):

        ''' Bivariate case: Tests wether a causal edge X_i - X_j has independent or correlated partitions (confounding), using pa_i -> i to infer the partition of i (resp j)
        Also counts partition overlap, indicating anticausal direction

        :param pa_i: causal parents of i.
        :param i: node i.
        :param pa_j: causal parents of j
        :param j: node j.
        :param oracle_shifts_i: context partition (Xpai -> i) in the oracle case
        :param oracle_shifts_j: context partition (Xpaj -> j) in the oracle case

        :return: mutual information between partition vectors of Xi and Xj
        '''

        assert (i in pa_j) and not (j in pa_i)  # edge i->j

        if self.test.is_oracle():
            assert (oracle_map_i is not None and oracle_map_j is not None)

        hash_key = self._get_hash(pa_i, i, pa_j, j)

        if self.scores.__contains__(hash_key):
            return

        # Test each pair of contexts for distribution shift
        co_shifts_i = self._test_mechanism_shifts(i, pa_i, oracle_shifts_i)
        co_shifts_j = self._test_mechanism_shifts(j, pa_j, oracle_shifts_j)
        co_map_i = shifts_to_map(co_shifts_i, self.n_c)
        co_map_j = shifts_to_map(co_shifts_j, self.n_c)
        #-> todo do a safe conversion between co shifts and partitions in the non-oracle case. does it make sense to use singleton groups for each shifted context?

        if self.test.is_oracle():
            co_shifts_i, co_shifts_j = oracle_shifts_i, oracle_shifts_j
            co_map_i, co_map_j = oracle_map_i, oracle_map_j
            #co_map_i, co_map_j =shifts_to_map(co_shifts_i, self.n_c),shifts_to_map(co_shifts_j, self.n_c)
        #self.pair_shifts[hash_key] = (co_shifts_i, co_shifts_j)

        score_dict = self._score_partitions(co_shifts_i, co_shifts_j , co_map_i, co_map_j, set_size_i=1) # 1 parent
        self.scores[hash_key] = score_dict

    def _score_partitions(self, co_shifts_i, co_shifts_j, co_map_i, co_map_j, set_size_i):

        n_shifts_i = len(np.unique(co_shifts_i)) - 1
        n_shifts_j = len(np.unique(co_shifts_j)) - 1

        n_c = len(co_map_i)
        assert 0 <= n_shifts_i and n_shifts_i < n_c

        # Test whether the pairwise context conditional dependence vectors are correlated
        ##mi_shifts, ami_shifts, correction = self._mi_shifts(co_shifts_i, co_shifts_j) #can remove, well defined on partitions not shift vectors
        # NEW Test whether clustering labels are correlated -- do we need a correction
        # mi_pi, ami_pi,  _ = self._mi_partitions(co_map_i, co_map_j) #TODO mi and correction instd of ami?

        mi, ami, emi, h1, h2 = mutual_info_scores(co_map_i, co_map_j)

        cond_ent = h1 - mi
        cond_ent_adj = h1 - ami # TODO do we need to adjust h somehow here?
        cond_ent_rev = h2 - mi
        cond_ent_rev_adj = h2 - ami

        cf, _, _ = test_confounded_Z_mi_sampling(co_map_i, co_map_j, self.sampler)
        cf2, _, _ = test_confounded_Z_ami_sampling(co_map_i, co_map_j, self.sampler)

        #emp_h1, emp_h2, emp_emi, comparison_emi = self._empiricals(n_c, n_shifts_i, n_shifts_j, set_size_i)


        #_, emp_emi_anypi, _ = exp_entropy_empirical_anypi(emp_sampling_repeats, n_c)

        #emp_eent = emp_h1 - emp_emi  # TODO emi or emp_emi? emi - emp_expected_entropy
        # TODO closed form eent (expected cond entropy) based on hypergeom distribution as in emi?


        # Decisions based on cutoffs
        caus = 1
        if np.isclose(cond_ent, 0):
            caus = 0

        caus2 = 0
        #if cond_ent >= emp_eent:
        #    caus2 = 1

        # TODO thresholds.
        #cf2 = 0
        #if mi >= emi: #TODO mi >= emp_emi:  # mi > emi:
        #    cf2 = 1

        #cf = 0
        #if ami >= emi: #np.isclose(mi, emp_emi_anypi):
        #    cf = 1

        # TODO another way is to generate more "global" emi cutoffs by generating causal/confounded examples, i.e. generate partition vectors
        # TODO and take np.min if confounded or anticausal and then find a cutoff. (or is this exactly the same thing?).

        # Environments where the mechanisms of two variables shift jointly
        joint_shifts = len(np.where(np.array(co_shifts_i) + np.array(co_shifts_j) == 2)[0])
        # Environments where one shifts, the other does not; relevant for causal direction determination. e.g. if X->Y is correct, but we test Y->X, there should be NO indep shifts as all shifts in Y influence X as well. (holds still when both are confounded through Z).
        indep_shifts = len(np.where(np.array(co_shifts_i) - np.array(co_shifts_j) == 1)[0])

        mss_shifts = len(np.where(co_shifts_j==1)[0]) #TODO double check


        score_dict = {
            'pa_ent': h1,#TODO pass down
            'ch_ent': h2,#TODO pass down
            #'pa_eent': emp_h1,#TODO pass down
            #'ch_eent': emp_h2,#TODO pass down
            'ent': cond_ent,
            'aent': cond_ent_adj,
            'emp_eent': 0, #emp_eent,
            'ent_rev': cond_ent_rev,
            'aent_rev': cond_ent_rev_adj,
            'mi': mi,
            'ami': ami,
            'emi': emi,
            'emp_emi': 0, #emp_emi,
            #DECISIONS: all of these are based only on the partition for i|pa_i and j|pa_j (where i->j and i in pa_j), better return a score for this as in mss. also, mss only uses j|pa_j and not i|pa_i, what does this imply? also, how to generalize to an input SET w multiple i's??
            # Decision is-confounded
            # -based on mi and emp-emi
            'exp_cf': cf,
            # -based on mi and emi
            'exp_cf2': cf2,
            # - based on conditional entropy: todo nonzero but smaller than for causal cases...? can prob also directly look at mi instd of doing this.
            # -based on counts todo empirical threshold for joint_shifts (not quite mss!)
            #'count_cf' : count_cf,
            # Decision is-causal
            # -based on ent and zero
            'exp_caus': caus,
            # -based on ent and emp-eent
            'exp_caus2': caus2,
            # -based on counts
            #'count_caus' : count_caus, todo empirical threshold for indep_shifts
            # Shift counts
            'indep_shifts': indep_shifts,
            'joint_shifts': joint_shifts,
            'mss_shifts': mss_shifts #this is for MSS, it basically adds up all of these conditionals for all edges in a MEC.
            # tests,consider removing
            #'mishifts': mi_shifts,
            #'amishifts': ami_shifts,
            #'ent_shifts': cond_ent_shifts,
            #'ent_rev_shifts': cond_ent_rev_shifts,
        }
        return score_dict

    def compare_directions(self, i, j, pa_i, pa_j, rev_pa_j, rev_pa_i):

        scores_causal = self.get_scores(pa_i, i, pa_j, j)
        scores_anticausal = self.get_scores(rev_pa_j, j, rev_pa_i, i)

        comp_ent_caus = 0
        if scores_causal['ent'] > scores_anticausal['ent']:
            comp_ent_caus = 1
        comp_aent_caus = 0
        if scores_causal['aent'] > scores_anticausal['aent']:
            comp_aent_caus = 1
        comp_bothent_caus = 0
        if scores_causal['ent'] + scores_causal['mi'] > scores_anticausal['ent']+ scores_anticausal['mi'] :
            comp_bothent_caus = 1
        comp_bothaent_caus = 0
        if scores_causal['aent'] + scores_causal['ami'] > scores_anticausal['aent']+ scores_anticausal['ami'] :
            comp_bothaent_caus = 1
        comp_paent_caus = 0
        if scores_causal['pa_ent'] < scores_anticausal['ch_ent'] :
            comp_paent_caus = 1
        comp_chent_caus = 0
        if scores_causal['ch_ent'] < scores_anticausal['pa_ent']:
            comp_chent_caus = 1

        comp_entami_caus = 0
        if scores_causal['ent'] + scores_causal['ami']  < scores_anticausal['ent'] + scores_anticausal['ami']: # TODO ent used before
            comp_entami_caus = 1

        comp_ent_rev_caus = 0
        if scores_causal['ent_rev'] > scores_anticausal['ent_rev']:
            comp_ent_rev_caus = 1

        comp_mi_caus = 0
        if scores_causal['mi'] - scores_causal['emp_emi'] < scores_anticausal['mi']- scores_anticausal['emp_emi'] :
            comp_mi_caus = 1
        comp_ami_caus = 0
        if scores_causal['ami']- scores_causal['emp_emi']  < scores_anticausal['ami']- scores_anticausal['emp_emi'] :
            comp_ami_caus = 1

        mss_caus = 0
        if scores_causal['mss_shifts'] < scores_anticausal['mss_shifts']:
            mss_caus = 1

        joint_caus = 0
        if scores_causal['joint_shifts'] < scores_anticausal['joint_shifts']:
            joint_caus = 1

        #shifts_caus = 0
        #if scores_causal['joint_shifts'] -scores_causal['indep_shifts'] < scores_anticausal['joint_shifts'] -scores_anticausal['indep_shifts'] :
        #    shifts_caus = 1

        indep_caus = 1 #TODO this thing could show us whether counting indep shifts is of benefit for causal direction determination, as we can compare it more directly to mss
        if scores_causal['indep_shifts'] == 0: #TODO emp estim.
            indep_caus = 0

        # TODO joint shifts w emp estimate, as well as indep or joint shift emp estimates for confounding decision

        score_dict = {
            #cond entropies
            'comp_ent_caus': comp_ent_caus,
            'comp_aent_caus': comp_aent_caus,
            'comp_ent_rev_caus': comp_ent_rev_caus,
            'comp_bothent_caus': comp_bothent_caus,
            'comp_bothaent_caus': comp_bothaent_caus,
            #marginal entropies
            'comp_entpa_caus': comp_paent_caus,
            'comp_entch_caus': comp_chent_caus,
            #combination entropies
            'comp_entami_caus': comp_entami_caus,
            # mutual inf
            'comp_mi_caus' : comp_mi_caus,
            'comp_ami_caus' : comp_ami_caus,
            # counts
            'mss_caus': mss_caus,
            'indep_shifts_caus': indep_caus,
            'joint_shifts_caus': joint_caus
                }
        return score_dict

    def _test_mechanism_shifts(self, i, pa_i, oracle_shifts_i):
        ''' Context Groups: Group membership of all pairs of contexts

        :param pa_i: causal parents of i
        :param i: target node i
        :param oracle_shifts_i: partition for pa_i -> i if known
        :return:
            A. Indicator vector: CO = [1 if Pi(ck)!=Pi(cl) else 0 for ck for cl > ck], where Pi is known (oracle) or estimated for node i given pa_i.
            B. Soft Score: CO = [pval(P (i | pa_i, ck) =|= P (i | pa_i, cl)) for ck for cl > ck].
        '''

        if self.test.is_oracle():
            return oracle_shifts_i

            # Find a partition of the contexts, convert it to pairwise indicator vectors:
        elif not self.test.is_pairwise():  # self.test == CoTestType.GRP_GP:
            partition = co_groups(self.Dc, i, pa_i, self.test)
            mp = partition_to_map(partition)  # group_map(partition, self.n_c)
            shifts = map_to_shifts(mp)  # [1 if i != j else 0 for k, i in enumerate(map) for j in map[k + 1:]]
            return shifts

        # Test distributions between pairs of contexts for equality, use pvalues or indicators:
        elif self.test.is_pairwise():
            shifts = co_pair_grouped(self.Dc, i, pa_i, self.test)
            return shifts
        else:
            raise ValueError("Unknown CoTestType")

    def _empiricals(self, n_c, n_shifts_i, n_shifts_j, set_size_i, emp_sampling_repeats=20):
        return 0,0,0,0 #exp_entropy_empirical(emp_sampling_repeats, n_c, n_shifts_i, n_shifts_j) #TODO, set_size_i)
        # TODO store these f lookup, as well as adjust them for the parent set size somehow.

