
def hyp_p(N, K, n, k):
    '''
    :param N: sample size
    :param K:
    :param n:
    :param k:
    :return:
    '''
    # K=bj, n =ai, k=nij
    return (comb(K, k) * comb(N - k, n - k)) / comb(N, n)

def sum_hyp_p(S, K, n, k):
    return sum([hyp_p(s, K, n, k) for s in S])

def emi(mA, mB, mC, constrained: bool):
    return 0
    contingency = contingency_matrix(mA, mB,  sparse=True)
    contingency = contingency.astype(np.float64, copy=False)
    contingency_AC = contingency_matrix(mA, mC,  sparse=True)
    contingency_AC = contingency_AC.astype(np.float64, copy=False)
    a = np.ravel(contingency.sum(axis=1).astype(np.int64, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int64, copy=False))
    c = np.ravel(contingency_AC.sum(axis=0).astype(np.int64, copy=False))
    return emi_abc(a, b, c, constrained)

def emi_mv(mA, mB, mC, constrained : bool):
    return 0
    contingency = contingency_matrix(mA, mB,  sparse=True)
    contingency = contingency.astype(np.float64, copy=False)
    contingency_AC = contingency_matrix(mA, mC,  sparse=True)
    contingency_AC = contingency_AC.astype(np.float64, copy=False)
    a = np.ravel(contingency.sum(axis=1).astype(np.int64, copy=False))
    b = np.ravel(contingency.sum(axis=0).astype(np.int64, copy=False))
    c = np.ravel(contingency_AC.sum(axis=0).astype(np.int64, copy=False))
    return emi_mv_abc(a, b, c, constrained)

def emi_abc(a, b, c, constrained):
    N =sum(a)
    assert sum(b)==N and sum(c)==N
    #A, B,
        #confounder_sizes):
    emi = 0
    #contingency = contingency_matrix(A, B,  sparse=True)
    #contingency = contingency.astype(np.float64, copy=False)
    #contingency_AC = contingency_matrix(A, C,  sparse=True)
    #contingency_AC = contingency.astype(np.float64, copy=False)
    #n_rows, n_cols = contingency.shape
    #a = np.ravel(contingency.sum(axis=1).astype(np.int64, copy=False))
    #b = np.ravel(contingency.sum(axis=0).astype(np.int64, copy=False))
    #c = np.ravel(contingency_AC.sum(axis=0).astype(np.int64, copy=False))
    #C = confounder_sizes
    #N = sum(C)
    n_rows, n_cols = len(a), len(b)
    a_view = a
    b_view = b

    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    nijs[0] = 1
    term1 = nijs / N


    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    log_a = np.log(a)
    log_b = np.log(b)
    # term2 uses log(N * nij) = log(N) + log(nij)
    log_Nnij = np.log(N) + np.log(nijs)

    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    #gln_Na = gammaln(N - a + 1)
    #gln_Nb = gammaln(N - b + 1)
    #gln_Nnij = gammaln(nijs + 1) + gammaln(N + 1)

    # emi itself is a summation over the various values.
    for i in range(n_rows):
        for j in range(n_cols):
            start = max(1, a_view[i] - N + b_view[j])
            end = min(a_view[i], b_view[j]) + 1
            for nij in range(start, end):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                term3 = 0
                for ck in c:
                    if (ck < a[i]) or (ck < b[j]):
                        continue
                    # Numerators are positive, denominators are negative.

                    gln_Na = gammaln(ck - a + 1)
                    gln_Nb = gammaln(ck - b + 1)
                    gln_Nnij = gammaln(nijs + 1) + gammaln(ck + 1)

                    gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                           - gln_Nnij[nij] - lgamma(a_view[i] - nij + 1)
                           - lgamma(b_view[j] - nij + 1)
                           - lgamma(N - a_view[i] - b_view[j] + nij + 1))
                    #TODO term3 += exp(gln)
                    if constrained:
                        term3 += 1/perm_p(ck, c) * hyp_p(ck, b[j], a[i], nij) #ck/sum(c)?
                    else:
                        term3 += hyp_p(N, b[j], a[i], nij) #exp(gln)
                        break

                if constrained:
                    emi += (term1[nij] * term2 *1/N* term3)
                else:
                    emi += (term1[nij] * term2 * term3)

    return emi


def perm_p(ck, c):
    return len(c)



'''def confound_partition2(Astar, C): 
    part_z = partition_to_vector(C)
    part_A = partition_to_vector(Astar)
    vec_z = map_to_shifts(part_z)
    vec_A = map_to_shifts(part_A)
    vec_A = np.maximum(vec_A, vec_z)
    N = len(part_A)
    A_labels = shifts_to_map(vec_A, N) #as labels!
    C_labels = part_z
    return A_labels, C_labels 
    #starting from A, per group: split into maximal subsets that C allows
    #starting from C, use subgroups according to A

    mAstar = partition_to_map(Astar)
    mC = partition_to_map(C)
    mA = mC

    curidx_mA = -1
    for i in range(max(mC)+1):
        curidx_mA += 1
        #0 elements in C inherit the numbers of these elements in Astar
        elem_C = np.where(np.array(mC) == i)[0]
        #print(i, elem_C)
        skip_first = False
        for j in range(max(mAstar)+1):
            #Search for groups w intersecting elements
            elem_A = np.where(np.array(mAstar) == j)[0]


            if len(elem_A)==0 or len(np.intersect1d(elem_C, elem_A))==0:
                #print("\t", j, elem_A, "{}")
                continue
            #else we have a subset like elem_A=[3]  and elem_C = [1,2,3] and want 3 separate (we can skip the first of these, or skip altogether if there is only one and elem_A=elem_C)
            if not skip_first:
                skip_first = True
                #print("\t", j, elem_A, "skip")
                continue
            #if elem_A is a subset of elem_C: create this subset also in mA

            mA = [ma if ma <= curidx_mA else ma +1 for ma in mA]
            #print("\t", j, elem_A, "prev", mA)
            curidx_mA += 1
            mA = [curidx_mA if idx in (np.intersect1d(elem_C, elem_A)) else ma for (idx, ma) in enumerate(mA)]

            #print("\t up", mA)
    return mA
'''
def pi_valid_splits(N, k):
    splits = pi_enum(N, permute=False)
    valid_splits = []
    for split in splits:
        if (len(split)) == k:
            valid_splits.append([len(s) for s in split])
    return valid_splits



def emi_mv_abc(a, b, c, constrained : bool):
    N =sum(a)
    assert sum(b)==N and sum(c)==N
    emi = 0
    a_view = a
    b_view = b
    c_view = c

    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    miks = np.arange(0, max(np.max(a), np.max(c)) + 1, dtype='float')
    ljks = np.arange(0, max(np.max(b), np.max(c)) + 1, dtype='float')
    nijs[0], miks[0], ljks[0] = 1,1,1
    term1 = nijs / N

    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    log_a = np.log(a)
    log_b = np.log(b)
    # term2 uses log(N * nij) = log(N) + log(nij)
    log_Nnij = np.log(N) + np.log(nijs)

    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    #gln_a = gammaln(a + 1)
    #gln_b = gammaln(b + 1)
    #gln_Na = gammaln(N - a + 1)
    #gln_Nb = gammaln(N - b + 1)
    #gln_Nnij = gammaln(nijs + 1) + gammaln(N + 1)

    # emi itself is a summation over the various values.
    for i in range(len(a)):
        for j in range(len(b)):
            start = max(1, a_view[i] - N + b_view[j])
            end = min(a_view[i], b_view[j]) + 1
            for nij in range(start, end):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                term3 = 0
                for k in range(len(c)):
                    subterm3 = 0
                    st = max(1, a_view[i] - N + c_view[k])
                    en = min(a_view[i], c_view[k]) + 1
                    for mik in range(st, en):
                        if constrained and mik != a[i]:
                            continue
                        stt = max(1, b_view[j] - N + c_view[k])
                        enn = min(b_view[j], c_view[k]) + 1
                        for ljk in range(stt, enn):
                            if constrained and ljk != b[j]:
                                continue
                            for rijk in range(max(start, st, stt), min(end, en, enn)):
                                if constrained:
                                    h = hyp_p(N=c[k], K=b[j], n=a[i], k=nij) * hyp_p(N =N, K=c[k], n =b[j], k=ljk) * hyp_p(N =N, K=c[k], n =a[i], k=mik) # * hyp_p(N =N, K=a[i], n= c[k], k=mik)# 1/ math.comb(N, rijk) # *1/(len(range(st, en))*len(range(stt, enn)))
                                    #h = hyp_mv(nij, mik, ljk, rijk, a[i], b[j], c[k], N) #*1/(len(range(max(start, st, stt), min(end, en, enn)))) # (1/math.comb(N,c[k]))
                                else:
                                    h = hyp_mv(nij, mik, ljk, rijk, a[i], b[j], c[k], N)
                                subterm3 += h
                    if constrained:
                        term3 += subterm3#*(1/len(range(len(c)))) #TODO needed?
                    else:
                        term3 += subterm3 #*(1/len(range(len(c)))) #TODO needed?

                if constrained:
                    emi += (term1[nij] * term2 * term3)
                else:
                    emi += (term1[nij] * term2  * term3)


                #if  term2 < 0:
                #    print(i,j, nij, a[i], b[j],  N)
    return emi

def hyp_mv (nij, mik, ljk, rijk, ai, bj, ck, N):
    n = nij+mik+ljk-2*rijk
    K0 = ai
    k0 = nij
    K1 = bj-nij
    k1 = ljk-rijk
    K2 = ck-ljk
    k2 = mik-rijk
    if (k0<0 or k1<0 or k2<0 or k0>K0 or k1>K1 or k2>K2): return 0
    return (1/math.comb(N, n))*math.comb(K0, k0)*math.comb(K1, k1)*math.comb(K2, k2)

