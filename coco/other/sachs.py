import networkx as nx
import numpy as np
import pandas as pd

from coco.co_test_type import CoShiftTestType, CoCoTestType
from coco.mi_sampling import Sampler
from experiments.exp_coco.results_coco import MethodType
from linc.data_read import read_sachs
from sparse_shift import dag2cpdag, cpdag2dags

def read():
    df1 = pd.read_excel(r'~/impls/linc/data_sachs/1.cd3cd28.xls')
    df2 = pd.read_excel(r'~/impls/linc/data_sachs/2.cd3cd28icam2.xls')
    df3 = pd.read_excel(r'~/impls/linc/data_sachs/3.cd3cd28+aktinhib.xls')
    df4 = pd.read_excel(r'~/impls/linc/data_sachs/4. cd3cd28+g0076.xls')
    df5 = pd.read_excel(r'~/impls/linc/data_sachs/5. cd3cd28+psitect.xls')
    df6 = pd.read_excel(r'~/impls/linc/data_sachs/6. cd3cd28+u0126.xls')
    df7 = pd.read_excel(r'~/impls/linc/data_sachs/7. cd3cd28+ly.xls')
    df8 = pd.read_excel(r'~/impls/linc/data_sachs/8. pma.xls')
   # not used  df9 = pd.read_excel(r'/data_sachs/9. b2camp.xls')

    Dc = [df1, df2, df3, df4, df5, df6, df7, df8]
    Dc = [pd.DataFrame.to_numpy(di) for di in Dc]

    return Dc, df1.columns

def run_coco_sachs():

    #standard params for coco
    KNOWN_DAG = False
    SHIFT_TEST = CoShiftTestType.PI_KCI
    CONFOUNDING_TEST = CoCoTestType.MI_ZTEST
    ALPHA_SHIFT_TEST =0.05


    Dc, nms = read()

    # CPDAG
    dag = np.zeros((11, 11))
    dag[2, np.asarray([3, 4])] = 1
    dag[4, 3] = 1
    dag[8, np.asarray([10, 7, 0, 1, 9])] = 1
    dag[7, np.asarray([0, 1, 5, 6, 9, 10])] = 1
    dag[0, 1] = 1
    dag[1, 5] = 1
    dag[5, 6] = 1

    #DAG as in nocadillac
    dag = np.zeros((11, 11))

    dag[8, np.asarray([10, 7, 0, 9])] = 1
    dag[2, np.asarray([3, 7])] = 1
    dag[7, 5] = 1
    dag[0, 1] = 1
    dag[3, 8] = 1
    dag[4, np.asarray([10, 7, 9, 1, 6])] = 1
    dag[1, 5] = 1
    dag[5, np.asarray([6, 3])] = 1

    #true_dag = dag
    #true_cpdag = dag2cpdag(true_dag)
    #mec_size = len(cpdag2dags(true_cpdag))
    #total_edges = np.sum(true_dag)
    #unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2
    nodes_true = range(len(dag))

    n_components = 1

    results = [0 for _ in nodes_true]

    Dc = [
        np.log(
            pd.read_csv(f'../linc/cytometry/dataset_{i}.csv')
        ) for i in range(1, 10)
    ]

    Dc = [np.array(X[:707]) for X in Dc]
    Dc = np.array(Dc)
    s_result_fci_jci = "FCI_JCI\n"
    s_result_coco = "COCO\n"
    for confounder_i, confounder in [8]: # enumerate(nodes_true):
        nodes = [n for n in nodes_true if n != confounder]
        G= nx.DiGraph([])
        edges = []
        G_observed= nx.DiGraph([])
        edges_observed = []
        print("\nCONFOUNDER:", nms[confounder])
        s_result_fci_jci += "\nCONFOUNDER:" + nms[confounder]
        s_result_coco += "\nCONFOUNDER:" + nms[confounder]
        for ind, i in enumerate(nodes_true):
                for j in nodes_true:
                    if dag[i][j]==1:
                        edges.append((i, j))
                        if i!=confounder: # and j!=confounder:
                            print(f"E\t{i}.{nms[i]}->{j}.{nms[j]}")
                            s_result_coco += f"\nE\t{i}.{nms[i]}->{j}.{nms[j]}"
                            edges_observed.append((i, j))
                        else:
                            s_result_coco += f"\nE\t{i}.{nms[i]}->{j}.{nms[j]}"
                            print(f"C\t{i}.{nms[i]}->{j}.{nms[j]}")

        s_result_coco += "\n\n"
        G.add_nodes_from(nodes_true)
        G.add_edges_from(edges)
        G_observed.add_nodes_from(nodes)
        G_observed.add_edges_from(edges_observed)
        nms_observed  = [nm  for i, nm in enumerate(nms) if i!= confounder]
        Dobs = np.array([Dc[ci][:, nodes] for ci in range(len(Dc))])
        #
        coco = CoCo(G_observed, Dc, CONFOUNDING_TEST,  SHIFT_TEST,
                    Sampler(), 1, None, nms, alpha_shift_test=ALPHA_SHIFT_TEST, known_mec=KNOWN_DAG)

        coco._estimated_graph_cuts(1)
        results[confounder_i] = coco
        if len(coco.estimated_cuts):
            print("\tCUTS:", coco.estimated_cuts, [nm[j] for j in results[i].estimated_cuts[0]])
            nm = nms
            if (len(results[i].maps_estimated) == 10):
                nm = nms_observed
            s_result_coco += f"\n{coco.estimated_cuts}" + str([nm[j] for j in results[i].estimated_cuts[0]])
        #for n_i, n_j in coco.G_estimated.edges():
        #    print(f"{nms[n_i]}->{nms[n_j]}")# -> provide the true DAG.

        #fci = FCI_JCI(Dobs, G_observed, G, None,   independence_test='fisherz',
        #                                                          method=MethodType.FCI_JCI)
        #for i in range(len(fcii.G_estim_bg.graph)):
        #    for j in range(len(fcii.G_estim_bg.graph)):
        #        if fci.G_estim_bg.graph[i][j] == 1 and fci.G_estim_bg.graph[j][i] == 1:
        #            print(i, j, nms_observed[i], nms_observed[j])
        #            s_result_fci_jci += f"\n\t{i}. {nms_observed[i]} <-> {j}. {nms_observed[j]} by {nms[confounder]}"


        print()
        #praf, pmek, pip3, p38, pka, jnk, pkc

        for i in range(len(results)):
            confounder = i
            print(f"CFD: {i}. {nms[i]}")
            nms_observed = [nm for i, nm in enumerate(nms) if i != confounder]
            nm = nms
            if (len(results[i].maps_estimated) == 10):
                nm = nms_observed
                print("\tVARS:", nms_observed)
            else:
                print("\t", [v for v in nms])
            results[i]._estimated_graph_cuts(1)
            if len(results[i].estimated_cuts):
                print("\tCUTS:", results[i].estimated_cuts,
                      [nm[j] for j in results[i].estimated_cuts[0]])  # , nms[[j for j in results[i].estimated_cuts][0]])

        #TODO: may want to consider resolving disagreements in the partitions in favor of invariance, instd. of changes.
        #TODO correction 0.05/len(nms_observed) necessary??

        #print("\nRESULT:", nms[confounder])
        #for ind, i in enumerate(nodes_true):
        #        for j in nodes_true:
        #            if dag[i][j]==1:
        #                if i!=confounder:
         #                   print(f"E: {(i)}. {nms[i]} -> {(j)}. {nms[j]}")
         #               else:
         #                   print(f"C: {(i)}. {nms[i]} -> {(j)}. {nms[j]}")

                    #obs.praf -> pmek
                    #obs.pmek -> p44 / 42
                    #obs.plcg -> PIP2
                    #obs.plcg -> PIP3
                    #obs.p44 / 42 -> pakts473
                    #obs.PKA -> P38
                    #obs.PKA -> pjnk
                    #hidden.PKC -> P38
                    #hidden.PKC -> pjnk

'''                    
obs. praf -> pmek [1, 1, 1, 1, 1, 1, 1, 1, 0]
obs. pmek -> p44/42 [0, 0, 0, 0, 0, 1, 0, 0, 0]
obs. plcg -> PIP2 [1, 0, 1, 1, 1, 1, 1, 1, 0]
obs. plcg -> PIP3 [1, 1, 1, 1, 1, 1, 1, 1, 0]
obs. p44/42 -> pakts473 [1, 1, 1, 1, 1, 1, 1, 1, 0]
obs. PKA -> P38 [1, 1, 1, 1, 1, 1, 1, 1, 0]
obs. PKA -> pjnk [1, 1, 1, 1, 1, 1, 1, 1, 0]
hidden. PKC -> P38 [1, 1, 1, 1, 1, 1, 1, 1, 0]
hidden. PKC -> pjnk [1, 1, 1, 1, 1, 1, 1, 1, 0]

EDGE
                    Index([], dtype='object') -> praf 	[1, 1, 1, 1, 1, 1, 1, 1, 0]
                    change 	(0, 1), 3.1518305079503973e-09
                    change 	(0, 2), 0.0025418622501905013
                    change 	(0, 3), 5.969089342650588e-195
                    change 	(0, 4), 0.00453020783317471
                    change 	(0, 5), 5.559981853403384e-144
                    change 	(0, 6), 0.0025418622501905013
                    change 	(0, 7), 3.1126519351269816e-78
                    change 	(0, 8), 3.670191167692012e-29
                    change 	(1, 2), 2.9939755034170543e-07
                    change 	(1, 3), 1.508977367709944e-178
                    change 	(1, 4), 1.8061897807593373e-15
                    change 	(1, 5), 2.35283645982829e-131
                    change 	(1, 6), 2.9939755034170543e-07
                    change 	(1, 7), 1.7275385446491661e-71
                    change 	(1, 8), 7.511467299259934e-15
                    change 	(2, 3), 4.787176764533539e-188
                    change 	(2, 4), 0.00048684193236064416
                    change 	(2, 5), 3.3080549936730787e-138
                    invariant 	(2, 6), 1.0
                    change 	(2, 7), 5.854623564650958e-61
                    change 	(2, 8), 5.107492894555755e-23
                    change 	(3, 4), 6.189623098637688e-190
                    change 	(3, 5), 1.5060167996579037e-07
                    change 	(3, 6), 4.787176764620002e-188
                    change 	(3, 7), 2.9050159076462886e-241
                    change 	(3, 8), 3.579811450691048e-221
                    change 	(4, 5), 1.3743635238696702e-138
                    change 	(4, 6), 0.00048684193236071924
                    change 	(4, 7), 1.7644928937042522e-79
                    change 	(4, 8), 8.534200470744436e-39
                    change 	(5, 6), 3.308054993627805e-138
                    change 	(5, 7), 3.1350203735107635e-201
                    change 	(5, 8), 9.131850934855449e-174
                    change 	(6, 7), 5.854623564650958e-61
                    change 	(6, 8), 5.107492894555755e-23
                    change 	(7, 8), 1.7331815225444544e-40
                ED
                E Index(['praf'], dtype='object') -> pmek 	 [1, 1, 1, 1, 1, 1, 1, 1, 0]
                change 	(0, 1), 0.0
                invariant 	(0, 2), 0.21313380886846645
                change 	(0, 3), 0.0
                invariant 	(0, 4), 0.26558904259711325
                change 	(0, 5), 0.0
                invariant 	(0, 6), 0.21313380886846645
                change 	(0, 7), 0.0
                change 	(0, 8), 0.0
                change 	(1, 2), 0.0
                change 	(1, 3), 0.0
                change 	(1, 4), 0.0
                change 	(1, 5), 0.0
                change 	(1, 6), 0.0
                change 	(1, 7), 1.1102230246251565e-15
                change 	(1, 8), 0.0
                change 	(2, 3), 0.0
                invariant 	(2, 4), 0.6391951192304651
                change 	(2, 5), 0.0
                invariant 	(2, 6), 1.0
                change 	(2, 7), 0.0
                change 	(2, 8), 0.0
                change 	(3, 4), 0.0
                change 	(3, 5), 3.2085445411667024e-13
                change 	(3, 6), 0.0
                change 	(3, 7), 0.0
                change 	(3, 8), 0.0
                change 	(4, 5), 0.0
                invariant 	(4, 6), 0.6391951192231281
                change 	(4, 7), 0.0
                change 	(4, 8), 0.0
                change 	(5, 6), 0.0
                change 	(5, 7), 0.0
                change 	(5, 8), 0.0
                change 	(6, 7), 0.0
                change 	(6, 8), 0.0
                change 	(7, 8), 0.0
            ED
            E Index([], dtype='object') -> plcg 	 [1, 1, 1, 1, 1, 1, 1, 0, 0]
            change 	(0, 1), 5.698307090587265e-18
            change 	(0, 2), 3.37644406531039e-17
            change 	(0, 3), 4.5712100150449105e-265
            change 	(0, 4), 1.872188502622987e-125
            change 	(0, 5), 0.0024678570231891333
            change 	(0, 6), 5.0528225015518035e-30
            change 	(0, 7), 8.154203240308251e-15
            change 	(0, 8), 2.0970968158381487e-09
            change 	(1, 2), 3.196831842669666e-58
            change 	(1, 3), 9.392321764859176e-253
            change 	(1, 4), 6.803138664010326e-190
            change 	(1, 5), 6.047398394276224e-30
            change 	(1, 6), 2.20875464011837e-82
            invariant 	(1, 7), 0.02916108396019456
            change 	(1, 8), 0.004250741360303605
            change 	(2, 3), 2.2116680691314692e-265
            change 	(2, 4), 1.4760635665660954e-67
            change 	(2, 5), 2.3233570336264945e-11
            change 	(2, 6), 0.0019225024467496071
            change 	(2, 7), 9.732741520390673e-46
            change 	(2, 8), 2.443339735467372e-41
            change 	(3, 4), 4.0270485209824294e-288
            change 	(3, 5), 9.19339912638387e-266
            change 	(3, 6), 1.2217014812533283e-270
            change 	(3, 7), 3.0709508670332465e-248
            change 	(3, 8), 4.6671693441291196e-259
            change 	(4, 5), 1.145162556049264e-116
            change 	(4, 6), 1.1009925968444803e-53
            change 	(4, 7), 8.339996512192697e-164
            change 	(4, 8), 1.0972015007380662e-165
            change 	(5, 6), 1.046190016533116e-20
            change 	(5, 7), 1.4352335365854424e-25
            change 	(5, 8), 5.3324560271065255e-19
            change 	(6, 7), 4.60980152931529e-67
            change 	(6, 8), 8.53848478399302e-62
            invariant 	(7, 8), 0.014296301118991387
        ED
        E Index(['plcg'], dtype='object') -> PIP2 	 [1, 0, 1, 1, 1, 1, 1, 1, 0]
        change 	(0, 1), 6.002420782635909e-12
        change 	(0, 2), 0.0
        invariant 	(0, 3), 0.046481717926941
        change 	(0, 4), 0.0
        change 	(0, 5), 0.001263902043478482
        change 	(0, 6), 6.972565680385401e-08
        change 	(0, 7), 0.0
        change 	(0, 8), 0.0
        change 	(1, 2), 0.0
        change 	(1, 3), 0.0
        change 	(1, 4), 0.0
        change 	(1, 5), 0.0
        change 	(1, 6), 0.0
        change 	(1, 7), 0.0
        invariant 	(1, 8), 0.06500629515563683
        change 	(2, 3), 8.117007066488213e-11
        change 	(2, 4), 0.0
        change 	(2, 5), 0.0
        change 	(2, 6), 0.0
        change 	(2, 7), 0.0
        change 	(2, 8), 0.0
        invariant 	(3, 4), 0.013531539307258145
        invariant 	(3, 5), 0.02856963028503756
        invariant 	(3, 6), 0.016993512253297816
        change 	(3, 7), 0.0
        change 	(3, 8), 2.0106138975961585e-13
        change 	(4, 5), 0.0
        change 	(4, 6), 0.0
        change 	(4, 7), 0.0
        change 	(4, 8), 0.0
        change 	(5, 6), 0.0018116502625544761
        change 	(5, 7), 6.816769371198461e-14
        change 	(5, 8), 0.0
        change 	(6, 7), 6.047384815133228e-12
        change 	(6, 8), 0.0
        change 	(7, 8), 0.0
ED
E Index(['plcg'], dtype='object') -> PIP3 	 [1, 1, 1, 1, 1, 1, 1, 1, 0]
change 	(0, 1), 0.0
change 	(0, 2), 0.0
change 	(0, 3), 0.0012030917039956845
change 	(0, 4), 0.0
change 	(0, 5), 0.00038348458992654155
change 	(0, 6), 1.6064927166326015e-13
change 	(0, 7), 0.0
change 	(0, 8), 0.00031952975724358623
change 	(1, 2), 0.0
change 	(1, 3), 2.0905499553691698e-13
change 	(1, 4), 2.762131035005666e-08
change 	(1, 5), 0.0
change 	(1, 6), 2.220446049250313e-16
invariant 	(1, 7), 0.6820908081404364
change 	(1, 8), 0.0
change 	(2, 3), 1.6905199806682703e-07
change 	(2, 4), 0.0
change 	(2, 5), 0.0
change 	(2, 6), 0.0
change 	(2, 7), 0.0
change 	(2, 8), 0.0
invariant 	(3, 4), 0.3350671358195989
change 	(3, 5), 0.0005063674784698824
change 	(3, 6), 0.00017567512061700263
change 	(3, 7), 0.0
change 	(3, 8), 2.631228568361621e-14
change 	(4, 5), 4.374278717023117e-14
change 	(4, 6), 0.0
change 	(4, 7), 3.709004403606997e-07
change 	(4, 8), 5.651035195342047e-14
change 	(5, 6), 0.0001854915658128231
change 	(5, 7), 0.0
change 	(5, 8), 2.797060028036924e-09
change 	(6, 7), 5.551115123125783e-15
change 	(6, 8), 0.0
change 	(7, 8), 0.0
ED
E Index(['pmek'], dtype='object') -> p 4 4/42 	 [0, 0, 0, 0, 0, 1, 0, 0, 0]
change 	(0, 1), 4.5911547015276e-06
change 	(0, 2), 9.585881295370857e-05
change 	(0, 3), 7.409710955919024e-09
change 	(0, 4), 2.220446049250313e-16
change 	(0, 5), 0.0
change 	(0, 6), 0.004227950921723345
change 	(0, 7), 0.0
invariant 	(0, 8), 0.07265101223923343
invariant 	(1, 2), 0.038391524939747734
change 	(1, 3), 1.9095836023552692e-14
change 	(1, 4), 0.00010156179006182153
change 	(1, 5), 0.0
change 	(1, 6), 0.0
change 	(1, 7), 0.0
invariant 	(1, 8), 0.2950188323748273
change 	(2, 3), 8.379216817067103e-07
invariant 	(2, 4), 0.028278159582376228
change 	(2, 5), 0.0
change 	(2, 6), 1.1102230246251565e-16
change 	(2, 7), 0.0
invariant 	(2, 8), 0.12171025411053693
change 	(3, 4), 3.461972676643832e-06
change 	(3, 5), 0.0
change 	(3, 6), 2.0807022771407446e-10
change 	(3, 7), 0.0022060939345361907
invariant 	(3, 8), 0.03632648772350944
change 	(4, 5), 0.0
change 	(4, 6), 0.0
change 	(4, 7), 1.6089352072867769e-12
invariant 	(4, 8), 0.011799742705420058
change 	(5, 6), 0.0
change 	(5, 7), 0.0
change 	(5, 8), 2.7910961006960555e-05
change 	(6, 7), 0.0
invariant 	(6, 8), 0.05504533938959477
invariant 	(7, 8), 0.1054897458559132
ED
E Index(['p44/42'], dtype='object') -> pakts473 	 [1, 1, 1, 1, 1, 1, 1, 1, 0]
invariant 	(0, 1), 0.07106644634906933
invariant 	(0, 2), 0.2667194933079514
change 	(0, 3), 0.0
change 	(0, 4), 0.0
change 	(0, 5), 0.0
invariant 	(0, 6), 0.015086481440279798
change 	(0, 7), 5.139481686344816e-06
change 	(0, 8), 0.0
invariant 	(1, 2), 0.014890282873048721
change 	(1, 3), 0.0
change 	(1, 4), 0.0
change 	(1, 5), 0.0
change 	(1, 6), 1.8030263615465003e-08
change 	(1, 7), 0.003235983935443665
change 	(1, 8), 0.0
change 	(2, 3), 0.0
change 	(2, 4), 8.321121569565548e-13
change 	(2, 5), 0.0
change 	(2, 6), 3.9935967977022813e-07
change 	(2, 7), 1.2844332597516939e-08
change 	(2, 8), 0.0
change 	(3, 4), 0.0
change 	(3, 5), 9.89665360773273e-09
change 	(3, 6), 0.0
change 	(3, 7), 0.0
change 	(3, 8), 0.0
change 	(4, 5), 0.0
change 	(4, 6), 0.0
change 	(4, 7), 0.0
change 	(4, 8), 0.0
change 	(5, 6), 0.0
change 	(5, 7), 0.0
change 	(5, 8), 0.0
change 	(6, 7), 3.914646384828302e-12
change 	(6, 8), 0.0
change 	(7, 8), 0.0
ED
E Index([], dtype='object') -> PKA 	 [1, 1, 1, 1, 1, 1, 1, 1, 0]
change 	(0, 1), 1.3559731803504102e-34
change 	(0, 2), 1.2148970616519761e-06
change 	(0, 3), 2.4377598450100664e-275
change 	(0, 4), 0.00015015120945905766
change 	(0, 5), 5.95916236280164e-74
change 	(0, 6), 0.00028699319143239254
change 	(0, 7), 6.501450685178444e-19
change 	(0, 8), 6.165271386508908e-12
change 	(1, 2), 1.0411926846852529e-13
change 	(1, 3), 3.9643635002775575e-282
change 	(1, 4), 3.860214504386284e-22
change 	(1, 5), 4.851566865102629e-97
change 	(1, 6), 9.511815498989132e-21
change 	(1, 7), 6.973784414430994e-05
change 	(1, 8), 3.59418796786304e-61
change 	(2, 3), 1.2384467696421443e-284
change 	(2, 4), 0.0035894478420485784
change 	(2, 5), 5.478298516337264e-79
change 	(2, 6), 0.00266962446714309
change 	(2, 7), 9.988087972106293e-06
change 	(2, 8), 1.7524867661816321e-22
change 	(3, 4), 1.6903019892539923e-275
change 	(3, 5), 1.093446441992369e-230
change 	(3, 6), 2.1269919997597828e-279
change 	(3, 7), 3.7708025049435816e-289
change 	(3, 8), 2.9554008474101984e-282
change 	(4, 5), 1.0713214740215916e-61
change 	(4, 6), 5.751497576238103e-05
change 	(4, 7), 3.165152893660586e-12
change 	(4, 8), 3.145426211565691e-13
change 	(5, 6), 5.594288665228451e-80
change 	(5, 7), 1.1249557699470992e-92
change 	(5, 8), 2.240353595876129e-62
change 	(6, 7), 4.103229525786894e-09
change 	(6, 8), 1.105406062833028e-23
change 	(7, 8), 1.2777769433177604e-43
ED
E Index(['PKA'], dtype='object') -> P38 	 [1, 1, 1, 1, 1, 1, 1, 1, 0]
change 	(0, 1), 1.1102230246251565e-16
change 	(0, 2), 2.7075722774938527e-09
change 	(0, 3), 1.6197183594357512e-09
change 	(0, 4), 3.4416913763379853e-15
invariant 	(0, 5), 0.36828173259604924
invariant 	(0, 6), 0.041577298550218744
change 	(0, 7), 0.0036362253745461093
change 	(0, 8), 0.0
change 	(1, 2), 3.0937585826507075e-11
change 	(1, 3), 3.1837462166617314e-07
change 	(1, 4), 0.0
change 	(1, 5), 1.4385094782021213e-08
change 	(1, 6), 0.0
change 	(1, 7), 1.0714412468360024e-08
change 	(1, 8), 0.0
change 	(2, 3), 0.0002164168012462886
change 	(2, 4), 0.0
change 	(2, 5), 0.004934454515799103
change 	(2, 6), 5.939997382853335e-09
change 	(2, 7), 3.151023886260873e-11
change 	(2, 8), 0.0
change 	(3, 4), 6.053364476343859e-10
change 	(3, 5), 0.0
change 	(3, 6), 3.3562541634779564e-10
change 	(3, 7), 0.00026677922134854537
change 	(3, 8), 0.00015892652457050183
change 	(4, 5), 4.8117535622616003e-08
change 	(4, 6), 0.0
change 	(4, 7), 0.0
change 	(4, 8), 0.0
invariant 	(5, 6), 0.9001929344670734
change 	(5, 7), 0.0017137051546605164
change 	(5, 8), 0.0
change 	(6, 7), 1.3179956124376169e-05
change 	(6, 8), 0.0
change 	(7, 8), 0.0
ED
E Index(['PKA'], dtype='object') -> pjnk 	 [1, 1, 1, 1, 1, 1, 1, 1, 0]
change 	(0, 1), 0.0
change 	(0, 2), 0.0
change 	(0, 3), 1.9299718267973276e-08
invariant 	(0, 4), 0.018850571662444304
change 	(0, 5), 0.0
change 	(0, 6), 1.0246850590256429e-08
change 	(0, 7), 0.0
change 	(0, 8), 0.0
change 	(1, 2), 0.0
change 	(1, 3), 1.5237420563085635e-06
change 	(1, 4), 0.0
change 	(1, 5), 0.0
change 	(1, 6), 0.0
change 	(1, 7), 0.0
change 	(1, 8), 0.0
change 	(2, 3), 7.787071046283511e-05
change 	(2, 4), 0.0
invariant 	(2, 5), 0.005220303632057832
change 	(2, 6), 0.0
change 	(2, 7), 0.0
change 	(2, 8), 0.0
change 	(3, 4), 1.0016669826917735e-08
change 	(3, 5), 0.0
change 	(3, 6), 6.474710034787279e-08
change 	(3, 7), 0.0006464224755987624
change 	(3, 8), 2.326240644157096e-05
change 	(4, 5), 0.0
change 	(4, 6), 4.73160836067521e-08
change 	(4, 7), 0.0
change 	(4, 8), 0.0
change 	(5, 6), 0.0
change 	(5, 7), 0.0
change 	(5, 8), 0.0
change 	(6, 7), 0.0
change 	(6, 8), 0.0
change 	(7, 8), 0.0

'''
