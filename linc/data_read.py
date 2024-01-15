import pandas as pd

import networkx as nx

def read_sachs():
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
    return Dc


def read_reged():
    #TODO train data?
    df0 = pd.read_csv(r'/data_reged/reged0_train.data', ' ')
    df1 = pd.read_csv(r'/data_reged/reged1_train.data', ' ')
    df2 = pd.read_csv(r'/data_reged/reged2_train.data', ' ')

    Dc = [df0, df1, df2]
    Dc = [pd.DataFrame.to_numpy(di) for di in Dc]
    return Dc


def reged_network_A():

    true_edges =  pd.read_csv(r'/data_reged/reged_truth.txt', sep='\t')
    edge_list = true_edges.values.tolist()
    G = nx.DiGraph()
    for i in range(len(edge_list)):
        G.add_edge(edge_list[i][0], edge_list[i][1])

#22  29  65 711 995
#G.edges(22)
#[(22, 65)] [(65, 995)] [(995, 711)] [(711, 29)] [29, --]

    Dc = read_reged()
    Xc = [Dc[i][:, [22, 29, 65, 711, 995]] for i in range(3)]
    truth =  {(0,2), (2,4), (4, 3), (3, 1)}
    trueG = [[0 for _ in range(5)] for _ in range(5)]
    for (i, j) in truth:
        trueG[i][j] = 1
    index = ['22', '29', '65', '711', '995']
    return Xc, trueG, index
